import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_layernorm_kernel(
    x_ptr, residual_ptr, weight_ptr, bias_ptr, out_ptr, 
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr
):
    # 获取当前处理的行号
    row_idx = tl.program_id(0)
    row_start = row_idx * stride
    
    # 计算当前行的偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # 1. 读入数据（访存：读两次）
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # 2. 残差相加
    x = x + residual
    
    # 3. 计算均值 Mean
    # TODO 1: 求当前行的均值 (用 tl.sum 除以 N)
    mean = tl.sum(x, axis=0) / N
    
    # 4. 计算方差 Variance
    diff = x - mean
    # TODO 2: 求方差 (求 diff 的平方和，再除以 N)
    var = tl.sum(diff * diff, axis=0) / N
    
    # 5. 归一化
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * rstd
    
    # 6. 乘权重加偏置 (仿射变换)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    # TODO 3: 计算最终输出
    out = x_norm * weight + bias
    
    # 7. 写回显存（访存：写一次）
    tl.store(out_ptr + row_start + col_offsets, out, mask=mask)

# 启动函数
def fused_add_layernorm(x, residual, weight, bias, eps=1e-5):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (M,)
    fused_add_layernorm_kernel[grid](
        x, residual, weight, bias, out,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    
    # 模拟大一点的矩阵：1024 行，2048 列（这就相当于一个 1024 token 的提示词，每个 token 特征维度 2048）
    M, N = 1024, 2048
    x = torch.randn((M, N), device='cuda')
    residual = torch.randn((M, N), device='cuda')
    weight = torch.randn((N,), device='cuda')
    bias = torch.randn((N,), device='cuda')
    eps = 1e-5
    
    # 1. 你的 Triton 融合版
    triton_out = fused_add_layernorm(x, residual, weight, bias, eps)
    
    # 2. PyTorch 官方版（分步执行）
    x_add = x + residual
    torch_out = torch.nn.functional.layer_norm(x_add, (N,), weight, bias, eps)
    
    print(f"最大误差: {torch.max(torch.abs(triton_out - torch_out)).item():.6f}")

    # 3. 测速对比（用 CUDA Event 比较）
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 测 PyTorch 速度
    start.record()
    for _ in range(100): # 循环100次，减少误差
        torch.nn.functional.layer_norm(x + residual, (N,), weight, bias, eps)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end)

    # 测 Triton 速度
    start.record()
    for _ in range(100):
        fused_add_layernorm(x, residual, weight, bias, eps)
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end)

    print(f"--- 性能对比 (运行 100 次) ---")
    print(f"PyTorch 耗时: {pytorch_time:.2f} ms")
    print(f"Triton  耗时: {triton_time:.2f} ms")
    print(f"提速: {pytorch_time / triton_time:.2f} 倍")