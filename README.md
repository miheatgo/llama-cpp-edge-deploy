# Edge LLM Deployment: llama.cpp on RTX 4050 (6GB VRAM)

## 📖 项目简介
本项目旨在探索在边缘设备（Laptop with RTX 4050, 6GB VRAM）上进行大语言模型的极致部署与性能优化。
基于开源的 `llama.cpp` 源码，完成了从 C++ 源码编译、CUDA 硬件加速，到多并发 API 服务压测的全链路落地。

## 💻 硬件环境
- **GPU**: NVIDIA RTX 4050 Laptop (6GB VRAM)
- **CUDA Toolkit**: v12.6
- **系统**: Windows 10 / Visual Studio 2022 Build Tools
- **模型**: Qwen2-7B (4-bit 量化, 4.4GB GGUF格式)

## 🚀 核心工作流 (SOP)
1. 克服网络墙，拉取 `llama.cpp` 源码。
2. 解决 Windows 下 CUDA 工具集与 C++ 编译器绑定报错。
3. 使用 CMake 进行带 CUDA 加速的源码级编译。
4. 从 Ollama 的 blob 库中提取原始 `.gguf` 权重文件。
5. 使用 `llama-cli` 调参压榨单点性能极限。
6. 启动 `llama-server` 进行并发 API 压测。

## 📊 性能压测报告 (真实数据)

### 1. 算力与加速对比（单用户请求）
| 引擎/模式 | 读题速度 (Prompt) | 生成速度 (Generation) | 备注 |
| :--- | :--- | :--- | :--- |
| 纯 CPU 编译版 | 32.5 t/s | 11.6 t/s | 未开启 CUDA，全靠 CPU |
| **GPU 加速版 (极致调优)** | **607.9 t/s** | **39.2 t/s** | 使用 `-ngl 99` 全量加载至显存 |

### 2. 并发架构对比（3用户并发，生成长度 100 token）
| 推理引擎 | 调度机制 | 用户1耗时 | 用户2耗时 | 用户3耗时 | 平均速度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Ollama (商业封装) | 静态批处理 (排队) | 13.71s | 19.50s | 7.91s | ~14.5 t/s (极度不均) |
| **llama-server (手搓底层)** | **连续批处理 (Continuous Batching)** | **3.75s** | **3.75s** | **3.75s** | **26.6 t/s (完美均衡)** |

## 🧠 底层优化思考 (Why it works?)
1. **`-ngl` 的威力**：将模型所有层强制塞入 GPU 显存，彻底消除了 CPU 与 GPU 之间的 PCIe 数据搬运瓶颈，读题速度暴涨 3 倍。
2. **连续批处理的降维打击**：相比于 Ollama 的排队机制，`llama-server` 的 `n_slots` 机制让 GPU 可以在多个用户的请求间来回穿插，极大地降低了长尾延迟。

## 📂 仓库文件
- `benchmark.py`: 基于 Python `requests` 的并发压测脚本（适配 OpenAI 标准 API）。
