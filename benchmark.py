import requests
import time
import threading

# 注意：指向你 8080 端口的标准 API
URL = "http://127.0.0.1:8080/v1/chat/completions"
PROMPT = "请写一篇关于人工智能的短故事。"

def test_request(user_id):
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 100,  # 限制生成 100 个字
        "stream": False
    }
    
    start_time = time.time()
    try:
        # 发送请求，设置 120 秒超时
        response = requests.post(URL, json=payload, timeout=120) 
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            duration = end_time - start_time
            # llama.cpp 会在返回的 JSON 里带上生成数量和耗时
            usage = data.get("usage", {})
            gen_tokens = usage.get("completion_tokens", 0)
            speed = gen_tokens / duration if duration > 0 else 0
            
            print(f"[用户 {user_id}] 耗时: {duration:.2f}秒 | 生成字数: {gen_tokens} | 速度: {speed:.2f} 字/秒")
        else:
            print(f"[用户 {user_id}] 请求失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"[用户 {user_id}] 发生报错: {e}")

if __name__ == "__main__":
    print("--- 开始测试：3 个用户同时提问 ---")
    threads = []
    for i in range(3):
        t = threading.Thread(target=test_request, args=(i+1,))
        threads.append(t)
        t.start()
    
    # 等所有用户都回答完毕
    for t in threads:
        t.join()
    
    print("\n--- 压测结束 ---")