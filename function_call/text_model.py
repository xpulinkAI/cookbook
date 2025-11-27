import os
import requests

# 从环境变量读取 API Key
API_KEY = os.getenv("XPULINK_API_KEY")
if not API_KEY:
    raise ValueError("请在环境变量中设置 XPULINK_API_KEY")

# 云端模型接口信息
MODEL_NAME = "qwen3-32b"
BASE_URL = "https://www.xpulink.ai/v1/chat/completions"

# 构造请求头
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 构造请求体
payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "user", "content": "你好，请简单介绍一下你自己。"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
}

# 发送请求并打印结果
try:
    response = requests.post(BASE_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    print("模型返回内容：", result["choices"][0]["message"]["content"])
    print("测试通过！云端模型可正常跑通。")
except Exception as e:
    print("测试失败：", e)
