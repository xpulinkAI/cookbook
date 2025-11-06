# XPULink API Cookbook

这是一个面向 [www.xpulink.ai](https://www.xpulink.ai) 平台模型的 API 使用教程和示例代码集合。通过这些示例，您可以快速上手并集成 XPULink 提供的 AI 模型服务。

## 项目简介

本项目提供了使用 XPULink API 的完整示例，包括：
- 基础文本生成模型调用
- RAG（检索增强生成）应用实现
- 基于 BGE-M3 Embedding 模型的 PDF 文档问答系统
- 自定义 Embedding 模型集成
- 使用 OpenBench 进行模型评估和测试

## 功能特性

- **文本生成**: 演示如何调用云端大语言模型（如 Qwen3-32B）进行对话和文本生成
- **RAG 应用**: 展示如何使用 LlamaIndex 框架构建检索增强生成系统
- **PDF 智能问答**: 使用 BGE-M3 多语言 Embedding 模型构建完整的 PDF 文档问答系统
- **自定义 Embedding**: 提供 OpenAI 兼容的 Embedding 模型实现
- **模型评估**: 使用 OpenBench 框架对 XPULink 模型进行标准化评估和测试
- **生产就绪**: 包含错误处理、环境变量配置等最佳实践

## 环境要求

- Python 3.8+
- XPULink API Key（从 [www.xpulink.ai](https://www.xpulink.ai) 获取）

## 安装步骤

1. 克隆本仓库：
```bash
git clone <repository-url>
cd function_call
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：

创建 `.env` 文件并添加您的 API Key：
```bash
# 用于基础文本模型
XPULINK_API_KEY=your_api_key_here

# 用于 RAG 示例（Embedding 模型）
XPU_API_KEY=your_api_key_here
CLOUD_API_KEY=your_api_key_here
```

## 使用示例

### 1. 基础文本生成

运行 `text_model.py` 来测试基础的文本生成功能：

```bash
python text_model.py
```

这个示例展示了如何：
- 配置 API 认证
- 构造请求体
- 发送 POST 请求到 XPULink API
- 处理返回结果

**示例代码片段**：
```python
import os
import requests

API_KEY = os.getenv("XPULINK_API_KEY")
MODEL_NAME = "qwen3-32b"
BASE_URL = "https://www.xpulink.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "user", "content": "你好，请简单介绍一下你自己。"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
}

response = requests.post(BASE_URL, headers=headers, json=payload, timeout=30)
result = response.json()
print("模型返回内容：", result["choices"][0]["message"]["content"])
```

### 2. RAG（检索增强生成）

RAG 目录包含两个完整的文档问答系统示例，展示如何使用 LlamaIndex 框架构建智能检索增强生成应用。

#### 📄 PDF 智能问答系统（推荐）

**使用 BGE-M3 Embedding 模型的 PDF RAG 系统**：
```bash
cd RAG
# 准备 PDF 文档
mkdir -p data
cp your_document.pdf data/

# 运行 Notebook
jupyter notebook pdf_rag_with_bge_m3.ipynb
```

**核心功能**：
- ✅ 专门针对 PDF 文档优化
- ✅ 使用 BGE-M3 多语言 Embedding 模型（对中文支持极佳）
- ✅ 完整的文档加载、向量化、检索流程
- ✅ 基于检索内容的智能问答
- ✅ 交互式查询界面
- ✅ 详细的中文注释和使用说明

**BGE-M3 模型优势**：
- 🌍 支持 100+ 种语言，中英文效果特别好
- 📊 在多个基准测试中表现优异
- 🎯 支持最长 8192 token 的输入
- 🔄 支持密集检索、稀疏检索和多向量检索

#### 🔧 基础 RAG 示例

**使用 OpenAI 兼容 API 的通用 RAG 系统**：
```bash
cd RAG
jupyter notebook process.ipynb
```

**主要特性**：
- 使用 `SimpleDirectoryReader` 加载文档
- 实现了 `OpenAICompatibleEmbedding` 类，支持 OpenAI 风格的 API
- 批处理支持，提高效率
- 完整的错误处理

**详细使用说明请参考** `RAG/README.md`

### 3. 模型评估（OpenBench）

使用 OpenBench 框架对 XPULink 托管的模型进行标准化评估和测试。

**查看详细指南**：
```bash
cd Evaluation
cat README.md
```

**快速开始**：

1. 安装 OpenBench：
```bash
pip install openbench
```

2. 配置环境变量：
```bash
export XPU_API_KEY=your_api_key_here
export OPENAI_API_BASE=https://www.xpulink.ai/v1
```

3. 运行评估：
```bash
openbench evaluate \
  --model-type openai \
  --model-name qwen3-32b \
  --api-key $XPU_API_KEY \
  --base-url https://www.xpulink.ai/v1 \
  --benchmark mmlu
```

**评估功能**：
- 支持多种标准基准测试（MMLU、GSM8K、HellaSwag 等）
- 自定义评估任务
- 详细的性能报告和分析
- 批量对比多个模型

完整的使用说明和代码示例请参考 `Evaluation/README.md`。

## API 配置说明

### 文本生成 API

**端点**: `https://www.xpulink.ai/v1/chat/completions`

**请求参数**:
- `model`: 模型名称（如 "qwen3-32b"）
- `messages`: 对话历史数组
- `max_tokens`: 最大生成 token 数
- `temperature`: 温度参数（0-2），控制随机性

### Embedding API

**端点**: `https://xpulink.ai/v1/embeddings`

**请求参数**:
- `model`: Embedding 模型名称（如 "text-embedding-ada-002"）
- `input`: 单个字符串或字符串数组

## 项目结构

```
function_call/
├── README.md                      # 项目说明文档
├── requirements.txt               # Python 依赖列表
├── text_model.py                 # 基础文本生成示例
├── RAG/
│   ├── README.md                 # RAG 示例详细说明
│   ├── process.ipynb             # 基础 RAG 应用示例
│   └── pdf_rag_with_bge_m3.ipynb # PDF 智能问答系统（使用 BGE-M3）⭐ 推荐
└── Evaluation/
    └── README.md                 # OpenBench 模型评估指南
```

## 依赖说明

主要依赖包：
- `llama-index-core`: LlamaIndex 核心框架
- `llama-index-embeddings-openai`: OpenAI Embedding 支持
- `requests`: HTTP 请求库
- `python-dotenv`: 环境变量管理
- `jupyter`: Jupyter Notebook 支持

完整依赖列表请查看 `requirements.txt`。

## 常见问题

### Q: 如何获取 API Key？
A: 访问 [www.xpulink.ai](https://www.xpulink.ai) 注册账号并在控制台获取您的 API Key。

### Q: 支持哪些模型？
A: 目前示例中使用了：
- 文本生成模型：`qwen3-32b`
- Embedding 模型：`bge-m3`（推荐，特别适合中文）、`text-embedding-ada-002`
更多模型请查看 XPULink 官方文档。

### Q: API 请求失败怎么办？
A: 请检查：
1. API Key 是否正确配置
2. 网络连接是否正常
3. API 配额是否充足
4. 请求参数是否符合规范

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请访问 [www.xpulink.ai](https://www.xpulink.ai) 或提交 Issue。

---

**注意**: 请妥善保管您的 API Key，不要将其提交到公开仓库中。建议使用 `.env` 文件并将其添加到 `.gitignore`。
