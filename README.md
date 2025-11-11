# XPULink API Cookbook

这是一个面向 [www.xpulink.ai](https://www.xpulink.ai) 平台模型的 API 使用教程和示例代码集合。通过这些示例，您可以快速上手并集成 XPULink 提供的 AI 模型服务。

## 项目简介

本项目提供了使用 XPULink API 的完整示例，包括：
- 基础文本生成模型调用
- RAG（检索增强生成）应用实现
- 基于 BGE-M3 Embedding 模型的 PDF 文档问答系统
- 自定义 Embedding 模型集成
- Qwen3-32B LoRA 参数高效微调
- 使用 OpenBench 进行模型评估和测试

## 功能特性

- **文本生成**: 演示如何调用云端大语言模型（如 Qwen3-32B）进行对话和文本生成
- **RAG 应用**: 展示如何使用 LlamaIndex 框架构建检索增强生成系统
- **PDF 智能问答**: 使用 BGE-M3 多语言 Embedding 模型构建完整的 PDF 文档问答系统
- **自定义 Embedding**: 提供 OpenAI 兼容的 Embedding 模型实现
- **LoRA 微调**: 使用参数高效的 LoRA 方法对 Qwen3-32B 进行定制化微调
- **设备监控智能Agent**: 基于 Qwen3-32B 的工业设备智能监控与运维Agent
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

### 3. LoRA 微调（推荐）

LoRA 目录包含使用 XPULink API 对 Qwen3-32B 进行参数高效微调的完整示例，让您可以轻松定制专属的 AI 模型。

#### 🎯 什么是 LoRA 微调？

**LoRA (Low-Rank Adaptation)** 是一种参数高效的微调技术：
- ✅ **低成本**: 只训练少量参数，成本远低于全参数微调
- ✅ **高效率**: 训练速度快，通常几分钟到几小时即可完成
- ✅ **效果好**: 在特定任务上接近全参数微调的效果
- ✅ **易部署**: 可以为不同任务训练多个 LoRA 适配器

#### 📦 使用场景

- **企业知识注入**: 将公司产品、流程、规范等知识注入模型
- **领域专家**: 训练医疗、法律、金融等专业领域的对话模型
- **风格定制**: 定制特定语气、格式或风格的文本输出
- **任务优化**: 针对代码生成、文本摘要等特定任务优化

#### 🚀 快速开始

**使用 Jupyter Notebook (推荐):**
```bash
cd LoRA
jupyter notebook lora_finetune_example.ipynb
```

**使用 Python 脚本:**
```bash
cd LoRA

# 1. 准备训练数据
python prepare_training_data.py

# 2. 运行微调（需要先编辑脚本配置）
python lora_finetune.py
```

#### 💡 核心功能

- 📝 **训练数据准备**: 提供工具快速创建符合格式的训练数据
- ☁️ **云端微调**: 所有训练在 XPULink 云端完成，本地无需 GPU
- ⚙️ **超参数配置**: 灵活调整学习率、LoRA 秩等关键参数
- 📊 **进度监控**: 实时查看微调任务状态和进度
- 🧪 **模型测试**: 微调完成后立即测试模型效果

#### 📚 示例代码片段

```python
from lora_finetune import XPULinkLoRAFineTuner

# 初始化微调器
finetuner = XPULinkLoRAFineTuner()

# 准备训练数据
training_data = [
    {
        "messages": [
            {"role": "system", "content": "你是一个专业的Python助手。"},
            {"role": "user", "content": "什么是装饰器?"},
            {"role": "assistant", "content": "装饰器是Python中..."}
        ]
    },
    # 更多训练样本...
]

# 保存并上传数据
finetuner.prepare_training_data(training_data, "data/training.jsonl")
file_id = finetuner.upload_training_file("data/training.jsonl")

# 创建微调任务
job_id = finetuner.create_finetune_job(
    training_file_id=file_id,
    model="qwen3-32b",
    suffix="my-model",
    hyperparameters={
        "n_epochs": 3,
        "learning_rate": 5e-5,
        "lora_r": 8
    }
)

# 等待完成并测试
status = finetuner.wait_for_completion(job_id)
finetuned_model = status['fine_tuned_model']
finetuner.test_finetuned_model(finetuned_model, "测试问题")
```

**详细使用说明和最佳实践请参考** `LoRA/README.md`

### 4. 设备监控智能Agent

Agent 目录包含基于 **Qwen3-32B** 大语言模型的智能设备监控与运维Agent，提供设备状态分析、故障诊断、预防性维护建议和自动化报告生成功能。

#### 🎯 核心功能

- **设备状态实时分析**: 多维度传感器数据分析（温度、压力、振动、能耗等）
- **历史日志智能分析**: 错误和警告模式识别，根本原因推断
- **维修记录关联分析**: 故障频率统计，高频故障部件识别
- **综合诊断报告**: 整合多数据源的全面诊断，结构化的Markdown报告
- **智能行动计划**: 按优先级生成行动项，资源需求评估
- **交互式问答**: 自然语言问答，专业技术建议

#### 📦 应用场景

- **工业制造**: 生产设备监控，生产线故障诊断
- **能源电力**: 发电机组监控，变压器健康管理
- **交通运输**: 车辆车队管理，铁路设备监控
- **建筑设施**: 暖通空调系统，电梯设备管理
- **数据中心**: 服务器监控，冷却系统管理

#### 🚀 快速开始

**使用 Jupyter Notebook (推荐):**
```bash
cd Agent
jupyter notebook device_agent_example.ipynb
```

**使用 Python 脚本:**
```bash
cd Agent

# 运行简单示例
python simple_example.py

# 或使用完整功能
python device_agent.py
```

#### 💡 核心功能示例

```python
from device_agent import DeviceMonitorAgent
import json

# 初始化Agent
agent = DeviceMonitorAgent()

# 准备设备数据
device_data = {
    "device_id": "PUMP-001",
    "device_type": "水泵",
    "temperature": 85.5,
    "pressure": 3.2,
    "vibration": 2.8,
    "power_consumption": 45.2,
    "runtime_hours": 15420,
    "status": "running"
}

# 分析设备状态
result = agent.analyze_device_status(device_data)
print("设备健康评分:", result['health_score'])
print("关键发现:", result['key_findings'])

# 综合诊断
comprehensive_report = agent.comprehensive_diagnosis(
    device_data=device_data
)

# 生成行动计划
action_plan = agent.generate_action_plan(
    diagnosis_report=comprehensive_report['report'],
    priority="immediate"
)

print("行动计划:", action_plan['action_plan'])
```

**详细使用说明和最佳实践请参考** `Agent/README.md`

### 5. 模型评估（OpenBench）

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
├── LoRA/
│   ├── README.md                 # LoRA 微调详细说明
│   ├── lora_finetune.py          # LoRA 微调完整脚本
│   ├── lora_finetune_example.ipynb # LoRA 微调交互式教程 ⭐ 推荐
│   ├── prepare_training_data.py  # 训练数据准备工具
│   └── data/                     # 训练数据目录
├── Agent/
│   ├── README.md                 # 设备监控智能Agent详细说明
│   ├── device_agent.py           # 核心Agent实现
│   ├── device_agent_example.ipynb # Agent交互式教程 ⭐ 推荐
│   ├── simple_example.py         # 简单使用示例
│   └── data/                     # 示例数据目录
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

## 云端推理框架：vLLM

XPULink 平台的所有模型服务都基于 **vLLM (Very Large Language Model)** 推理框架搭建，为用户提供高性能、低成本的 AI 模型服务体验。

### 🚀 vLLM 的核心优势

#### 1. **超高吞吐量**
- 相比传统推理框架（如 HuggingFace Transformers），**吞吐量提升 15-30 倍**
- 通过高效的内存管理和批处理优化，能够同时处理更多并发请求
- 适合大规模生产环境和高并发场景

#### 2. **PagedAttention：革命性的内存管理**
- 借鉴操作系统的虚拟内存分页思想，将 KV Cache 分块存储
- **内存浪费降低 50% 以上**，显著提高 GPU 利用率
- 支持更长的上下文长度和更大的批处理规模
- 动态管理内存，避免传统方式的内存碎片问题

#### 3. **连续批处理 (Continuous Batching)**
- 支持动态调整批次大小，无需等待所有请求完成
- 新请求可以立即加入正在处理的批次
- **大幅降低平均响应延迟**，提升用户体验
- 充分利用 GPU 资源，避免空闲浪费

#### 4. **广泛的模型支持**
- 原生支持主流开源模型：GPT、LLaMA、Qwen、ChatGLM、Baichuan 等
- 兼容 HuggingFace 模型格式，易于部署
- 支持多种量化方案（AWQ、GPTQ、SqueezeLLM）
- 支持 LoRA 适配器动态加载和切换

#### 5. **OpenAI 兼容 API**
- 完全兼容 OpenAI API 规范，无需修改现有代码
- 支持流式输出 (streaming)、Function Calling 等高级特性
- 降低迁移成本，快速切换到自托管或私有云部署

#### 6. **低延迟推理**
- 优化的 CUDA 内核和算子融合技术
- 支持 FP16、BF16、INT8 等多种精度，灵活平衡速度与质量
- 针对 Transformer 架构深度优化，首 token 延迟更低

#### 7. **高可扩展性**
- 支持张量并行和流水线并行
- 轻松扩展到多 GPU、多节点集群
- 适配各类 GPU（MXC500，NVIDIA A100、H100、A10 等）

### 💡 为什么选择 vLLM？

| 对比维度 | vLLM | 传统推理框架 |
|---------|------|-------------|
| **吞吐量** | ⭐⭐⭐⭐⭐ 15-30x | ⭐ 1x |
| **内存效率** | ⭐⭐⭐⭐⭐ 节省 50%+ | ⭐⭐ 常见浪费 |
| **延迟** | ⭐⭐⭐⭐ 动态批处理 | ⭐⭐⭐ 静态批处理 |
| **并发能力** | ⭐⭐⭐⭐⭐ 超高并发 | ⭐⭐ 有限并发 |
| **API 兼容** | ⭐⭐⭐⭐⭐ OpenAI 标准 | ⭐⭐⭐ 需要适配 |
| **模型支持** | ⭐⭐⭐⭐⭐ 广泛支持 | ⭐⭐⭐ 部分支持 |

### 🎯 实际应用场景

通过使用 vLLM，XPULink 能够为您提供：

- **高并发对话服务**：同时支持数千用户在线交互
- **实时 RAG 应用**：快速检索并生成高质量回答
- **批量内容生成**：高效处理大规模文本生成任务
- **成本优化**：相同硬件下服务更多用户，降低单次推理成本
- **稳定可靠**：久经考验的工业级推理引擎，保障服务稳定性

### 📚 了解更多

- vLLM 官方仓库：[https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- vLLM 论文：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

**通过 vLLM 的强大能力，XPULink 为您的 AI 应用提供极致性能和卓越体验！**

## 常见问题

### Q: 如何获取 API Key？
A: 访问 [www.xpulink.ai](https://www.xpulink.ai) 注册账号并在控制台获取您的 API Key。

### Q: 支持哪些模型？
A: 目前示例中使用了：
- 文本生成模型：`qwen3-32b`（支持 LoRA 微调）
- Embedding 模型：`bge-m3`（推荐，特别适合中文）、`text-embedding-ada-002`
更多模型请查看 XPULink 官方文档。

### Q: 什么时候需要使用 LoRA 微调？
A: 以下场景建议使用 LoRA 微调：
- 需要模型了解特定领域知识（如企业内部产品、专业术语等）
- 希望模型按特定风格或格式输出内容
- 提升模型在特定任务上的表现（如代码生成、文本摘要等）
- 需要模型遵守特定的对话规范或准则

LoRA 微调成本低、速度快，通常 50-100 个高质量训练样本即可见效。

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
