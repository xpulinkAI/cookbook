# RAG (检索增强生成) 示例

本目录包含使用 XPULink API 构建 RAG（Retrieval-Augmented Generation）应用的完整示例代码。

## 💻 本地环境依赖

本系统对本地计算机的要求非常低，您只需要安装基本的 Python 环境即可运行所有示例：

### 最低配置要求
- **操作系统**: Windows 10+/macOS 10.15+/Ubuntu 18.04+
- **Python 版本**: 3.8-3.11（推荐 3.9）
- **内存**: 至少 4GB RAM
- **存储空间**: 至少 100MB 可用空间
- **网络连接**: 稳定的互联网连接（用于访问云端 API）

### 本地安装依赖
```bash
# 基础依赖
pip install -r requirements.txt

# 或者手动安装核心组件
pip install llama-index-core llama-index-llms-openai python-dotenv pypdf
```

> ⚠️ **重要**: 所有机器学习模型（包括 BGE-M3 Embedding 模型和 Qwen3-32B 大语言模型）均部署在 XPULink 云端，本地无需安装任何大型模型文件。

## 📁 文件说明

### 1. `process.ipynb`
基础 RAG 示例，展示如何使用 LlamaIndex 框架和 OpenAI 兼容的 Embedding 接口构建文档问答系统。

**主要功能：**
- 使用 `SimpleDirectoryReader` 加载文档
- 自定义 `OpenAICompatibleEmbedding` 类
- 基于 OpenAI 风格的 API 进行文档向量化
- 支持批处理提高效率

### 2. `pdf_rag_with_bge_m3.ipynb` ⭐ 推荐
**完整的 PDF 文档 RAG 系统**，使用 XPULink 平台托管的 BGE-M3 Embedding 模型。

**主要特性：**
- ✅ 专门处理 PDF 文档
- ✅ 使用 BGE-M3 多语言 Embedding 模型（对中文支持极佳）
- ✅ 完整的向量索引构建流程
- ✅ 智能文档检索和问答
- ✅ 交互式查询界面
- ✅ 详细的中文注释和使用说明

## 🚀 快速开始

### 环境准备

1. **安装依赖包**
```bash
pip install llama-index-core llama-index-llms-openai python-dotenv pypdf
```

2. **配置 API Key**

创建 `.env` 文件（在项目根目录）：
```bash
# XPULink API Key
XPU_API_KEY=your_api_key_here

# 可选：用于不同的 API 端点
CLOUD_API_KEY=your_api_key_here
XPULINK_API_KEY=your_api_key_here
```

### 使用 BGE-M3 构建 PDF RAG 系统

这是最推荐的方式，特别适合处理中文 PDF 文档。

1. **准备 PDF 文档**

在 RAG 目录下创建 `data` 文件夹，并放入您的 PDF 文件：
```bash
mkdir -p data
cp your_document.pdf data/
```

2. **运行 Jupyter Notebook**

```bash
cd RAG
jupyter notebook pdf_rag_with_bge_m3.ipynb
```

3. **按照 Notebook 中的步骤执行**
   - 单元格 1-3: 安装依赖和配置环境
   - 单元格 4: 加载 PDF 文档
   - 单元格 5: 构建向量索引（这一步会调用 BGE-M3 模型）
   - 单元格 6-7: 创建查询引擎并测试查询
   - 单元格 8: 启动交互式查询（可选）

### 使用基础 RAG 示例

```bash
cd RAG
jupyter notebook process.ipynb
```

## 💡 核心概念

### 什么是 RAG？

RAG（Retrieval-Augmented Generation）是一种结合信息检索和文本生成的技术：

1. **检索（Retrieval）**: 从文档库中找到与问题最相关的内容
2. **增强（Augmented）**: 将检索到的内容作为上下文
3. **生成（Generation）**: LLM 基于检索内容生成准确答案

### BGE-M3 Embedding 模型

BGE-M3 是由智源研究院开发的强大多语言 Embedding 模型：

- 🌍 **多语言支持**: 支持 100+ 种语言，中英文效果特别好
- 📊 **高性能**: 在多个基准测试中表现优异
- 🎯 **长文本**: 支持最长 8192 token 的输入
- 🔄 **多功能**: 支持密集检索、稀疏检索和多向量检索

## 📊 工作流程

```
PDF 文档
   ↓
文档加载 (SimpleDirectoryReader)
   ↓
文本分块
   ↓
向量化 (BGE-M3 Embedding)
   ↓
构建索引 (VectorStoreIndex)
   ↓
用户查询 → 相似度检索 → 提取相关片段
   ↓
LLM 生成回答 (Qwen3-32B)
   ↓
返回结果
```

## 🔧 高级配置

### 调整检索参数

```python
query_engine = index.as_query_engine(
    similarity_top_k=3,      # 返回最相似的 K 个片段（默认 3）
    response_mode="compact"   # 响应模式：compact/tree_summarize/refine
)
```

### 自定义 Embedding 批处理大小

```python
Settings.embed_model = BGEM3Embedding(
    api_base="https://xpulink.ai/v1",
    model="bge-m3",
    embed_batch_size=10  # 根据 API 限制和网络情况调整
)
```

### 调整 LLM 参数

```python
Settings.llm = OpenAI(
    api_key=os.getenv("XPU_API_KEY"),
    api_base="https://www.xpulink.ai/v1",
    model="qwen3-32b",
    temperature=0.7,      # 0-2，越高越有创造性
    max_tokens=2000      # 最大生成 token 数
)
```

## 🎓 使用场景

- **智能文档问答**: 快速从大量 PDF 文档中找到答案
- **知识库构建**: 将企业文档转化为可查询的知识库
- **研究辅助**: 帮助研究人员快速检索学术论文
- **客服系统**: 基于产品文档自动回答用户问题
- **法律/医疗文档分析**: 快速定位专业文档中的关键信息

## 🔍 常见问题

### Q: 为什么选择 BGE-M3 而不是其他 Embedding 模型？

A: BGE-M3 的优势：
- 对中文支持更好
- 支持更长的文本（8192 tokens）
- 在多个基准测试中性能优异
- XPULink 平台原生支持，调用方便

### Q: 处理大型 PDF 需要多长时间？

A: 时间取决于：
- PDF 页数和内容量
- 网络速度
- API 响应速度

一般情况下，100 页的 PDF 文档处理时间在 1-3 分钟左右。

### Q: 如何提高查询准确度？

A: 可以尝试：
1. 增加 `similarity_top_k` 的值以检索更多相关片段
2. 使用更具体、更详细的查询问题
3. 调整文档分块大小（修改 LlamaIndex 的 chunk_size）
4. 使用更强大的 LLM 模型

### Q: 支持哪些文档格式？

A:
- `pdf_rag_with_bge_m3.ipynb`: 专门处理 PDF
- `process.ipynb`: 支持 LlamaIndex 支持的所有格式（txt, pdf, docx, etc.）

### Q: 能否离线使用？

A: 不能。本示例依赖 XPULink 云端 API，需要网络连接。

## 📚 相关资源

- [XPULink 官网](https://www.xpulink.ai)
- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [BGE-M3 GitHub](https://github.com/FlagOpen/FlagEmbedding)
- [RAG 技术详解](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这些示例！

## 📝 许可证

MIT License

---

**提示**: 请确保妥善保管您的 API Key，不要将 `.env` 文件提交到公开仓库。
