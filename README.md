# XPULink API Cookbook

**Build powerful AI applications with zero infrastructure hassle** - A comprehensive collection of examples for [www.xpulink.net](https://www.xpulink.net) 🚀

## Why XPULink?

### 🎯 **No GPU? No Problem!**
- **100% Cloud-Hosted**: All models run on XPULink's infrastructure
- **Zero Setup**: No CUDA, no drivers, no expensive hardware needed
- **Instant Access**: Get started in minutes with just an API key

### ⚡ **Powered by vLLM - Enterprise-Grade Performance**
- **15-30x Faster** than traditional inference frameworks
- **50% Better Memory Efficiency** with PagedAttention technology
- **High Concurrency**: Handle thousands of requests simultaneously
- **Low Latency**: Optimized CUDA kernels for blazing-fast responses

### 🔌 **OpenAI-Compatible API**
- Drop-in replacement for OpenAI API
- Use with LangChain, LlamaIndex, and other popular frameworks
- Minimal code changes to switch from OpenAI

### 💰 **Cost-Effective**
- Pay only for what you use
- No idle infrastructure costs
- Transparent pricing

---

## 📚 What's Inside

This cookbook provides production-ready examples for:

| Feature | Description | Best For |
|---------|-------------|----------|
| 🤖 **Text Generation** | Basic LLM inference with Qwen3-32B | Chat, content generation |
| 📄 **RAG System** | PDF Q&A with BGE-M3 embeddings | Document analysis, knowledge bases |
| 🎯 **LoRA Fine-tuning** | Custom model training | Domain adaptation, style transfer |
| 🏭 **Device Monitoring Agent** | Industrial IoT diagnostics | Predictive maintenance, anomaly detection |
| 📊 **Model Evaluation** | Benchmark testing with OpenBench | Model comparison, performance analysis |

**All examples now use LiteLLM** for elegant, production-ready integration with custom APIs!

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- XPULink API Key from [www.xpulink.net](https://www.xpulink.net)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd cookbook

# Install dependencies
pip install -r requirements.txt

# Set up your API key
echo "XPULINK_API_KEY=your_api_key_here" > .env
```

### Your First API Call (30 seconds!)

```python
from litellm import completion

response = completion(
    model="openai/qwen3-32b",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="your_api_key",
    api_base="https://www.xpulink.net/v1",
    custom_llm_provider="openai"
)

print(response.choices[0].message.content)
```

**That's it! No GPU setup, no model downloads, just pure API magic.** ✨

---

## 📖 Examples

### 1. 💬 Text Generation

**The simplest way to use LLMs**

```bash
cd function_call
python text_model.py
```

**What you get:**
- OpenAI-compatible chat completions
- Streaming support
- Function calling (when available)
- Full control over temperature, tokens, etc.

**Why it's easy with XPULink:**
- ✅ No model downloads (GBs of data)
- ✅ No GPU required
- ✅ Instant API access
- ✅ Auto-scaling infrastructure

---

### 2. 📄 RAG System (Retrieval-Augmented Generation)

**Build ChatGPT for your documents**

```bash
cd RAG

# Put your PDFs in data/
mkdir -p data
cp your_document.pdf data/

# Run the system
python pdf_rag_bge_m3.py
```

**Features:**
- 🌍 **BGE-M3 Embeddings**: Best-in-class multilingual model
- 📚 **PDF Processing**: Automatic text extraction and chunking
- 🔍 **Semantic Search**: Find relevant context for any question
- 🤖 **LLM Integration**: Generate answers based on your documents
- 💾 **Vector Storage**: Efficient retrieval with LlamaIndex

**Why RAG on XPULink:**
- ✅ **No Embedding Server**: BGE-M3 hosted for you
- ✅ **No LLM Hosting**: Qwen3-32B ready to use
- ✅ **Automatic Retries**: Built-in error handling
- ✅ **LiteLLM Integration**: Clean, maintainable code

**Use Cases:**
- Corporate knowledge bases
- Customer support bots
- Research paper analysis
- Legal document search

See `RAG/README.md` for detailed documentation.

---

### 3. 🎯 LoRA Fine-tuning

**Customize models for your specific needs - on the cloud!**

```bash
cd LoRA

# Interactive notebook (recommended)
jupyter notebook lora_finetune_example.ipynb

# Or use Python script
python lora_finetune.py
```

**What is LoRA?**
- **Parameter-Efficient**: Train only 0.1% of model parameters
- **Fast**: Minutes to hours (vs. days for full fine-tuning)
- **Cheap**: Much lower compute costs
- **Effective**: Near full fine-tuning quality

**Why Fine-tune on XPULink:**
- ✅ **Cloud Training**: Zero local GPU needed
- ✅ **Managed Infrastructure**: We handle everything
- ✅ **Easy API**: Upload, configure, train, deploy
- ✅ **Quick Turnaround**: Get results fast

**Perfect For:**
- 🏢 **Enterprise**: Inject company knowledge
- 🏥 **Domain Experts**: Medical, legal, finance terminology
- ✍️ **Style**: Custom tone, format, personality
- 🎯 **Task Optimization**: Code generation, summarization, etc.

**Example:**
```python
from lora_finetune import XPULinkLoRAFineTuner

finetuner = XPULinkLoRAFineTuner()

# Prepare data
training_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a Python expert."},
            {"role": "user", "content": "Explain decorators"},
            {"role": "assistant", "content": "Decorators in Python..."}
        ]
    },
    # ... more examples
]

# Train in the cloud
file_id = finetuner.upload_training_file("training.jsonl")
job_id = finetuner.create_finetune_job(file_id, model="qwen3-32b")
status = finetuner.wait_for_completion(job_id)

# Use your custom model
finetuned_model = status['fine_tuned_model']
```

See `LoRA/README.md` for best practices and advanced configuration.

---

### 4. 🏭 Device Monitoring Agent

**AI-powered predictive maintenance**

```bash
cd Agent

# Interactive demo
jupyter notebook device_agent_example.ipynb

# Or quick test
python simple_example.py
```

**Capabilities:**
- 📊 **Real-time Analysis**: Multi-sensor data interpretation
- 📝 **Log Intelligence**: Pattern recognition in error logs
- 🔧 **Maintenance Planning**: Predictive scheduling
- 📈 **Trend Analysis**: Identify degradation patterns
- 📋 **Automated Reports**: Structured diagnostic output

**Industry Applications:**
- Manufacturing: Production line monitoring
- Energy: Power generation equipment
- Transportation: Fleet management
- Data Centers: Server health monitoring

**Why on XPULink:**
- ✅ **Always Available**: 24/7 cloud inference
- ✅ **No Latency Issues**: Fast response times
- ✅ **Scalable**: Monitor thousands of devices
- ✅ **Cost-Effective**: No dedicated servers needed

See `Agent/README.md` for implementation details.

---

### 5. 📊 Model Evaluation

**Benchmark your models with OpenBench**

```bash
cd Evaluation

# Install OpenBench
pip install openbench

# Run evaluation
openbench evaluate \
  --model-type openai \
  --model-name qwen3-32b \
  --api-key $XPULINK_API_KEY \
  --base-url https://www.xpulink.net/v1 \
  --benchmark mmlu
```

**Supported Benchmarks:**
- MMLU (Massive Multitask Language Understanding)
- GSM8K (Math reasoning)
- HellaSwag (Common sense reasoning)
- Custom benchmarks

See `Evaluation/README.md` for comprehensive guide.

---

## 🏗️ Architecture

### Built on vLLM - The Fastest Inference Engine

XPULink uses **vLLM** (Very Large Language Model) for all model serving:

| Feature | vLLM (XPULink) | Traditional Frameworks |
|---------|---------------|----------------------|
| **Throughput** | ⚡ **15-30x faster** | 1x baseline |
| **Memory** | 💾 **50% more efficient** | Standard |
| **Latency** | 🚀 **Dynamic batching** | Static batching |
| **Concurrency** | 🌐 **Thousands of users** | Limited |
| **API** | ✅ **OpenAI compatible** | Custom |

**Key Technologies:**
- **PagedAttention**: Revolutionary memory management
- **Continuous Batching**: No waiting for batch completion
- **Tensor Parallelism**: Multi-GPU scaling
- **Quantization**: FP16, INT8 support

**Learn more:** [vLLM GitHub](https://github.com/vllm-project/vllm)

---

## 🛠️ Technical Stack

### LiteLLM Integration

All examples use **LiteLLM** for elegant API integration:

```python
from litellm import completion

# Clean, consistent API across all providers
response = completion(
    model="openai/qwen3-32b",
    messages=[...],
    api_key=api_key,
    api_base="https://www.xpulink.net/v1",
    custom_llm_provider="openai"
)
```

**Why LiteLLM:**
- ✅ **No Hacks**: No workarounds or monkey-patching
- ✅ **Production-Ready**: Used by thousands of developers
- ✅ **Unified Interface**: Works with 100+ LLM providers
- ✅ **Built-in Retries**: Automatic error handling
- ✅ **Easy Migration**: Switch providers with one line

---

## 📁 Project Structure

```
cookbook/
├── README.md                          # This file
├── requirements.txt                   # Shared dependencies
│
├── function_call/                     # Basic text generation
│   ├── text_model.py
│   └── requirements.txt
│
├── RAG/                              # Document Q&A system
│   ├── README.md
│   ├── pdf_rag_bge_m3.py            # Main RAG system
│   ├── requirements.txt
│   └── data/                         # Your PDFs go here
│
├── LoRA/                             # Model fine-tuning
│   ├── README.md
│   ├── lora_finetune.py             # Fine-tuning manager
│   ├── lora_finetune_example.ipynb  # Interactive tutorial
│   ├── requirements.txt
│   └── data/                         # Training data
│
├── Agent/                            # Device monitoring
│   ├── README.md
│   ├── device_agent.py              # Agent implementation
│   ├── simple_example.py
│   ├── requirements.txt
│   └── data/                         # Sample device data
│
└── Evaluation/                       # Model benchmarking
    └── README.md
```

---

## 💡 Best Practices

### API Key Security
```bash
# ✅ DO: Use environment variables
XPULINK_API_KEY=your_key python script.py

# ❌ DON'T: Hardcode keys
api_key = "sk-..."  # Never do this!
```

### Error Handling
```python
# LiteLLM provides automatic retries
response = completion(
    model="openai/qwen3-32b",
    messages=[...],
    api_key=api_key,
    api_base="https://www.xpulink.net/v1",
    custom_llm_provider="openai",
    num_retries=3  # Automatic retry on failure
)
```

### Performance Optimization
- Use appropriate `temperature` for your use case
- Set reasonable `max_tokens` limits
- Batch requests when possible
- Use streaming for real-time applications

---

## 🤝 Support & Community

### Getting Help
- 📚 **Documentation**: [www.xpulink.net/docs](https://www.xpulink.net/docs)
- 💬 **Issues**: Open an issue on GitHub
- 📧 **Email**: support@xpulink.net
- 🌐 **Website**: [www.xpulink.net](https://www.xpulink.net)

### Contributing
We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

MIT License - see LICENSE file for details

---

## 🌟 Why Developers Love XPULink

> "No GPU setup, no model downloads - I had a RAG system running in 10 minutes!"
> — *Sarah, ML Engineer*

> "The fine-tuning API saved us weeks of infrastructure work. Just upload and train."
> — *Mike, Startup Founder*

> "vLLM performance + OpenAI compatibility = perfect combo"
> — *Alex, DevOps Lead*

---

## 🚀 Ready to Build?

1. **Get your API key**: [www.xpulink.net](https://www.xpulink.net)
2. **Pick an example**: Start with RAG or text generation
3. **Run the code**: Copy, paste, customize
4. **Ship to production**: Scale with confidence

**No credit card needed to start experimenting!** 🎉

---

**Built with ❤️ by the XPULink team**

*Powered by vLLM | OpenAI-Compatible | Production-Ready*
