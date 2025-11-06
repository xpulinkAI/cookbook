# 使用 OpenBench 评估 XPULink 模型

本指南将帮助您使用 OpenBench 框架来评估和测试托管在 [XPULink](https://www.xpulink.ai) 平台上的 AI 模型。OpenBench 是一个强大的模型评估工具，支持多种基准测试和评估指标。

## 什么是 OpenBench？

OpenBench 是一个开源的 AI 模型评估框架，支持：
- 多种标准评估基准（MMLU、GSM8K、HellaSwag 等）
- 自定义评估任务
- 与 OpenAI 兼容的 API 接口
- 详细的性能报告和分析

## 环境要求

- Python 3.8+
- XPULink API Key（从 [www.xpulink.ai](https://www.xpulink.ai) 获取）
- OpenBench 框架

## 安装步骤

### 1. 安装 OpenBench

```bash
# 使用 pip 安装 OpenBench
pip install openbench

# 或从源码安装
git clone https://github.com/OpenBMB/OpenBench.git
cd OpenBench
pip install -e .
```

### 2. 配置环境变量

创建 `.env` 文件或在系统中设置以下环境变量：

```bash
# XPULink API Key
export XPU_API_KEY=your_api_key_here

# XPULink API Base URL
export OPENAI_API_BASE=https://www.xpulink.ai/v1
```

或者在项目目录下创建 `.env` 文件：

```
XPU_API_KEY=your_api_key_here
OPENAI_API_BASE=https://www.xpulink.ai/v1
```

## 使用 OpenBench 测试 XPULink 模型

### 基础配置示例

创建一个配置文件 `xpulink_config.yaml`：

```yaml
# XPULink 模型配置
model:
  type: openai  # 使用 OpenAI 兼容接口
  name: qwen3-32b  # XPULink 上的模型名称
  api_key: ${XPU_API_KEY}  # 从环境变量读取
  base_url: https://www.xpulink.ai/v1  # XPULink API 基础 URL

# 评估配置
evaluation:
  benchmarks:
    - mmlu  # 多任务语言理解
    - gsm8k  # 数学推理
    - hellaswag  # 常识推理

  # 生成参数
  generation:
    temperature: 0.0  # 确定性输出
    max_tokens: 2048
    top_p: 1.0
```

### Python 代码示例

```python
import os
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

# 配置 OpenAI 客户端连接到 XPULink
openai.api_key = os.getenv("XPU_API_KEY")
openai.api_base = "https://www.xpulink.ai/v1"

# 测试连接
def test_xpulink_model():
    """测试 XPULink 模型是否可访问"""
    try:
        response = openai.ChatCompletion.create(
            model="qwen3-32b",
            messages=[
                {"role": "user", "content": "请用一句话介绍人工智能。"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        print("模型响应成功！")
        print("返回内容：", response.choices[0].message.content)
        return True
    except Exception as e:
        print(f"连接失败: {e}")
        return False

if __name__ == "__main__":
    test_xpulink_model()
```

### 运行评估

#### 方法 1: 使用命令行

```bash
# 运行单个基准测试
openbench evaluate \
  --model-type openai \
  --model-name qwen3-32b \
  --api-key $XPU_API_KEY \
  --base-url https://www.xpulink.ai/v1 \
  --benchmark mmlu

# 运行多个基准测试
openbench evaluate \
  --config xpulink_config.yaml \
  --output results/xpulink_evaluation.json
```

#### 方法 2: 使用 Python 脚本

创建 `run_evaluation.py`：

```python
import os
from dotenv import load_dotenv
from openbench import Evaluator

# 加载环境变量
load_dotenv()

# 配置评估器
evaluator = Evaluator(
    model_type="openai",
    model_name="qwen3-32b",
    api_key=os.getenv("XPU_API_KEY"),
    base_url="https://www.xpulink.ai/v1"
)

# 运行评估
results = evaluator.run_benchmarks([
    "mmlu",      # 多任务语言理解
    "gsm8k",     # 数学推理
    "hellaswag"  # 常识推理
])

# 保存结果
evaluator.save_results(results, "results/xpulink_evaluation.json")

# 打印摘要
print("\n评估结果摘要：")
for benchmark, scores in results.items():
    print(f"{benchmark}: {scores['accuracy']:.2%}")
```

运行脚本：

```bash
python run_evaluation.py
```

## 高级配置

### 自定义评估任务

创建 `custom_evaluation.py`：

```python
import os
from dotenv import load_dotenv
import openai

load_dotenv()

# 配置 XPULink API
openai.api_key = os.getenv("XPU_API_KEY")
openai.api_base = "https://www.xpulink.ai/v1"

def evaluate_custom_task(questions, model="qwen3-32b"):
    """
    自定义评估任务

    Args:
        questions: 问题列表
        model: 模型名称

    Returns:
        评估结果
    """
    results = []

    for q in questions:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的 AI 助手。"},
                {"role": "user", "content": q["question"]}
            ],
            max_tokens=512,
            temperature=0.0
        )

        answer = response.choices[0].message.content
        results.append({
            "question": q["question"],
            "model_answer": answer,
            "expected_answer": q.get("expected_answer", None),
            "correct": answer.strip() == q.get("expected_answer", "").strip()
        })

    return results

# 示例问题集
questions = [
    {
        "question": "什么是机器学习？",
        "expected_answer": "机器学习是人工智能的一个分支，通过数据和经验让计算机自动改进性能。"
    },
    {
        "question": "Python 是什么类型的编程语言？",
        "expected_answer": "Python 是一种高级、解释型、面向对象的编程语言。"
    }
]

# 运行评估
results = evaluate_custom_task(questions)

# 计算准确率
accuracy = sum(1 for r in results if r["correct"]) / len(results)
print(f"准确率: {accuracy:.2%}")
```

### 批量测试多个模型

```python
import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("XPU_API_KEY")
openai.api_base = "https://www.xpulink.ai/v1"

# 定义要测试的模型列表
models_to_test = [
    "qwen3-32b",
    "qwen3-14b",
    "llama3-70b"
]

def benchmark_models(models, test_prompt):
    """对比多个模型的表现"""
    results = {}

    for model in models:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=100,
                temperature=0.0
            )
            results[model] = {
                "success": True,
                "response": response.choices[0].message.content,
                "tokens": response.usage.total_tokens
            }
        except Exception as e:
            results[model] = {
                "success": False,
                "error": str(e)
            }

    return results

# 运行对比测试
test_prompt = "请解释什么是深度学习，用不超过50个字。"
comparison = benchmark_models(models_to_test, test_prompt)

# 输出结果
for model, result in comparison.items():
    print(f"\n模型: {model}")
    if result["success"]:
        print(f"响应: {result['response']}")
        print(f"Token 使用: {result['tokens']}")
    else:
        print(f"错误: {result['error']}")
```

## 支持的评估基准

OpenBench 支持以下标准评估基准来测试 XPULink 模型：

| 基准测试 | 描述 | 评估能力 |
|---------|------|----------|
| MMLU | 大规模多任务语言理解 | 知识广度、专业领域理解 |
| GSM8K | 小学数学问题 | 数学推理、问题解决 |
| HellaSwag | 常识推理 | 常识理解、情境补全 |
| TruthfulQA | 真实性问答 | 事实准确性、诚实性 |
| HumanEval | 代码生成 | 编程能力、代码理解 |
| MBPP | Python 编程基准 | 基础编程能力 |

## 结果分析

评估完成后，OpenBench 会生成详细的报告，包括：

- **准确率 (Accuracy)**: 正确回答的问题比例
- **F1 分数**: 精确率和召回率的调和平均
- **推理时间**: 每个任务的平均响应时间
- **Token 使用量**: API 调用的 token 消耗
- **成本估算**: 基于 token 使用的成本预估

查看结果示例：

```python
import json

# 加载评估结果
with open("results/xpulink_evaluation.json", "r") as f:
    results = json.load(f)

# 打印详细结果
print(json.dumps(results, indent=2, ensure_ascii=False))
```

## 常见问题

### Q: 如何获取 XPU_API_KEY？
A: 访问 [www.xpulink.ai](https://www.xpulink.ai) 注册账号，在控制台的 API 密钥管理页面创建并获取您的 API Key。

### Q: 评估过程中 API 超时怎么办？
A: 您可以在配置中增加超时时间：
```python
openai.request_timeout = 60  # 设置为 60 秒
```

### Q: 如何查看详细的评估日志？
A: 启用详细日志模式：
```bash
openbench evaluate --config config.yaml --verbose --log-file evaluation.log
```

### Q: 支持哪些 XPULink 模型？
A: OpenBench 支持 XPULink 平台上所有兼容 OpenAI API 格式的模型。常见的包括：
- qwen3-32b
- qwen3-14b
- llama3-70b
- deepseek-chat

请访问 XPULink 官方文档查看完整的模型列表。

### Q: 如何控制评估成本？
A: 建议采取以下措施：
1. 先在小数据集上测试
2. 使用 `max_tokens` 限制生成长度
3. 设置 `temperature=0.0` 确保确定性输出，避免重复测试
4. 使用缓存机制避免重复调用

## 最佳实践

1. **API Key 安全**:
   - 始终使用环境变量存储 API Key
   - 将 `.env` 文件添加到 `.gitignore`
   - 不要在代码中硬编码密钥

2. **评估策略**:
   - 从小规模数据集开始测试
   - 逐步增加评估任务的复杂度
   - 定期保存中间结果

3. **错误处理**:
   - 实现重试机制处理网络波动
   - 记录失败的测试用例
   - 监控 API 配额使用情况

4. **结果对比**:
   - 保存历史评估结果用于对比
   - 使用相同的随机种子确保可重复性
   - 记录评估时的模型版本和配置

## 示例项目结构

```
Evaluation/
├── README.md                    # 本文档
├── config/
│   ├── xpulink_config.yaml     # XPULink 配置
│   └── benchmarks.yaml          # 基准测试配置
├── scripts/
│   ├── test_connection.py       # 测试连接脚本
│   ├── run_evaluation.py        # 运行评估脚本
│   └── custom_evaluation.py     # 自定义评估
└── results/
    └── xpulink_evaluation.json  # 评估结果
```

## 相关资源

- [OpenBench 官方文档](https://github.com/OpenBMB/OpenBench)
- [XPULink API 文档](https://www.xpulink.ai/docs)
- [OpenAI API 兼容规范](https://platform.openai.com/docs/api-reference)

## 技术支持

如有问题或建议，请：
1. 访问 [XPULink 官网](https://www.xpulink.ai)
2. 查看 OpenBench 项目的 Issue 页面
3. 在本项目提交 Issue

---

**注意**: 请合理使用 API 配额，避免产生不必要的费用。建议在进行大规模评估前先估算成本。
