# LoRA 微调示例

本目录包含使用 XPULink API 对 Qwen3-32B 模型进行 LoRA (Low-Rank Adaptation) 微调的完整示例代码。

## 💻 本地环境依赖

本系统对本地计算机的要求非常低,您只需要安装基本的 Python 环境即可运行所有示例:

### 最低配置要求
- **操作系统**: Windows 10+/macOS 10.15+/Ubuntu 18.04+
- **Python 版本**: 3.8-3.11 (推荐 3.9)
- **内存**: 至少 4GB RAM
- **存储空间**: 至少 100MB 可用空间
- **网络连接**: 稳定的互联网连接 (用于访问云端 API)

### 本地安装依赖
```bash
# 基础依赖
pip install requests python-dotenv

# 或使用项目根目录的 requirements.txt
pip install -r ../requirements.txt
```

> ⚠️ **重要**: Qwen3-32B 模型及所有训练过程均在 XPULink 云端完成,本地无需 GPU 或大量计算资源。

## 📁 文件说明

### 1. `lora_finetune.py`
完整的 LoRA 微调管理脚本,提供了一个易用的 Python 类来管理整个微调流程。

**主要功能:**
- ✅ 训练数据准备和格式化
- ✅ 文件上传到 XPULink 平台
- ✅ 创建和管理微调任务
- ✅ 监控微调进度
- ✅ 测试微调后的模型

### 2. `lora_finetune_example.ipynb` ⭐ 推荐
**交互式 Jupyter Notebook 教程**,逐步指导您完成 LoRA 微调全流程。

**主要特性:**
- ✅ 详细的中文注释和说明
- ✅ 分步骤的交互式执行
- ✅ 实时查看微调进度
- ✅ 即时测试微调效果
- ✅ 包含完整的数据准备示例

### 3. `prepare_training_data.py`
训练数据准备工具,帮助您快速创建符合格式要求的训练数据。

**主要功能:**
- 单轮对话数据生成
- 多轮对话数据生成
- 数据格式验证
- 包含多个领域的示例数据

## 🚀 快速开始

### 环境准备

1. **配置 API Key**

在项目根目录创建 `.env` 文件:
```bash
# XPULink API Key
XPULINK_API_KEY=your_api_key_here
```

2. **安装依赖**
```bash
pip install requests python-dotenv
```

### 方法一: 使用 Jupyter Notebook (推荐新手)

这是最直观的学习方式,适合初学者。

```bash
cd LoRA
jupyter notebook lora_finetune_example.ipynb
```

按照 Notebook 中的步骤逐个执行单元格即可。

### 方法二: 使用 Python 脚本

适合有经验的开发者快速集成到现有系统。

1. **准备训练数据**

首先运行数据准备脚本生成示例数据:
```bash
cd LoRA
python prepare_training_data.py
```

或者创建自己的训练数据:
```python
from prepare_training_data import create_conversation, save_training_data

system_prompt = "你是一个专业的助手。"
conversations = [
    create_conversation(
        system_prompt,
        "用户问题1",
        "助手回答1"
    ),
    # 添加更多对话...
]

save_training_data(conversations, "data/my_training_data.jsonl")
```

2. **运行微调脚本**

修改 `lora_finetune.py` 文件末尾,取消注释示例代码:
```python
if __name__ == "__main__":
    # 取消下面这行的注释
    example_basic_usage()
```

然后运行:
```bash
python lora_finetune.py
```

3. **自定义微调流程**

```python
from lora_finetune import XPULinkLoRAFineTuner

# 初始化
finetuner = XPULinkLoRAFineTuner()

# 上传训练文件
file_id = finetuner.upload_training_file("data/my_training_data.jsonl")

# 创建微调任务
job_id = finetuner.create_finetune_job(
    training_file_id=file_id,
    model="qwen3-32b",
    suffix="my-custom-model",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "lora_r": 8,
        "lora_alpha": 16
    }
)

# 等待完成
status = finetuner.wait_for_completion(job_id)

# 测试模型
finetuned_model = status['fine_tuned_model']
finetuner.test_finetuned_model(finetuned_model, "测试问题")
```

## 💡 核心概念

### 什么是 LoRA?

LoRA (Low-Rank Adaptation) 是一种参数高效的微调技术:

**传统微调 vs LoRA:**
```
传统全参数微调:
- 训练所有模型参数 (数十亿参数)
- 需要大量 GPU 显存
- 训练时间长
- 成本高

LoRA 微调:
- 只训练少量适配器参数 (数百万参数)
- 显存需求低
- 训练速度快
- 成本低廉
- 效果接近全参数微调
```

**工作原理:**

LoRA 在模型的注意力层中插入低秩矩阵,只训练这些新增的参数:

```
原始权重矩阵: W ∈ R^(d×d)
LoRA 分解: ΔW = B × A
  其中: B ∈ R^(d×r), A ∈ R^(r×d)
  r << d (r 是秩,远小于 d)

最终权重: W' = W + α × ΔW
```

### 关键超参数说明

| 参数 | 说明 | 典型值 | 调整建议 |
|------|------|--------|----------|
| `n_epochs` | 训练轮数 | 3-5 | 数据少用3轮,数据多用5轮 |
| `batch_size` | 批次大小 | 2-8 | 数据量小用2-4,大用4-8 |
| `learning_rate` | 学习率 | 5e-5 | 效果不好可尝试 1e-5 到 1e-4 |
| `lora_r` | LoRA 秩 | 4-16 | 越大效果越好但参数越多 |
| `lora_alpha` | LoRA 缩放系数 | 16-32 | 通常设为 lora_r 的 2 倍 |
| `lora_dropout` | Dropout 比率 | 0.05-0.1 | 防止过拟合 |

## 📊 工作流程

```
1. 准备训练数据
   ↓
   格式化为 JSONL
   ↓
2. 上传到 XPULink 平台
   ↓
3. 配置超参数
   ↓
4. 创建微调任务
   ↓
5. 云端训练 (自动进行)
   - 加载基础模型 (Qwen3-32B)
   - 插入 LoRA 适配器
   - 训练 LoRA 参数
   - 验证和保存
   ↓
6. 获取微调模型
   ↓
7. 调用微调模型
```

## 🎓 使用场景

### 1. 企业知识库定制
将企业内部知识、产品信息、业务流程等注入模型。

**示例数据:**
```json
{
  "messages": [
    {"role": "system", "content": "你是ABC公司的AI助手,了解公司所有产品和服务。"},
    {"role": "user", "content": "我们公司的主打产品是什么?"},
    {"role": "assistant", "content": "ABC公司的主打产品包括:\n1. 企业级云计算平台...\n2. AI智能分析系统..."}
  ]
}
```

### 2. 特定领域专家
训练医疗、法律、金融等专业领域的对话模型。

**示例数据:**
```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业的法律咨询助手。"},
    {"role": "user", "content": "合同违约如何处理?"},
    {"role": "assistant", "content": "合同违约处理流程:\n1. 确认违约事实...\n2. 查看合同约定..."}
  ]
}
```

### 3. 特定风格输出
训练模型以特定语气、格式或风格输出内容。

**示例:**
- 技术文档撰写风格
- 营销文案创作风格
- 儿童友好语言风格

### 4. 任务特化
针对特定任务优化模型表现。

**示例:**
- 代码生成和调试
- 文本摘要
- 数据分析解释
- 翻译特定领域内容

## 🔧 高级配置

### 调整训练策略

根据数据量和任务复杂度调整:

**小数据集 (< 100 样本):**
```python
hyperparameters = {
    "n_epochs": 5,          # 增加训练轮数
    "batch_size": 2,        # 小批次
    "learning_rate": 1e-4,  # 较高学习率
    "lora_r": 4,            # 较小的秩
}
```

**中等数据集 (100-1000 样本):**
```python
hyperparameters = {
    "n_epochs": 3,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "lora_r": 8,
}
```

**大数据集 (> 1000 样本):**
```python
hyperparameters = {
    "n_epochs": 2,          # 减少轮数避免过拟合
    "batch_size": 8,        # 大批次
    "learning_rate": 2e-5,  # 较低学习率
    "lora_r": 16,           # 更大的秩
}
```

### 数据质量优化

**好的训练数据特征:**
```python
# ✅ 好的示例: 回答详细、准确、格式规范
{
  "messages": [
    {"role": "system", "content": "你是Python专家。"},
    {"role": "user", "content": "什么是装饰器?"},
    {"role": "assistant", "content": "装饰器是Python的高级特性...\n\n示例:\n```python\n@decorator\ndef function():\n    pass\n```"}
  ]
}

# ❌ 避免: 回答过于简短或模糊
{
  "messages": [
    {"role": "user", "content": "装饰器?"},
    {"role": "assistant", "content": "就是装饰函数的东西。"}
  ]
}
```

### 防止过拟合

如果发现模型过拟合(训练集表现好但泛化能力差):

1. **减少训练轮数**: `n_epochs` 从 5 降到 3
2. **增加 dropout**: `lora_dropout` 从 0.05 增到 0.1
3. **降低 LoRA 秩**: `lora_r` 从 16 降到 8
4. **增加训练数据多样性**

## 📈 效果评估

### 评估方法

1. **定性评估**
```python
# 准备测试问题
test_cases = [
    "测试问题1",
    "测试问题2",
    # ...
]

# 对比原模型和微调模型
for question in test_cases:
    print(f"问题: {question}")
    print(f"原模型: {test_base_model(question)}")
    print(f"微调模型: {test_finetuned_model(question)}")
    print("-" * 60)
```

2. **定量评估**
- 准备验证集
- 计算准确率、F1 分数等指标
- 对比微调前后的性能提升

### 迭代优化

```
第一次微调
   ↓
评估效果
   ↓
发现问题 → 调整数据/超参数
   ↓
第二次微调
   ↓
继续评估...
```

## 🔍 常见问题

### Q: 需要准备多少训练数据?

A:
- **最少**: 20-30 个高质量样本即可看到效果
- **推荐**: 50-100 个样本可获得较好效果
- **理想**: 200+ 个样本可获得最佳效果
- **关键**: 质量比数量更重要!

### Q: 微调需要多长时间?

A: 取决于数据量和超参数:
- 50 个样本,3 轮训练: 约 5-15 分钟
- 200 个样本,3 轮训练: 约 20-40 分钟
- 1000 个样本,3 轮训练: 约 1-2 小时

### Q: 微调后的模型会忘记原有能力吗?

A: LoRA 微调通常**不会**导致灾难性遗忘:
- LoRA 只添加适配器,不修改原始权重
- 模型保留基础能力,只在特定任务上增强
- 如果担心,可以在训练数据中加入通用对话样本

### Q: 如何选择合适的 lora_r 值?

A:
- **lora_r = 4**: 参数最少,训练最快,适合简单任务
- **lora_r = 8**: 平衡选择,大多数场景推荐
- **lora_r = 16**: 参数更多,效果更好,适合复杂任务
- **lora_r = 32**: 接近全参数微调效果,但成本增加

### Q: 微调失败了怎么办?

A: 检查以下几点:
1. ✅ 数据格式是否正确 (使用 `validate_training_data()` 验证)
2. ✅ API Key 是否有效
3. ✅ 训练数据是否足够 (至少 20 个样本)
4. ✅ 超参数是否合理
5. ✅ 查看错误信息,联系 XPULink 支持

### Q: 可以微调后再次微调吗?

A:
- 技术上可行,但**不推荐**多次迭代微调
- 建议: 收集所有数据后一次性微调
- 如需更新: 基于原始 Qwen3-32B 用新数据重新微调

### Q: 微调成本如何?

A:
- LoRA 微调成本远低于全参数微调
- 具体费用请参考 XPULink 官方定价
- 建议先用小数据集测试,满意后再扩大规模

## 📚 数据准备最佳实践

### 1. 系统提示词 (System Prompt)

**好的系统提示词:**
```python
# ✅ 明确、具体、有约束
"你是一个专业的Python编程助手,擅长解释概念并提供代码示例。回答要简洁、准确,代码要包含注释。"

# ❌ 过于宽泛
"你是一个助手。"
```

### 2. 对话质量

**高质量对话特征:**
- ✅ 回答详细但不冗长
- ✅ 使用具体示例
- ✅ 格式规范统一
- ✅ 语言准确专业
- ✅ 包含必要的警告或注意事项

### 3. 数据多样性

确保训练数据覆盖:
- 不同类型的问题
- 不同难度级别
- 不同表达方式
- 边界情况和特殊场景

### 4. 数据平衡

```python
# 避免数据不平衡
# ❌ 错误: 100个问题都是关于同一个话题
# ✅ 正确: 均匀分布在不同话题上

topic_distribution = {
    "基础语法": 30,
    "数据结构": 25,
    "文件操作": 20,
    "面向对象": 15,
    "高级特性": 10
}
```

## 🎯 成功案例参考

### 案例 1: Python 编程助手

**目标**: 创建专注于 Python 教学的助手

**数据准备**:
- 60 个 Python 概念解释
- 40 个代码调试示例

**超参数**:
```python
{
    "n_epochs": 3,
    "lora_r": 8,
    "learning_rate": 5e-5
}
```

**效果**: 对 Python 问题的回答更加专业和结构化

### 案例 2: 企业客服助手

**目标**: 训练了解公司产品的客服机器人

**数据准备**:
- 100 个常见客户问题及标准回答
- 50 个多轮对话示例

**超参数**:
```python
{
    "n_epochs": 4,
    "lora_r": 8,
    "learning_rate": 5e-5
}
```

**效果**: 准确回答公司产品相关问题,客户满意度提升

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这些示例!

## 📝 许可证

MIT License

## ⚠️ 重要提示

1. **本示例基于 OpenAI Fine-tuning API 标准编写**
   - 实际 XPULink API 接口可能有所不同
   - 使用前请参考 [XPULink 官方文档](https://www.xpulink.ai)

2. **妥善保管 API Key**
   - 不要将 `.env` 文件提交到版本控制
   - 不要在代码中硬编码 API Key

3. **数据安全**
   - 不要上传包含敏感信息的训练数据
   - 遵守数据隐私和安全法规

4. **合理使用**
   - 遵守 XPULink 服务条款
   - 不要训练生成有害内容的模型

---

**需要帮助?** 访问 [XPULink 官网](https://www.xpulink.ai) 或查看官方文档。
