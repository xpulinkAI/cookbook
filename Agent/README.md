# 设备监控智能Agent

基于 **Qwen3-32B** 大语言模型的智能设备监控与运维Agent，提供设备状态分析、故障诊断、预防性维护建议和自动化报告生成功能。

## 📋 目录

- [功能特性](#功能特性)
- [应用场景](#应用场景)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [API参考](#api参考)
- [示例数据](#示例数据)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

## ✨ 功能特性

### 1. 🔍 设备状态实时分析
- 多维度传感器数据分析（温度、压力、振动、能耗等）
- 异常检测和预警
- 健康度评分（0-100）
- 潜在风险识别

### 2. 📊 历史日志智能分析
- 错误和警告模式识别
- 关键事件提取
- 时间序列趋势分析
- 根本原因推断

### 3. 🔧 维修记录关联分析
- 故障频率统计
- 高频故障部件识别
- 维护成本分析
- 预防性维护建议

### 4. 🎯 综合诊断报告
- 整合多数据源的全面诊断
- 结构化的Markdown报告
- 优先级排序的问题清单
- 详细的行动建议

### 5. 📋 智能行动计划
- 按优先级生成行动项
- 资源需求评估
- 执行步骤规划
- 成功标准定义

### 6. 💬 交互式问答
- 自然语言问答
- 上下文感知
- 专业技术建议

## 🎯 应用场景

### 工业制造
- 生产设备监控
- 生产线故障诊断
- 设备保养计划

### 能源电力
- 发电机组监控
- 变压器健康管理
- 配电设备运维

### 交通运输
- 车辆车队管理
- 铁路设备监控
- 机场设施维护

### 建筑设施
- 暖通空调系统
- 电梯设备管理
- 消防系统监控

### 数据中心
- 服务器监控
- 冷却系统管理
- UPS设备维护

## 🚀 快速开始

### 环境要求

- Python 3.8+
- XPULink API Key（从 [www.xpulink.ai](https://www.xpulink.ai) 获取）

### 安装依赖

```bash
pip install requests python-dotenv
```

### 配置API Key

方式一：设置环境变量
```bash
export XPULINK_API_KEY=your_api_key_here
```

方式二：在代码中指定
```python
from device_agent import DeviceMonitorAgent

agent = DeviceMonitorAgent(api_key="your_api_key_here")
```

### 5分钟快速体验

#### 方式一：使用Jupyter Notebook（推荐）

```bash
cd DeviceAgent
jupyter notebook device_agent_example.ipynb
```

按照Notebook中的步骤执行即可。

#### 方式二：Python脚本

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
print(result['analysis'])
```

#### 方式三：命令行测试

```bash
cd DeviceAgent
python device_agent.py
```

## 📖 使用指南

### 1. 设备状态分析

```python
from device_agent import DeviceMonitorAgent
import json

agent = DeviceMonitorAgent()

# 加载设备数据
with open('data/examples/device_data.json', 'r') as f:
    device_data = json.load(f)

# 进行分析
result = agent.analyze_device_status(device_data)

print(result['analysis'])
```

**输出示例：**
```json
{
  "status": "warning",
  "health_score": 75,
  "key_findings": [
    "温度超过正常阈值5.5°C",
    "振动水平偏高",
    "功率消耗略有增加"
  ],
  "risks": [
    "持续高温可能导致密封件损坏",
    "振动增加可能引起轴承磨损"
  ],
  "summary": "设备整体运行正常但存在预警信号，建议尽快检查冷却系统和轴承状况"
}
```

### 2. 日志分析

```python
# 加载日志
with open('data/examples/device_logs.txt', 'r') as f:
    logs = [line.strip() for line in f.readlines()]

# 分析日志
result = agent.analyze_logs(
    logs,
    context="设备ID: PUMP-001, 设备类型: 离心泵"
)

print(result['analysis'])
```

### 3. 维修历史分析

```python
# 加载维修记录
with open('data/examples/maintenance_records.json', 'r') as f:
    maintenance_records = json.load(f)

# 分析维修历史
result = agent.analyze_maintenance_history(maintenance_records)

print(result['analysis'])
```

### 4. 综合诊断

```python
# 综合所有数据源
comprehensive_report = agent.comprehensive_diagnosis(
    device_data=device_data,
    logs=logs[:30],  # 最近30条日志
    maintenance_records=maintenance_records
)

print(comprehensive_report['report'])

# 导出报告
agent.export_report(
    comprehensive_report,
    output_path="reports/diagnosis_report.md",
    format="markdown"
)
```

### 5. 生成行动计划

```python
# 基于诊断报告生成行动计划
action_plan = agent.generate_action_plan(
    diagnosis_report=comprehensive_report['report'],
    priority="immediate"  # immediate/short_term/medium_term/long_term/all
)

print(action_plan['action_plan'])
```

### 6. 交互式问答

```python
# 询问关于设备的问题
question = "当前设备最需要关注的问题是什么？"

context = {
    "device_data": device_data,
    "recent_logs": logs[:10]
}

answer = agent.query(question, context=context)
print(answer)
```

## 📚 API参考

### DeviceMonitorAgent 类

#### 初始化

```python
agent = DeviceMonitorAgent(api_key=None, model="qwen3-32b")
```

**参数：**
- `api_key` (str, optional): XPULink API Key，如未提供则从环境变量读取
- `model` (str): 使用的模型名称，默认为 "qwen3-32b"

#### 主要方法

##### analyze_device_status()

分析设备运行状态

```python
result = agent.analyze_device_status(device_data: Dict[str, Any]) -> Dict[str, Any]
```

**参数：**
- `device_data`: 设备数据字典

**返回：**
- 包含分析结果的字典

##### analyze_logs()

分析设备日志

```python
result = agent.analyze_logs(
    logs: List[str],
    context: Optional[str] = None
) -> Dict[str, Any]
```

**参数：**
- `logs`: 日志列表
- `context`: 额外的上下文信息

**返回：**
- 日志分析结果字典

##### analyze_maintenance_history()

分析维修历史记录

```python
result = agent.analyze_maintenance_history(
    maintenance_records: List[Dict[str, Any]]
) -> Dict[str, Any]
```

**参数：**
- `maintenance_records`: 维修记录列表

**返回：**
- 维修历史分析结果字典

##### comprehensive_diagnosis()

综合诊断

```python
result = agent.comprehensive_diagnosis(
    device_data: Optional[Dict[str, Any]] = None,
    logs: Optional[List[str]] = None,
    maintenance_records: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]
```

**参数：**
- `device_data`: 设备当前状态数据
- `logs`: 历史日志
- `maintenance_records`: 维修历史记录

**返回：**
- 综合诊断报告字典

##### generate_action_plan()

生成行动计划

```python
result = agent.generate_action_plan(
    diagnosis_report: str,
    priority: str = "all"
) -> Dict[str, Any]
```

**参数：**
- `diagnosis_report`: 诊断报告内容
- `priority`: 优先级过滤（immediate/short_term/medium_term/long_term/all）

**返回：**
- 行动计划字典

##### query()

自由问答

```python
answer = agent.query(
    question: str,
    context: Optional[Dict[str, Any]] = None
) -> str
```

**参数：**
- `question`: 用户问题
- `context`: 相关上下文数据

**返回：**
- AI回答文本

##### export_report()

导出报告

```python
path = agent.export_report(
    report_data: Dict[str, Any],
    output_path: str,
    format: str = "markdown"
) -> str
```

**参数：**
- `report_data`: 报告数据
- `output_path`: 输出文件路径
- `format`: 输出格式（markdown/json/txt）

**返回：**
- 输出文件路径

### 便捷函数

#### quick_diagnosis()

快速诊断函数

```python
from device_agent import quick_diagnosis

result = quick_diagnosis(device_data, api_key=None)
```

## 📁 示例数据

项目包含完整的示例数据，位于 `data/examples/` 目录：

### device_data.json
包含完整的设备状态数据，包括：
- 设备基本信息
- 传感器读数（温度、压力、振动等）
- 性能指标
- 告警信息
- 环境条件

### device_logs.txt
50条真实感的设备运行日志，包括：
- 信息日志
- 警告日志
- 错误日志
- 时间戳

### maintenance_records.json
8条完整的维修记录，包括：
- 预防性维护
- 故障维修
- 更换配件
- 成本信息
- 技术发现

## 🎓 最佳实践

### 1. 数据准备

**设备数据规范：**
```python
device_data = {
    "device_id": "唯一设备ID",
    "device_type": "设备类型",
    "sensors": {
        "sensor_name": {
            "value": 当前值,
            "unit": "单位",
            "threshold_max": 最大阈值,
            "threshold_min": 最小阈值
        }
    },
    "status": "运行状态",
    # 其他相关字段
}
```

**日志格式建议：**
```
YYYY-MM-DD HH:MM:SS [LEVEL] DEVICE-ID: Message
```

**维修记录格式：**
```json
{
  "date": "YYYY-MM-DD",
  "type": "preventive/corrective",
  "description": "描述",
  "work_performed": ["工作项1", "工作项2"],
  "parts_replaced": [{"part_name": "名称", "cost": 成本}],
  "findings": "发现",
  "recommendations": "建议"
}
```

### 2. 成本控制

- 使用 `temperature` 参数控制输出的随机性（0.3用于诊断，0.7用于创意回答）
- 合理设置 `max_tokens` 限制输出长度
- 对于大量日志，只分析最近的关键日志（如最近50条）
- 使用缓存避免重复分析相同数据

### 3. 数据隐私

- 在生产环境中，确保敏感数据经过脱敏
- 不要在日志中包含个人信息或商业机密
- 导出的报告注意访问权限控制

### 4. 结果验证

- AI分析结果应作为辅助决策工具
- 关键决策需由专业人员验证
- 建立人工审核流程
- 记录AI建议的准确率

### 5. 集成建议

**与监控系统集成：**
```python
# 定时任务示例
import schedule
import time

def monitor_device():
    agent = DeviceMonitorAgent()
    device_data = get_device_data()  # 从监控系统获取数据
    result = agent.analyze_device_status(device_data)

    if result['health_score'] < 70:
        send_alert(result)  # 发送告警

schedule.every(1).hours.do(monitor_device)

while True:
    schedule.run_pending()
    time.sleep(60)
```

**与工单系统集成：**
```python
def auto_create_ticket():
    report = agent.comprehensive_diagnosis(...)
    action_plan = agent.generate_action_plan(report['report'], priority="immediate")

    # 自动创建工单
    create_ticket(
        title=f"设备 {device_id} 需要紧急维护",
        description=action_plan['action_plan'],
        priority="high"
    )
```

## ❓ 常见问题

### Q1: API调用失败怎么办？

**A:** 检查以下几点：
1. API Key是否正确配置
2. 网络连接是否正常
3. API配额是否充足
4. 请求超时设置是否合理（默认60秒）

```python
# 增加超时时间
agent = DeviceMonitorAgent()
agent.timeout = 120  # 设置为120秒
```

### Q2: 如何处理大量历史数据？

**A:** 建议分批处理：
```python
# 只分析最近的关键数据
recent_logs = logs[-50:]  # 最近50条
recent_maintenance = maintenance_records[:5]  # 最近5次维修

result = agent.comprehensive_diagnosis(
    device_data=device_data,
    logs=recent_logs,
    maintenance_records=recent_maintenance
)
```

### Q3: 分析结果不准确怎么办？

**A:** 尝试以下方法改进：
1. 提供更完整的设备数据
2. 增加上下文信息
3. 使用更低的temperature值（如0.2）提高确定性
4. 在问题中明确指出关注点

### Q4: 如何自定义分析维度？

**A:** 可以通过自由问答功能实现：
```python
custom_analysis = agent.query(
    question="""
    请从以下角度分析设备：
    1. 能源效率
    2. 环境影响
    3. 操作安全性
    """,
    context={"device_data": device_data}
)
```

### Q5: 支持哪些设备类型？

**A:** 理论上支持所有工业设备，包括但不限于：
- 泵、风机、压缩机
- 电机、变压器
- 机床、机器人
- 传送带、起重机
- 锅炉、冷却塔
- 等等

只要能提供设备运行数据，Agent就能进行分析。

### Q6: 如何提高分析速度？

**A:**
1. 减少输入数据量（只包含关键信息）
2. 降低 `max_tokens` 设置
3. 使用更快的模型（如有选择）
4. 实现异步调用

```python
import asyncio
import concurrent.futures

def parallel_analysis(devices):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(agent.analyze_device_status, device)
            for device in devices
        ]
        results = [f.result() for f in futures]
    return results
```

### Q7: 报告可以定制格式吗？

**A:** 可以，通过修改系统提示词或后处理：
```python
# 方式一：自定义提示词
agent.system_prompt += "\n\n请始终以表格形式展示关键指标。"

# 方式二：后处理
def format_report(report):
    # 添加公司logo、格式化等
    return formatted_report
```

## 📊 项目结构

```
DeviceAgent/
├── README.md                           # 本文档
├── device_agent.py                     # 核心Agent实现
├── device_agent_example.ipynb          # Jupyter Notebook示例
├── data/
│   ├── examples/                       # 示例数据
│   │   ├── device_data.json           # 设备数据示例
│   │   ├── device_logs.txt            # 日志示例
│   │   └── maintenance_records.json   # 维修记录示例
│   └── reports/                        # 报告输出目录（自动创建）
└── tests/                              # 测试文件（可选）
```

## 🔗 相关资源

- [XPULink官网](https://www.xpulink.ai)
- [Qwen3模型介绍](https://github.com/QwenLM/Qwen)
- [项目主README](../README.md)

## 📝 开发路线图

- [ ] 支持多设备批量分析
- [ ] 添加趋势预测功能
- [ ] 集成可视化图表
- [ ] 支持多语言报告
- [ ] 提供REST API接口
- [ ] 添加Web UI界面

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 💡 技术支持

如有问题或建议，请：
1. 查看本文档的[常见问题](#常见问题)部分
2. 访问 [XPULink官网](https://www.xpulink.ai) 获取API支持
3. 提交GitHub Issue

---

**注意**:
- 请妥善保管您的API Key，不要将其提交到公开仓库
- AI分析结果仅供参考，关键决策请由专业人员判断
- 注意API调用成本，合理使用资源
