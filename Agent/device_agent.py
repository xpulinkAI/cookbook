"""
设备监控智能Agent
基于 Qwen3-32B 模型实现设备状态分析、故障诊断和运维建议生成
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any


class DeviceMonitorAgent:
    """
    设备监控智能Agent

    功能：
    - 设备状态实时监控
    - 历史日志智能分析
    - 维修记录关联分析
    - 故障诊断与预测
    - 运维建议生成
    - 自动化报告生成
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "qwen3-32b"):
        """
        初始化设备监控Agent

        Args:
            api_key: XPULink API Key，如未提供则从环境变量读取
            model: 使用的模型名称，默认为 qwen3-32b
        """
        self.api_key = api_key or os.getenv("XPULINK_API_KEY")
        if not self.api_key:
            raise ValueError("请提供 API Key 或在环境变量中设置 XPULINK_API_KEY")

        self.model = model
        self.base_url = "https://www.xpulink.net/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Agent 系统提示词
        self.system_prompt = """你是一个专业的工业设备监控和运维专家AI助手。你的职责包括：

1. **设备状态分析**：根据设备运行数据，判断设备是否正常运行
2. **故障诊断**：分析异常数据、错误日志，定位故障原因
3. **预防性维护**：基于历史数据预测潜在问题
4. **运维建议**：提供专业的维修和维护建议
5. **报告生成**：生成结构化的诊断报告

你的分析应该：
- 准确：基于数据和事实进行判断
- 全面：考虑多个维度（性能、日志、历史记录）
- 专业：使用专业术语，提供技术细节
- 可操作：给出具体的下一步操作建议
- 结构化：按照清晰的格式组织信息

在回答时，请保持专业、客观、详细。"""

    def _call_llm(self, messages: List[Dict[str, str]],
                  temperature: float = 0.7,
                  max_tokens: int = 2000) -> str:
        """
        调用大语言模型

        Args:
            messages: 对话消息列表
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数

        Returns:
            模型生成的文本内容
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API 调用失败: {str(e)}")

    def analyze_device_status(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析设备运行状态

        Args:
            device_data: 设备数据字典，包含设备ID、传感器读数、运行参数等

        Returns:
            分析结果字典，包含状态、问题、建议等
        """
        # 构造分析提示
        prompt = f"""请分析以下设备的运行状态：

设备信息：
```json
{json.dumps(device_data, ensure_ascii=False, indent=2)}
```

请从以下几个方面进行分析：
1. 整体状态评估（正常/警告/异常/故障）
2. 关键指标分析（温度、压力、振动、能耗等）
3. 潜在风险识别
4. 异常原因分析（如有）

请以JSON格式返回结果，包含以下字段：
- status: 设备状态（normal/warning/abnormal/fault）
- health_score: 健康度评分（0-100）
- key_findings: 关键发现列表
- risks: 潜在风险列表
- anomalies: 异常项列表（如有）
- summary: 简要总结
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, temperature=0.3)

        return {
            "device_id": device_data.get("device_id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "analysis": response,
            "raw_data": device_data
        }

    def analyze_logs(self, logs: List[str], context: Optional[str] = None) -> Dict[str, Any]:
        """
        分析设备日志

        Args:
            logs: 日志列表
            context: 额外的上下文信息

        Returns:
            日志分析结果
        """
        logs_text = "\n".join([f"[{i+1}] {log}" for i, log in enumerate(logs)])

        prompt = f"""请分析以下设备日志：

日志内容：
```
{logs_text}
```
"""

        if context:
            prompt += f"\n上下文信息：\n{context}\n"

        prompt += """
请进行以下分析：
1. 错误和警告统计
2. 关键事件识别
3. 错误模式分析
4. 时间序列分析（如有时间戳）
5. 根本原因推断

请以JSON格式返回结果，包含：
- error_count: 错误数量
- warning_count: 警告数量
- critical_events: 关键事件列表
- error_patterns: 错误模式
- root_causes: 可能的根本原因
- summary: 分析总结
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, temperature=0.3)

        return {
            "timestamp": datetime.now().isoformat(),
            "log_count": len(logs),
            "analysis": response
        }

    def analyze_maintenance_history(self,
                                    maintenance_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析维修历史记录

        Args:
            maintenance_records: 维修记录列表

        Returns:
            维修历史分析结果
        """
        records_text = json.dumps(maintenance_records, ensure_ascii=False, indent=2)

        prompt = f"""请分析以下设备的维修历史记录：

维修记录：
```json
{records_text}
```

请进行以下分析：
1. 故障频率和类型统计
2. 维修成本分析
3. 故障模式识别
4. 高频故障部件识别
5. 维护周期评估
6. 预防性维护建议

请以JSON格式返回结果，包含：
- total_repairs: 总维修次数
- frequent_issues: 高频问题列表
- cost_analysis: 成本分析
- failure_patterns: 故障模式
- maintenance_suggestions: 维护建议
- summary: 分析总结
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, temperature=0.3)

        return {
            "timestamp": datetime.now().isoformat(),
            "records_analyzed": len(maintenance_records),
            "analysis": response
        }

    def comprehensive_diagnosis(self,
                               device_data: Optional[Dict[str, Any]] = None,
                               logs: Optional[List[str]] = None,
                               maintenance_records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        综合诊断：整合设备数据、日志和维修记录进行全面分析

        Args:
            device_data: 设备当前状态数据
            logs: 历史日志
            maintenance_records: 维修历史记录

        Returns:
            综合诊断报告
        """
        # 收集各部分分析结果
        analyses = []

        if device_data:
            device_analysis = self.analyze_device_status(device_data)
            analyses.append(f"## 设备状态分析\n{device_analysis['analysis']}")

        if logs:
            log_analysis = self.analyze_logs(logs)
            analyses.append(f"## 日志分析\n{log_analysis['analysis']}")

        if maintenance_records:
            maintenance_analysis = self.analyze_maintenance_history(maintenance_records)
            analyses.append(f"## 维修历史分析\n{maintenance_analysis['analysis']}")

        # 综合诊断
        combined_analysis = "\n\n".join(analyses)

        prompt = f"""基于以下各项分析结果，请提供一个综合诊断报告：

{combined_analysis}

请生成一个完整的综合诊断报告，包含：
1. **执行摘要**：设备整体健康状况和关键发现
2. **详细诊断**：
   - 当前状态评估
   - 识别的问题和风险
   - 根本原因分析
   - 趋势分析
3. **优先级排序**：按紧急程度排列需要处理的问题
4. **行动建议**：
   - 立即行动项（紧急）
   - 短期行动项（1周内）
   - 中期行动项（1个月内）
   - 长期优化建议
5. **预防措施**：避免问题复发的建议
6. **资源需求**：所需的人员、配件、时间估算

请使用清晰的Markdown格式组织报告。"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        report = self._call_llm(messages, temperature=0.5, max_tokens=3000)

        return {
            "timestamp": datetime.now().isoformat(),
            "report_type": "comprehensive_diagnosis",
            "device_id": device_data.get("device_id") if device_data else "unknown",
            "report": report,
            "component_analyses": {
                "device_status": device_data is not None,
                "logs": logs is not None and len(logs) > 0,
                "maintenance_history": maintenance_records is not None and len(maintenance_records) > 0
            }
        }

    def generate_action_plan(self, diagnosis_report: str,
                            priority: str = "all") -> Dict[str, Any]:
        """
        根据诊断报告生成行动计划

        Args:
            diagnosis_report: 诊断报告内容
            priority: 优先级过滤（immediate/short_term/medium_term/long_term/all）

        Returns:
            行动计划
        """
        priority_desc = {
            "immediate": "需要立即执行的紧急行动",
            "short_term": "短期行动（1周内）",
            "medium_term": "中期行动（1个月内）",
            "long_term": "长期优化行动",
            "all": "所有优先级的行动"
        }

        prompt = f"""基于以下诊断报告，生成详细的行动计划：

{diagnosis_report}

请生成{priority_desc.get(priority, '所有')}的详细行动计划，对每个行动项包含：
1. 行动描述
2. 优先级（P0-紧急/P1-高/P2-中/P3-低）
3. 预计耗时
4. 所需资源（人员、配件、工具）
5. 执行步骤
6. 成功标准
7. 风险和依赖项

请以JSON格式返回，包含action_items数组。"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self._call_llm(messages, temperature=0.4, max_tokens=2500)

        return {
            "timestamp": datetime.now().isoformat(),
            "priority_filter": priority,
            "action_plan": response
        }

    def query(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        自由问答：根据上下文回答设备相关问题

        Args:
            question: 用户问题
            context: 相关上下文数据

        Returns:
            AI回答
        """
        prompt = question

        if context:
            context_str = json.dumps(context, ensure_ascii=False, indent=2)
            prompt = f"""相关上下文信息：
```json
{context_str}
```

问题：{question}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        return self._call_llm(messages, temperature=0.7)

    def export_report(self, report_data: Dict[str, Any],
                     output_path: str,
                     format: str = "markdown") -> str:
        """
        导出报告到文件

        Args:
            report_data: 报告数据
            output_path: 输出文件路径
            format: 输出格式（markdown/json/txt）

        Returns:
            输出文件路径
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                   exist_ok=True)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)

        elif format == "markdown":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# 设备诊断报告\n\n")
                f.write(f"**生成时间**: {report_data.get('timestamp', 'N/A')}\n\n")

                if "device_id" in report_data:
                    f.write(f"**设备ID**: {report_data['device_id']}\n\n")

                f.write("---\n\n")
                f.write(report_data.get("report", str(report_data)))

        else:  # txt
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(str(report_data))

        return output_path


# 便捷函数
def quick_diagnosis(device_data: Dict[str, Any],
                   api_key: Optional[str] = None) -> str:
    """
    快速诊断：一键分析设备状态

    Args:
        device_data: 设备数据
        api_key: API Key

    Returns:
        诊断结果文本
    """
    agent = DeviceMonitorAgent(api_key=api_key)
    result = agent.analyze_device_status(device_data)
    return result["analysis"]


if __name__ == "__main__":
    # 示例用法
    print("设备监控Agent初始化测试...")

    try:
        agent = DeviceMonitorAgent()
        print("✓ Agent初始化成功")

        # 测试设备数据
        test_device = {
            "device_id": "PUMP-001",
            "device_type": "水泵",
            "temperature": 85.5,
            "pressure": 3.2,
            "vibration": 2.8,
            "power_consumption": 45.2,
            "runtime_hours": 15420,
            "status": "running"
        }

        print("\n正在分析测试设备...")
        result = agent.analyze_device_status(test_device)
        print("\n分析结果：")
        print(result["analysis"])

    except Exception as e:
        print(f"✗ 错误: {e}")
