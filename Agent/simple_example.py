"""
简单示例：快速使用设备监控Agent
"""

import json
from device_agent import DeviceMonitorAgent


def main():
    """简单使用示例"""

    print("=" * 80)
    print("设备监控Agent - 快速示例")
    print("=" * 80)
    print()

    # 1. 初始化Agent
    print("步骤1: 初始化Agent...")
    try:
        agent = DeviceMonitorAgent()
        print("✓ Agent初始化成功\n")
    except ValueError as e:
        print(f"✗ 初始化失败: {e}")
        print("请设置环境变量 XPULINK_API_KEY")
        return

    # 2. 加载设备数据
    print("步骤2: 加载设备数据...")
    try:
        with open('data/examples/device_data.json', 'r', encoding='utf-8') as f:
            device_data = json.load(f)

        print(f"✓ 已加载设备: {device_data['device_name']} ({device_data['device_id']})")
        print(f"  设备类型: {device_data['device_type']}")
        print(f"  运行状态: {device_data['status']['operational_status']}")
        print(f"  告警级别: {device_data['status']['alert_level']}")
        print()
    except FileNotFoundError:
        print("✗ 找不到示例数据文件")
        print("请确保在 DeviceAgent 目录下运行此脚本")
        return

    # 3. 进行设备状态分析
    print("步骤3: 分析设备状态...")
    print("这可能需要几秒钟...\n")

    try:
        result = agent.analyze_device_status(device_data)

        print("=" * 80)
        print("分析结果")
        print("=" * 80)
        print(result['analysis'])
        print()
        print("=" * 80)

    except Exception as e:
        print(f"✗ 分析失败: {e}")
        return

    # 4. 可选：导出报告
    print("\n步骤4: 导出报告...")
    try:
        output_path = "data/reports/quick_analysis.md"
        agent.export_report(result, output_path, format="markdown")
        print(f"✓ 报告已保存到: {output_path}")
    except Exception as e:
        print(f"⚠ 导出报告失败: {e}")

    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)
    print("\n提示：")
    print("- 查看 device_agent_example.ipynb 了解更多功能")
    print("- 查看 README.md 了解详细使用说明")
    print("- 使用你自己的设备数据替换示例数据")


if __name__ == "__main__":
    main()
