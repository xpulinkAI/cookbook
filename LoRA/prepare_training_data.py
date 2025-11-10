"""
训练数据准备工具

本脚本帮助用户将自己的数据转换为 LoRA 微调所需的 JSONL 格式。

作者: XPULink
日期: 2025-01
"""

import json
import os
from typing import List, Dict


def create_conversation(system_prompt: str, user_message: str, assistant_message: str) -> Dict:
    """
    创建单个对话样本

    Args:
        system_prompt: 系统提示词
        user_message: 用户消息
        assistant_message: 助手回复

    Returns:
        格式化的对话字典
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    }


def create_multi_turn_conversation(system_prompt: str, turns: List[Dict[str, str]]) -> Dict:
    """
    创建多轮对话样本

    Args:
        system_prompt: 系统提示词
        turns: 对话轮次列表,每个元素包含 'user' 和 'assistant' 键

    Returns:
        格式化的多轮对话字典
    """
    messages = [{"role": "system", "content": system_prompt}]

    for turn in turns:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    return {"messages": messages}


def save_training_data(conversations: List[Dict], output_file: str):
    """
    保存训练数据为 JSONL 格式

    Args:
        conversations: 对话列表
        output_file: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

    print(f"✅ 训练数据已保存到: {output_file}")
    print(f"📊 总对话数: {len(conversations)}")


def validate_training_data(file_path: str) -> bool:
    """
    验证训练数据格式是否正确

    Args:
        file_path: 训练数据文件路径

    Returns:
        是否验证通过
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                data = json.loads(line)

                # 检查必需字段
                if "messages" not in data:
                    print(f"❌ 第 {i} 行缺少 'messages' 字段")
                    return False

                messages = data["messages"]

                if not isinstance(messages, list) or len(messages) < 2:
                    print(f"❌ 第 {i} 行 'messages' 必须是包含至少2条消息的列表")
                    return False

                # 检查每条消息
                for j, msg in enumerate(messages):
                    if "role" not in msg or "content" not in msg:
                        print(f"❌ 第 {i} 行第 {j+1} 条消息缺少 'role' 或 'content' 字段")
                        return False

                    if msg["role"] not in ["system", "user", "assistant"]:
                        print(f"❌ 第 {i} 行第 {j+1} 条消息的 role 无效: {msg['role']}")
                        return False

        print(f"✅ 数据格式验证通过!")
        return True

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


# ============================================================================
# 示例 1: Python 编程助手训练数据
# ============================================================================

def example_python_tutor():
    """创建 Python 编程助手的训练数据"""
    system_prompt = "你是一个专业的Python编程助手,擅长解释Python概念并提供清晰的代码示例。"

    conversations = [
        create_conversation(
            system_prompt,
            "什么是Python列表推导式?",
            "列表推导式是Python中创建列表的简洁语法。\n\n基本格式:\n```python\n[表达式 for 变量 in 可迭代对象 if 条件]\n```\n\n示例:\n```python\n# 创建平方数列表\nsquares = [x**2 for x in range(10)]\n\n# 筛选偶数\nevens = [x for x in range(20) if x % 2 == 0]\n```"
        ),
        create_conversation(
            system_prompt,
            "如何在Python中读写文件?",
            "Python使用open()函数进行文件操作:\n\n**读取文件:**\n```python\n# 方法1: 使用with语句(推荐)\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n\n# 方法2: 逐行读取\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    for line in f:\n        print(line.strip())\n```\n\n**写入文件:**\n```python\nwith open('output.txt', 'w', encoding='utf-8') as f:\n    f.write('Hello, World!')\n```"
        ),
        create_conversation(
            system_prompt,
            "解释Python中的*args和**kwargs",
            "*args和**kwargs用于函数接受可变数量的参数:\n\n**args (可变位置参数):**\n```python\ndef sum_all(*args):\n    return sum(args)\n\nprint(sum_all(1, 2, 3, 4))  # 输出: 10\n```\n\n**kwargs (可变关键字参数):**\n```python\ndef print_info(**kwargs):\n    for key, value in kwargs.items():\n        print(f'{key}: {value}')\n\nprint_info(name='Alice', age=25)\n```"
        ),
    ]

    save_training_data(conversations, "data/python_tutor_training.jsonl")
    validate_training_data("data/python_tutor_training.jsonl")


# ============================================================================
# 示例 2: 客服对话训练数据
# ============================================================================

def example_customer_service():
    """创建客服对话的训练数据"""
    system_prompt = "你是一个专业友好的客服助手,负责帮助用户解决问题。"

    conversations = [
        create_multi_turn_conversation(
            system_prompt,
            [
                {
                    "user": "我的订单还没有收到,怎么办?",
                    "assistant": "非常抱歉给您带来不便。请提供您的订单号,我来帮您查询物流信息。"
                },
                {
                    "user": "订单号是 #12345",
                    "assistant": "好的,我查到您的订单已经发货3天了。根据物流信息,包裹目前在配送中,预计明天送达。您可以通过以下链接查看详细物流: [物流链接]"
                },
                {
                    "user": "好的,谢谢!",
                    "assistant": "不客气!如果明天还未收到包裹,请随时联系我们。祝您购物愉快!"
                }
            ]
        ),
        create_conversation(
            system_prompt,
            "你们支持退货吗?",
            "是的,我们支持退货。退货政策如下:\n\n1. 商品收到后7天内可申请退货\n2. 商品需保持原包装完整\n3. 不影响二次销售\n\n退货流程:\n1. 在订单页面点击'申请退货'\n2. 填写退货原因\n3. 等待审核通过\n4. 寄回商品\n5. 收到商品后3-5个工作日退款\n\n需要帮您申请退货吗?"
        ),
    ]

    save_training_data(conversations, "data/customer_service_training.jsonl")
    validate_training_data("data/customer_service_training.jsonl")


# ============================================================================
# 示例 3: 领域专家 (医疗咨询示例)
# ============================================================================

def example_medical_qa():
    """创建医疗咨询的训练数据 (仅作为格式示例)"""
    system_prompt = "你是一个医疗健康咨询助手,提供基础的健康建议。注意:严重疾病请咨询专业医生。"

    conversations = [
        create_conversation(
            system_prompt,
            "感冒了应该注意什么?",
            "感冒期间的注意事项:\n\n**休息和饮食:**\n- 保证充足睡眠,帮助身体恢复\n- 多喝温水,保持身体水分\n- 清淡饮食,多吃蔬菜水果\n\n**症状缓解:**\n- 适当运动,但避免剧烈活动\n- 保持室内通风\n- 可以喝姜茶、蜂蜜水缓解喉咙不适\n\n**就医建议:**\n- 如果发烧超过3天或症状加重,请及时就医\n- 特殊人群(老人、儿童、孕妇)建议咨询医生\n\n⚠️ 注意:本建议仅供参考,如有严重症状请及时就医。"
        ),
    ]

    save_training_data(conversations, "data/medical_qa_training.jsonl")
    validate_training_data("data/medical_qa_training.jsonl")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LoRA 训练数据准备工具")
    print("=" * 60)
    print()

    # 创建示例数据
    print("📝 生成示例训练数据...\n")

    print("1️⃣ 生成 Python 编程助手训练数据")
    example_python_tutor()
    print()

    print("2️⃣ 生成客服对话训练数据")
    example_customer_service()
    print()

    print("3️⃣ 生成医疗咨询训练数据(示例)")
    example_medical_qa()
    print()

    print("=" * 60)
    print("✅ 所有示例数据已生成!")
    print()
    print("💡 使用提示:")
    print("1. 参考上述示例创建您自己的训练数据")
    print("2. 使用 create_conversation() 创建单轮对话")
    print("3. 使用 create_multi_turn_conversation() 创建多轮对话")
    print("4. 使用 validate_training_data() 验证数据格式")
    print()
    print("⚠️  重要:")
    print("- 建议准备至少 50-100 个高质量训练样本")
    print("- 确保数据准确性和一致性")
    print("- 避免包含敏感或有害内容")
    print("=" * 60)
