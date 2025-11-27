"""
Qwen3-32B LoRA 微调 Streamlit 应用

本应用提供了一个友好的 Web 界面,用于上传训练数据并进行 LoRA 微调。

作者: XPULink
日期: 2025-01
"""

import streamlit as st
import json
import os
from io import StringIO
from typing import List, Dict
import pandas as pd
from lora_finetune import XPULinkLoRAFineTuner

# 设置页面配置
st.set_page_config(
    page_title="XPULink LoRA 微调平台",
    page_icon="🚀",
    layout="wide"
)

# 初始化 session state
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'file_id' not in st.session_state:
    st.session_state.file_id = None
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'finetuner' not in st.session_state:
    st.session_state.finetuner = None


def validate_jsonl_content(content: str) -> tuple[bool, str, List[Dict]]:
    """
    验证 JSONL 文件内容

    Args:
        content: JSONL 文件内容

    Returns:
        (是否有效, 错误信息, 解析后的数据)
    """
    lines = content.strip().split('\n')
    parsed_data = []

    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue

        try:
            data = json.loads(line)

            # 验证数据格式
            if 'messages' not in data:
                return False, f"第 {i} 行: 缺少 'messages' 字段", []

            messages = data['messages']
            if not isinstance(messages, list) or len(messages) == 0:
                return False, f"第 {i} 行: 'messages' 必须是非空数组", []

            # 验证每条消息格式
            for j, msg in enumerate(messages):
                if 'role' not in msg or 'content' not in msg:
                    return False, f"第 {i} 行, 消息 {j+1}: 缺少 'role' 或 'content' 字段", []

                if msg['role'] not in ['system', 'user', 'assistant']:
                    return False, f"第 {i} 行, 消息 {j+1}: role 必须是 'system', 'user' 或 'assistant'", []

            parsed_data.append(data)

        except json.JSONDecodeError as e:
            return False, f"第 {i} 行: JSON 解析错误 - {str(e)}", []

    if len(parsed_data) == 0:
        return False, "文件中没有有效的数据", []

    return True, "", parsed_data


def display_data_preview(data: List[Dict], max_samples: int = 3):
    """显示数据预览"""
    st.subheader("📊 数据预览")

    # 显示统计信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总对话数", len(data))
    with col2:
        total_messages = sum(len(d['messages']) for d in data)
        st.metric("总消息数", total_messages)
    with col3:
        avg_messages = total_messages / len(data) if len(data) > 0 else 0
        st.metric("平均消息数/对话", f"{avg_messages:.1f}")

    # 显示前几个样本
    st.write(f"**前 {min(max_samples, len(data))} 个对话样本:**")

    for i, conversation in enumerate(data[:max_samples], 1):
        with st.expander(f"对话 {i} - {len(conversation['messages'])} 条消息"):
            for msg in conversation['messages']:
                role = msg['role']
                content = msg['content']

                # 根据角色使用不同的样式
                if role == 'system':
                    st.info(f"**🔧 System**: {content}")
                elif role == 'user':
                    st.success(f"**👤 User**: {content}")
                else:  # assistant
                    st.warning(f"**🤖 Assistant**: {content}")


def initialize_finetuner(api_key: str):
    """初始化微调器"""
    try:
        finetuner = XPULinkLoRAFineTuner(api_key=api_key)
        st.session_state.finetuner = finetuner
        return True, finetuner
    except Exception as e:
        return False, str(e)


# 主界面
st.title("🚀 XPULink LoRA 微调平台")
st.markdown("---")

# 侧边栏 - API Key 配置
with st.sidebar:
    st.header("⚙️ 配置")

    api_key = st.text_input(
        "XPULink API Key",
        type="password",
        value=os.getenv("XPULINK_API_KEY", ""),
        help="输入您的 XPULink API Key"
    )

    if api_key:
        if st.session_state.finetuner is None:
            success, result = initialize_finetuner(api_key)
            if success:
                st.success("✅ API Key 已验证")
            else:
                st.error(f"❌ API Key 验证失败: {result}")
    else:
        st.warning("⚠️ 请输入 API Key")

    st.markdown("---")

    # 显示现有任务
    if st.button("🔄 刷新任务列表"):
        if st.session_state.finetuner:
            try:
                jobs = st.session_state.finetuner.list_finetune_jobs(limit=5)
                st.session_state.jobs = jobs
            except Exception as e:
                st.error(f"获取任务列表失败: {str(e)}")

    if 'jobs' in st.session_state and st.session_state.jobs:
        st.subheader("📋 最近的微调任务")
        for job in st.session_state.jobs[:3]:
            status = job.get('status', 'unknown')
            status_emoji = {
                'succeeded': '✅',
                'failed': '❌',
                'running': '⏳',
                'pending': '⏸️'
            }.get(status, '❓')

            st.text(f"{status_emoji} {job.get('id', '')[:8]}...")
            st.caption(f"状态: {status}")

# 主要内容区域
tab1, tab2, tab3 = st.tabs(["📁 上传数据", "🎯 配置微调", "📊 查看结果"])

# Tab 1: 上传数据
with tab1:
    st.header("📁 上传训练数据")

    st.markdown("""
    **支持的格式**: JSONL (JSON Lines)

    **数据格式示例**:
    ```json
    {"messages": [{"role": "system", "content": "你是一个有帮助的助手"}, {"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好!有什么我可以帮助你的吗?"}]}
    {"messages": [{"role": "user", "content": "什么是机器学习?"}, {"role": "assistant", "content": "机器学习是..."}]}
    ```
    """)

    # 文件上传
    uploaded_file = st.file_uploader(
        "选择 JSONL 文件",
        type=['jsonl', 'json'],
        help="上传包含训练数据的 JSONL 文件"
    )

    if uploaded_file is not None:
        # 读取文件内容
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        content = stringio.read()

        # 验证文件内容
        is_valid, error_msg, parsed_data = validate_jsonl_content(content)

        if is_valid:
            st.success(f"✅ 文件验证成功! 共 {len(parsed_data)} 条对话")
            st.session_state.training_data = parsed_data

            # 显示数据预览
            display_data_preview(parsed_data)

            # 保存到临时文件
            if st.button("💾 保存并准备上传"):
                temp_file_path = "LoRA/data/uploaded_training_data.jsonl"
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    for item in parsed_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

                st.session_state.temp_file_path = temp_file_path
                st.success(f"✅ 数据已保存到: {temp_file_path}")
                st.info("👉 请前往 '配置微调' 标签页继续")
        else:
            st.error(f"❌ 文件验证失败: {error_msg}")
            st.info("💡 请检查您的 JSONL 文件格式是否正确")

# Tab 2: 配置微调
with tab2:
    st.header("🎯 配置微调参数")

    if not st.session_state.training_data:
        st.warning("⚠️ 请先在 '上传数据' 标签页上传训练数据")
    elif not st.session_state.finetuner:
        st.warning("⚠️ 请先在侧边栏配置 API Key")
    else:
        st.success(f"✅ 已加载 {len(st.session_state.training_data)} 条训练数据")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("基础配置")

            model = st.selectbox(
                "基础模型",
                ["qwen3-32b", "qwen3-72b"],
                help="选择要微调的基础模型"
            )

            suffix = st.text_input(
                "模型后缀名",
                value="custom-model",
                help="为微调后的模型指定一个后缀名"
            )

        with col2:
            st.subheader("超参数配置")

            n_epochs = st.slider("训练轮数 (n_epochs)", 1, 10, 3)
            batch_size = st.slider("批次大小 (batch_size)", 1, 16, 4)
            learning_rate = st.select_slider(
                "学习率 (learning_rate)",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                value=5e-5,
                format_func=lambda x: f"{x:.0e}"
            )

            lora_r = st.slider("LoRA 秩 (lora_r)", 4, 64, 8)
            lora_alpha = st.slider("LoRA Alpha", 8, 128, 16)
            lora_dropout = st.slider("LoRA Dropout", 0.0, 0.3, 0.05, 0.01)

        st.markdown("---")

        # 显示配置摘要
        st.subheader("📋 配置摘要")
        config_summary = {
            "基础模型": model,
            "模型后缀": suffix,
            "训练样本数": len(st.session_state.training_data),
            "训练轮数": n_epochs,
            "批次大小": batch_size,
            "学习率": f"{learning_rate:.0e}",
            "LoRA 秩": lora_r,
            "LoRA Alpha": lora_alpha,
            "LoRA Dropout": lora_dropout
        }

        df = pd.DataFrame(list(config_summary.items()), columns=["参数", "值"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # 开始微调按钮
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 开始微调", type="primary", use_container_width=True):
                try:
                    # 1. 上传文件
                    with st.spinner("📤 正在上传训练文件..."):
                        file_id = st.session_state.finetuner.upload_training_file(
                            st.session_state.temp_file_path
                        )
                        st.session_state.file_id = file_id
                        st.success(f"✅ 文件上传成功! File ID: {file_id}")

                    # 2. 创建微调任务
                    with st.spinner("🎯 正在创建微调任务..."):
                        hyperparameters = {
                            "n_epochs": n_epochs,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "lora_r": lora_r,
                            "lora_alpha": lora_alpha,
                            "lora_dropout": lora_dropout
                        }

                        job_id = st.session_state.finetuner.create_finetune_job(
                            training_file_id=file_id,
                            model=model,
                            suffix=suffix,
                            hyperparameters=hyperparameters
                        )
                        st.session_state.job_id = job_id
                        st.success(f"✅ 微调任务创建成功! Job ID: {job_id}")

                    st.info("👉 请前往 '查看结果' 标签页查看训练进度")

                except Exception as e:
                    st.error(f"❌ 创建微调任务失败: {str(e)}")

# Tab 3: 查看结果
with tab3:
    st.header("📊 微调任务状态")

    if st.session_state.job_id:
        job_id = st.session_state.job_id

        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"**任务 ID**: `{job_id}`")
        with col2:
            if st.button("🔄 刷新状态"):
                st.rerun()

        if st.session_state.finetuner:
            try:
                status = st.session_state.finetuner.check_job_status(job_id)

                current_status = status.get('status', 'unknown')

                # 状态显示
                status_colors = {
                    'succeeded': 'success',
                    'failed': 'error',
                    'running': 'info',
                    'pending': 'warning'
                }
                status_color = status_colors.get(current_status, 'info')

                if status_color == 'success':
                    st.success(f"✅ 状态: {current_status}")
                elif status_color == 'error':
                    st.error(f"❌ 状态: {current_status}")
                elif status_color == 'info':
                    st.info(f"⏳ 状态: {current_status}")
                else:
                    st.warning(f"⏸️ 状态: {current_status}")

                # 显示详细信息
                st.json(status)

                # 如果成功,显示测试界面
                if current_status == 'succeeded':
                    st.success("🎉 微调完成!")

                    fine_tuned_model = status.get('fine_tuned_model')
                    if fine_tuned_model:
                        st.markdown("---")
                        st.subheader("🧪 测试微调模型")

                        st.info(f"**微调模型名称**: `{fine_tuned_model}`")

                        test_prompt = st.text_area(
                            "输入测试提示词",
                            height=100,
                            placeholder="输入您想测试的问题..."
                        )

                        max_tokens = st.slider("最大生成长度", 50, 1000, 200)

                        if st.button("🤖 测试模型"):
                            if test_prompt:
                                try:
                                    with st.spinner("🤔 模型思考中..."):
                                        response = st.session_state.finetuner.test_finetuned_model(
                                            fine_tuned_model,
                                            test_prompt,
                                            max_tokens
                                        )

                                    st.markdown("### 🤖 模型回答:")
                                    st.markdown(response)

                                except Exception as e:
                                    st.error(f"❌ 测试失败: {str(e)}")
                            else:
                                st.warning("⚠️ 请输入测试提示词")

            except Exception as e:
                st.error(f"❌ 获取任务状态失败: {str(e)}")
    else:
        st.info("ℹ️ 暂无正在进行的微调任务")
        st.markdown("请先在 '上传数据' 和 '配置微调' 标签页创建微调任务")

# 页脚
st.markdown("---")
st.caption("© 2025 XPULink - LoRA 微调平台")
