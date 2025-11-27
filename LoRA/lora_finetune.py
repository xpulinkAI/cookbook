"""
Qwen3-32B LoRA 微调示例

本脚本展示如何使用 XPULink API 对 Qwen3-32B 模型进行 LoRA (Low-Rank Adaptation) 微调。
LoRA 是一种参数高效的微调方法,可以用较少的计算资源对大型语言模型进行定制化训练。

使用 LiteLLM 提供统一的 LLM 接口，优雅地支持自定义模型 API。

作者: XPULink
日期: 2025-01
"""

import os
import json
import requests
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from litellm import completion

# 加载环境变量
load_dotenv()


class XPULinkLoRAFineTuner:
    """XPULink LoRA 微调管理类"""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://www.xpulink.ai/v1"):
        """
        初始化 LoRA 微调器

        Args:
            api_key: XPULink API Key (如果不提供,会从环境变量 XPULINK_API_KEY 读取)
            base_url: API 基础 URL
        """
        self.api_key = api_key or os.getenv("XPULINK_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 API Key,请设置 XPULINK_API_KEY 环境变量或传入 api_key 参数")

        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def prepare_training_data(self, conversations: List[Dict[str, str]], output_file: str):
        """
        准备训练数据,转换为 JSONL 格式

        Args:
            conversations: 对话列表,每个对话包含多轮交互
            output_file: 输出文件路径

        示例对话格式:
        [
            {
                "messages": [
                    {"role": "system", "content": "你是一个有帮助的AI助手。"},
                    {"role": "user", "content": "什么是机器学习?"},
                    {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
                ]
            }
        ]
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for conversation in conversations:
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')

        print(f"✅ 训练数据已保存到: {output_file}")
        print(f"📊 总对话数: {len(conversations)}")
        return output_file

    def upload_training_file(self, file_path: str) -> str:
        """
        上传训练文件到 XPULink

        Args:
            file_path: 训练数据文件路径

        Returns:
            file_id: 上传后的文件 ID
        """
        url = f"{self.base_url}/v1/files"

        with open(file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(file_path), f, 'application/json'),
                'purpose': (None, 'fine-tune')
            }

            # 移除 Content-Type header,让 requests 自动处理 multipart/form-data
            headers = {"Authorization": f"Bearer {self.api_key}"}

            response = requests.post(url, headers=headers, files=files, timeout=60)

            if response.status_code != 200:
                raise Exception(f"文件上传失败: {response.text}")

            result = response.json()
            file_id = result.get('id')

            print(f"✅ 文件上传成功!")
            print(f"📄 文件 ID: {file_id}")

            return file_id

    def create_finetune_job(
        self,
        training_file_id: str,
        model: str = "qwen3-32b",
        suffix: Optional[str] = None,
        hyperparameters: Optional[Dict] = None
    ) -> str:
        """
        创建 LoRA 微调任务

        Args:
            training_file_id: 训练文件 ID
            model: 基础模型名称
            suffix: 微调模型的后缀名称
            hyperparameters: 超参数配置

        Returns:
            job_id: 微调任务 ID
        """
        url = f"{self.base_url}/fine_tuning/jobs"

        # 默认超参数
        default_hyperparams = {
            "n_epochs": 3,              # 训练轮数
            "batch_size": 4,            # 批次大小
            "learning_rate": 5e-5,      # 学习率
            "lora_r": 8,                # LoRA 秩
            "lora_alpha": 16,           # LoRA alpha 参数
            "lora_dropout": 0.05        # LoRA dropout
        }

        if hyperparameters:
            default_hyperparams.update(hyperparameters)

        payload = {
            "training_file": training_file_id,
            "model": model,
            "hyperparameters": default_hyperparams
        }

        if suffix:
            payload["suffix"] = suffix

        response = requests.post(url, headers=self.headers, json=payload, timeout=30)

        if response.status_code not in [200, 201]:
            raise Exception(f"创建微调任务失败: {response.text}")

        result = response.json()
        job_id = result.get('id')

        print(f"✅ 微调任务创建成功!")
        print(f"🆔 任务 ID: {job_id}")
        print(f"📊 超参数配置:")
        for key, value in default_hyperparams.items():
            print(f"   - {key}: {value}")

        return job_id

    def check_job_status(self, job_id: str) -> Dict:
        """
        检查微调任务状态

        Args:
            job_id: 微调任务 ID

        Returns:
            任务状态信息
        """
        url = f"{self.base_url}/v1/fine_tuning/jobs/{job_id}"

        response = requests.get(url, headers=self.headers, timeout=30)

        if response.status_code != 200:
            raise Exception(f"获取任务状态失败: {response.text}")

        return response.json()

    def wait_for_completion(self, job_id: str, check_interval: int = 30) -> Dict:
        """
        等待微调任务完成

        Args:
            job_id: 微调任务 ID
            check_interval: 检查间隔(秒)

        Returns:
            最终任务状态
        """
        print(f"⏳ 等待微调任务完成... (任务 ID: {job_id})")

        while True:
            status = self.check_job_status(job_id)
            current_status = status.get('status')

            print(f"📊 当前状态: {current_status}")

            if current_status == 'succeeded':
                print(f"✅ 微调任务完成!")
                print(f"🎉 微调模型: {status.get('fine_tuned_model')}")
                return status

            elif current_status == 'failed':
                error = status.get('error', '未知错误')
                raise Exception(f"❌ 微调任务失败: {error}")

            elif current_status in ['cancelled', 'canceled']:
                raise Exception(f"❌ 微调任务已取消")

            # 仍在进行中
            time.sleep(check_interval)

    def list_finetune_jobs(self, limit: int = 10) -> List[Dict]:
        """
        列出微调任务

        Args:
            limit: 返回的任务数量限制

        Returns:
            微调任务列表
        """
        url = f"{self.base_url}/v1/fine_tuning/jobs?limit={limit}"

        response = requests.get(url, headers=self.headers, timeout=30)

        if response.status_code != 200:
            raise Exception(f"获取任务列表失败: {response.text}")

        return response.json().get('data', [])

    def test_finetuned_model(self, model_name: str, test_prompt: str, max_tokens: int = 200):
        """
        测试微调后的模型（使用 LiteLLM）

        Args:
            model_name: 微调模型名称
            test_prompt: 测试提示词
            max_tokens: 最大生成 token 数
        """
        try:
            # 使用 LiteLLM 调用模型
            response = completion(
                model=f"openai/{model_name}",  # LiteLLM 格式
                messages=[
                    {"role": "user", "content": test_prompt}
                ],
                api_key=self.api_key,
                api_base=self.base_url,
                custom_llm_provider="openai",  # 指定 OpenAI 风格 API
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=60
            )

            content = response.choices[0].message.content
            print(f"🤖 模型回答:\n{content}")
            return content

        except Exception as e:
            raise Exception(f"模型调用失败: {str(e)}")


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("Qwen3-32B LoRA 微调基础示例")
    print("=" * 60)

    # 1. 初始化微调器
    finetuner = XPULinkLoRAFineTuner()

    # 2. 准备训练数据
    training_data = [
        {
            "messages": [
                {"role": "system", "content": "你是一个专业的Python编程助手。"},
                {"role": "user", "content": "如何在Python中读取JSON文件?"},
                {"role": "assistant", "content": "在Python中读取JSON文件可以使用json模块:\n\n```python\nimport json\n\nwith open('data.json', 'r', encoding='utf-8') as f:\n    data = json.load(f)\n```"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "你是一个专业的Python编程助手。"},
                {"role": "user", "content": "什么是列表推导式?"},
                {"role": "assistant", "content": "列表推导式是Python中创建列表的简洁方法:\n\n```python\n# 基本语法\nnumbers = [x for x in range(10)]\n\n# 带条件\neven_numbers = [x for x in range(10) if x % 2 == 0]\n```"}
            ]
        },
        # 可以添加更多训练样本...
    ]

    # 保存训练数据
    data_file = finetuner.prepare_training_data(
        training_data,
        "medical_qa_training.jsonl"
    )

    # 3. 上传训练文件
    print("\n📤 上传训练文件...")
    file_id = finetuner.upload_training_file(data_file)

    # 4. 创建微调任务
    print("\n🚀 创建微调任务...")
    job_id = finetuner.create_finetune_job(
        training_file_id=file_id,
        model="qwen3-32b",
        suffix="python-assistant",
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "lora_r": 8
        }
    )

    # 5. 等待微调完成
    print("\n⏳ 开始微调...")
    final_status = finetuner.wait_for_completion(job_id)

    # 6. 测试微调模型
    finetuned_model = final_status.get('fine_tuned_model')
    if finetuned_model:
        print(f"\n🧪 测试微调模型: {finetuned_model}")
        finetuner.test_finetuned_model(
            finetuned_model,
            "如何在Python中处理异常?"
        )


def example_check_existing_jobs():
    """检查现有微调任务的示例"""
    print("=" * 60)
    print("查看现有微调任务")
    print("=" * 60)

    finetuner = XPULinkLoRAFineTuner()

    jobs = finetuner.list_finetune_jobs(limit=5)

    if not jobs:
        print("📭 暂无微调任务")
        return

    print(f"📋 共找到 {len(jobs)} 个微调任务:\n")

    for i, job in enumerate(jobs, 1):
        print(f"{i}. 任务 ID: {job.get('id')}")
        print(f"   状态: {job.get('status')}")
        print(f"   模型: {job.get('model')}")
        print(f"   创建时间: {job.get('created_at')}")
        if job.get('fine_tuned_model'):
            print(f"   微调模型: {job.get('fine_tuned_model')}")
        print()


if __name__ == "__main__":
    # 运行基础示例
    # 注意: 这只是示例代码,实际使用时需要根据 XPULink API 的实际接口进行调整

    print("""
    ⚠️  使用前请确保:
    1. 已设置环境变量 XPULINK_API_KEY
    2. 已准备好足够的训练数据 (建议至少 50+ 样本)
    3. 了解 XPULink 平台的微调 API 文档

    本脚本提供的是通用的 LoRA 微调流程示例。
    实际 API 接口可能有所不同,请根据官方文档进行调整。
    """)

    # 取消注释以运行示例
    example_basic_usage()
    # example_check_existing_jobs()
