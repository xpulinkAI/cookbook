"""
PDF RAG 系统 - 使用 BGE-M3 Embedding + LiteLLM

本脚本展示如何使用 XPULink 平台托管的模型构建 RAG（检索增强生成）系统。
使用 LiteLLM 优雅地支持自定义 OpenAI 风格 API，无需 hack 或绕过验证。

功能特性：
- 加载和处理 PDF 文档
- 使用 BGE-M3 Embedding 模型进行文档向量化
- 使用 LiteLLM 支持自定义 LLM（qwen3-32b）
- 构建向量索引实现高效检索
- 基于检索结果生成智能回答
- 预测向量数据库大小
- 交互式查询界面
- 自动重试机制处理网络问题

技术栈：
- LlamaIndex: RAG 框架
- LiteLLM: 统一 LLM 接口（支持自定义 API）
- BGE-M3: 多语言 Embedding 模型
- Qwen3-32B: 大语言模型

作者: XPULink
日期: 2025-01
"""

import os
import json
import requests
import time
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import Field, PrivateAttr

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.litellm import LiteLLM


class BGEM3Embedding(BaseEmbedding):
    """BGE-M3 Embedding 模型实现（基于 OpenAI 兼容 API）"""

    api_base: str = Field(default="https://xpulink.net/v1", description="XPULink API 基础地址")
    api_key: Optional[str] = Field(default=None, description="API 密钥")
    model_name: str = Field(default="bge-m3", description="模型名称")
    embed_batch_size: int = Field(default=10, description="批处理大小")

    def __init__(
        self,
        api_base: str = "https://xpulink.net/v1",
        api_key: Optional[str] = "",
        model: str = "bge-m3",
        embed_batch_size: int = 10,
        **kwargs
    ) -> None:
        """
        初始化 BGE-M3 Embedding 模型

        Args:
            api_base: XPULink API 基础地址
            api_key: API 密钥（从环境变量获取）
            model: 模型名称，默认为 bge-m3
            embed_batch_size: 批处理大小
        """
        # 处理 API key
        if api_key is None:
            api_key = os.getenv("XPU_API_KEY")

        if not api_key:
            raise ValueError("需要提供 API Key")

        # 处理 api_base
        api_base = api_base.rstrip('/')

        # 调用父类构造函数，传递所有参数
        super().__init__(
            api_base=api_base,
            api_key=api_key,
            model_name=model,
            embed_batch_size=embed_batch_size,
            **kwargs
        )

    def _call_api(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """
        调用 XPULink API 获取 embeddings，带重试机制

        Args:
            texts: 要处理的文本列表
            max_retries: 最大重试次数
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': self.model_name,
            'input': texts
        }

        last_exception = None

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/embeddings",
                    headers=headers,
                    json=data,
                    timeout=1000
                )
                response.raise_for_status()

                result = response.json()
                if result.get('data'):
                    return [item['embedding'] for item in result['data']]
                else:
                    raise Exception(f"API 返回格式错误: {result}")

            except (requests.exceptions.RequestException,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError) as e:
                last_exception = e

                if attempt < max_retries - 1:
                    # 指数退避: 等待 2^attempt 秒
                    wait_time = 2 ** attempt
                    print(f"⚠️  请求失败 (尝试 {attempt + 1}/{max_retries}), {wait_time}秒后重试...")
                    print(f"   错误: {str(e)}")
                    time.sleep(wait_time)
                else:
                    # 最后一次尝试也失败了
                    raise Exception(f"API 请求失败 (已重试{max_retries}次): {str(e)}")

        # 如果所有重试都失败
        raise Exception(f"API 请求失败: {str(last_exception)}")

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取单个查询的 embedding"""
        embeddings = self._call_api([query])
        return embeddings[0] if embeddings else []

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的 embedding"""
        embeddings = self._call_api([text])
        return embeddings[0] if embeddings else []

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的 embeddings"""
        all_embeddings = []

        # 分批处理
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            batch_embeddings = self._call_api(batch)
            all_embeddings.extend(batch_embeddings)

            if i + self.embed_batch_size < len(texts):
                print(f"已处理 {i + len(batch)}/{len(texts)} 个文本片段")

        return all_embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询 embedding（回退到同步方法）"""
        return self._get_query_embedding(query)


def estimate_vector_db_size(documents, embedding_dim=1024, dtype_bytes=4) -> Dict[str, Any]:
    """
    预测向量数据库的存储大小

    Args:
        documents: 文档列表
        embedding_dim: 向量维度（BGE-M3 默认为 1024）
        dtype_bytes: 数据类型字节数（float32 为 4 字节）

    Returns:
        dict: 包含各项大小统计的字典
    """
    if not documents:
        return {
            'error': '没有文档可供分析',
            'total_size_mb': 0
        }

    # 估算文档被分块的数量
    # LlamaIndex 默认 chunk_size=1024, chunk_overlap=20
    chunk_size = 1024
    chunk_overlap = 20
    effective_chunk_size = chunk_size - chunk_overlap

    total_chars = sum(len(doc.text) for doc in documents)
    estimated_chunks = max(1, total_chars // effective_chunk_size)

    # 计算向量存储大小
    # 每个 chunk 需要一个 embedding 向量
    vector_size_bytes = estimated_chunks * embedding_dim * dtype_bytes
    vector_size_mb = vector_size_bytes / (1024 * 1024)

    # 计算文本存储大小（UTF-8 编码，约每字符 2-3 字节，这里取 2.5）
    text_size_bytes = total_chars * 2.5
    text_size_mb = text_size_bytes / (1024 * 1024)

    # 计算元数据存储大小（估算每个 chunk 约 500 字节元数据）
    metadata_size_bytes = estimated_chunks * 500
    metadata_size_mb = metadata_size_bytes / (1024 * 1024)

    # 索引开销（FAISS 或其他索引结构，约为向量大小的 20-30%）
    index_overhead_factor = 0.25
    index_overhead_mb = vector_size_mb * index_overhead_factor

    # 总大小
    total_size_mb = vector_size_mb + text_size_mb + metadata_size_mb + index_overhead_mb

    return {
        'total_documents': len(documents),
        'total_characters': total_chars,
        'estimated_chunks': estimated_chunks,
        'embedding_dimension': embedding_dim,
        'vector_storage_mb': round(vector_size_mb, 2),
        'text_storage_mb': round(text_size_mb, 2),
        'metadata_storage_mb': round(metadata_size_mb, 2),
        'index_overhead_mb': round(index_overhead_mb, 2),
        'total_size_mb': round(total_size_mb, 2),
        'total_size_gb': round(total_size_mb / 1024, 3)
    }


def print_size_estimation(estimation: Dict[str, Any]) -> None:
    """打印格式化的大小预测结果"""
    if 'error' in estimation:
        print(f"❌ {estimation['error']}")
        return

    print("=" * 60)
    print("📊 向量数据库大小预测")
    print("=" * 60)
    print(f"\n📄 文档统计:")
    print(f"  - 文档数量: {estimation['total_documents']}")
    print(f"  - 总字符数: {estimation['total_characters']:,}")
    print(f"  - 预计分块数: {estimation['estimated_chunks']:,}")
    print(f"  - 向量维度: {estimation['embedding_dimension']}")

    print(f"\n💾 存储空间预测:")
    print(f"  - 向量存储: {estimation['vector_storage_mb']:,.2f} MB")
    print(f"  - 文本存储: {estimation['text_storage_mb']:,.2f} MB")
    print(f"  - 元数据存储: {estimation['metadata_storage_mb']:,.2f} MB")
    print(f"  - 索引开销: {estimation['index_overhead_mb']:,.2f} MB")

    print(f"\n📦 总计:")
    print(f"  - 总大小: {estimation['total_size_mb']:,.2f} MB ({estimation['total_size_gb']:.3f} GB)")

    # 添加建议
    total_mb = estimation['total_size_mb']
    print(f"\n💡 建议:")
    if total_mb < 100:
        print("  ✅ 内存占用较小，可以轻松在内存中处理")
    elif total_mb < 1000:
        print("  ⚠️  内存占用适中，确保有足够的可用内存")
    else:
        print("  ⚠️  内存占用较大，建议考虑:")
        print("     - 使用持久化向量数据库（如 Chroma、Weaviate）")
        print("     - 分批处理文档")
        print("     - 使用更强大的服务器")

    print("=" * 60)


def load_documents(data_dir: str = "./data/") -> Optional[List]:
    """
    加载 PDF 文档

    Args:
        data_dir: 数据目录路径

    Returns:
        文档列表，如果加载失败返回 None
    """
    # 确保数据目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 已创建数据目录: {data_dir}")
        print(f"⚠️  请将 PDF 文件放入此目录")
        return None

    # 加载文档
    try:
        documents = SimpleDirectoryReader(
            input_dir=data_dir,
            required_exts=[".pdf"]
        ).load_data()

        if documents:
            print(f"✅ 成功加载 {len(documents)} 个文档片段")
            print(f"\n📄 第一个文档片段预览:")
            print(f"  - 长度: {len(documents[0].text)} 字符")
            print(f"  - 内容预览: {documents[0].text[:200]}...")
            return documents
        else:
            print(f"⚠️  未在 {data_dir} 目录中找到 PDF 文件")
            print("请添加 PDF 文件后重试")
            return None

    except Exception as e:
        print(f"❌ 加载文档失败: {str(e)}")
        return None


def setup_rag_system(api_key: Optional[str] = None):
    """
    设置 RAG 系统配置

    Args:
        api_key: XPU API Key，如果不提供则从环境变量读取
    """
    # 加载环境变量
    load_dotenv()

    # 检查 API Key
    api_key = api_key or os.getenv('XPU_API_KEY')
    if not api_key:
        raise ValueError("❌ 未找到 XPU_API_KEY。请在 .env 文件中设置或作为参数传入。")

    print("✅ 找到 XPU_API_KEY 环境变量")

    # 配置 BGE-M3 Embedding 模型
    Settings.embed_model = BGEM3Embedding(
        api_base="https://www.xpulink.net/v1",
        model="bge-m3:latest",
        embed_batch_size=5  # 减少批次大小提高稳定性
    )

    # 配置 LLM（使用 LiteLLM - 优雅地支持自定义模型）
    Settings.llm = LiteLLM(
        model="openai/qwen3-32b",  # LiteLLM 格式: provider/model
        api_key=api_key,
        api_base="https://www.xpulink.net/v1",
        temperature=0.7,
        custom_llm_provider="openai"  # 指定这是 OpenAI 风格的 API
    )

    print("✅ LlamaIndex 配置完成（使用 LiteLLM）")
    print(f"  - Embedding 模型: {Settings.embed_model.model_name}")
    print(f"  - LLM 模型: qwen3-32b (via LiteLLM)")
    print(f"  - API 端点: https://www.xpulink.net/v1")


def build_index(documents: List, show_estimation: bool = True):
    """
    构建向量索引

    Args:
        documents: 文档列表
        show_estimation: 是否显示大小预测

    Returns:
        VectorStoreIndex 对象
    """
    if not documents:
        raise ValueError("文档列表为空")

    # 显示预测信息
    if show_estimation:
        print("🔄 正在分析文档并预测向量数据库大小...\n")
        estimation = estimate_vector_db_size(documents)
        print_size_estimation(estimation)
        print()

    print("🔄 开始构建向量索引...")
    print("   这可能需要几分钟时间，取决于文档大小\n")

    try:
        # 构建向量索引
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )

        print("\n✅ 向量索引构建完成！")
        print("   现在可以进行文档查询了")

        return index

    except Exception as e:
        print(f"❌ 构建索引失败: {str(e)}")
        raise


def create_query_engine(index, similarity_top_k: int = 3, response_mode: str = "compact"):
    """
    创建查询引擎

    Args:
        index: VectorStoreIndex 对象
        similarity_top_k: 返回最相似的 K 个片段
        response_mode: 响应模式 (compact/tree_summarize/refine)

    Returns:
        查询引擎对象
    """
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        response_mode=response_mode
    )

    print("✅ 查询引擎创建完成")
    print(f"  - 检索片段数: {similarity_top_k}")
    print(f"  - 响应模式: {response_mode}")

    return query_engine


def run_example_queries(query_engine):
    """运行示例查询"""
    print("\n" + "=" * 60)
    print("📝 运行示例查询")
    print("=" * 60)

    example_queries = [
        "文档的主要内容是什么？",
        "请总结文档中的关键要点"
    ]

    for i, query in enumerate(example_queries, 1):
        print(f"\n🔍 示例查询 {i}: {query}\n")

        try:
            response = query_engine.query(query)
            print("💡 回答:")
            print(response)

            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("\n📚 相关文档片段:")
                for j, node in enumerate(response.source_nodes, 1):
                    print(f"\n  片段 {j} (相似度: {node.score:.4f}):")
                    print(f"  {node.text[:200]}...")

            print("\n" + "-" * 60)

        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")


def interactive_query(query_engine):
    """
    交互式查询函数

    Args:
        query_engine: 查询引擎对象
    """
    print("\n" + "=" * 50)
    print("📖 PDF RAG 交互式查询系统")
    print("=" * 50)
    print("输入 'exit' 或 'quit' 退出\n")

    while True:
        try:
            query = input("\n🔍 请输入您的问题: ").strip()

            if query.lower() in ['exit', 'quit', '退出']:
                print("\n👋 再见！")
                break

            if not query:
                continue

            print("\n💭 思考中...\n")
            response = query_engine.query(query)

            print("💡 回答:")
            print(response)
            print("\n" + "-" * 50)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 查询出错: {str(e)}")


def main():
    """主函数"""
    print("=" * 60)
    print("PDF RAG 系统 - 使用 BGE-M3 Embedding 模型")
    print("=" * 60)
    print()

    try:
        # 1. 设置 RAG 系统
        print("步骤 1/5: 设置 RAG 系统配置")
        setup_rag_system()
        print()

        # 2. 加载文档
        print("步骤 2/5: 加载 PDF 文档")
        documents = load_documents("./data/")
        if not documents:
            print("\n⚠️  请将 PDF 文件放入 ./data/ 目录后重新运行")
            return
        print()

        # 3. 构建索引
        print("步骤 3/5: 构建向量索引")
        index = build_index(documents, show_estimation=True)
        print()

        # 4. 创建查询引擎
        print("步骤 4/5: 创建查询引擎")
        query_engine = create_query_engine(index, similarity_top_k=3)
        print()

        # 5. 运行示例查询
        print("步骤 5/5: 运行示例查询")
        run_example_queries(query_engine)

        # 6. 进入交互式查询
        print("\n" + "=" * 60)
        interactive_query(query_engine)

    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
