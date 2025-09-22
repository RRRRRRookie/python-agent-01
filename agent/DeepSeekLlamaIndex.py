from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# 设置 DeepSeek API
from dotenv import load_dotenv
load_dotenv()

# 配置 LLM 和 Embedding
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.environ["DEEPSEEK_API_KEY"])
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def basic_qa_demo():
    """基础问答demo"""
    print("=== DeepSeek + LlamaIndex 基础问答 Demo ===")

    # 创建一些示例文档内容
    sample_docs = """
    DeepSeek 是一家领先的人工智能公司，专注于大语言模型研发。
    DeepSeek-V3 是公司最新发布的大语言模型，支持128K上下文长度。
    公司成立于2023年，总部位于中国。
    DeepSeek 模型在代码生成、自然语言理解方面表现优异。
    """

    # 将文本内容写入临时文件
    with open("sample_doc.txt", "w", encoding="utf-8") as f:
        f.write(sample_docs)

    # 加载文档
    documents = SimpleDirectoryReader(input_files=["sample_doc.txt"]).load_data()

    # 创建向量索引
    index = VectorStoreIndex.from_documents(documents)

    # 创建查询引擎
    query_engine = index.as_query_engine()

    # 进行查询
    questions = [
        "DeepSeek 是什么公司？",
        "DeepSeek-V3 支持多长的上下文？",
        "DeepSeek 在哪些方面表现优异？"
    ]

    for question in questions:
        print(f"\n🤔 问题: {question}")
        response = query_engine.query(question)
        print(f"🤖 回答: {response}")
        print("-" * 50)


if __name__ == "__main__":
    basic_qa_demo()