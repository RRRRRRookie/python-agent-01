from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI  # 可以使用 LangChain 的 OpenAI 兼容接口
from dotenv import load_dotenv
import os
# 设置 DeepSeek API（如果提供 API，可类似 OpenAI 使用）
load_dotenv()
deep_seek_key=os.environ["DEEPSEEK_API_KEY"]
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=deep_seek_key,
    model="deepseek-chat"
)

# 定义工具（例如网络搜索）
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Web Search",
        func=search.run,
        description="Useful for searching the internet"
    )
]

# 创建 Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 使用 ReAct 模式
    verbose=True
)

agent.run("DeepSeek 最新动态是什么？")