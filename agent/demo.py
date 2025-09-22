# 导入OpenAI
from langchain_openai import OpenAI
# 导入SerpAPIWrapper即工具包
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.agents import AgentExecutor
# 选择要使用的大模型
# 从LangChain Hub中获取ReAct的提示
# 导入create_react_agent功能
from langchain.agents import create_react_agent
prompt = hub.pull("hwchase17/react")
print(prompt)
llm = OpenAI()

# 实例化SerpAPIWrapper
search = SerpAPIWrapper()
# 准备工具列表
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="当大模型没有相关知识时，用于搜索知识"
    ),
]
# 构建ReAct Agent
agent = create_react_agent(llm, tools, prompt)
# 导入AgentExecutor
# 创建Agent执行器并传入Agent和工具
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# 调用AgentExecutor
agent_executor.invoke({"input": "当前Agent最新研究进展是什么？"})