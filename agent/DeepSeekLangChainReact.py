from langchain.agents import AgentType, initialize_agent,AgentExecutor,load_tools,create_react_agent
from langchain.prompts import PromptTemplate


from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI  # 可以使用 LangChain 的 OpenAI 兼容接口
from dotenv import load_dotenv
import os
# 设置 DeepSeek API（如果提供 API，可类似 OpenAI 使用）
load_dotenv()
deep_seek_key=os.environ["DEEPSEEK_API_KEY"]
serper_api_key=os.environ["SERPAPI_API_KEY"]
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=deep_seek_key,
    model="deepseek-chat"
)

# 设置promt

template = (
    '尽你所能回答以下问题。如果能力不够，你可以使用以下工具:\n\n'
    '{tools}\n\n'
    'Use the following format:\n\n'
    'Question: the input question you must answer\n'
    'Thought: you should always think about what to do\n'
    'Action: the action to take, should be one of [{tool_names}]\n'
    'Action Input: the input to the action\n'
    'Observation: the result of the action\n'
    '... (this Thought/Action/Action Input/Observation can repeat N times)\n'
    'Thought: I now know the final answer\n'
    'Final Answer: the final answer to the'
    'original input question\n\n'
    'Begin!\n\n'
    'Question: {input}\n'
    'Thought:{agent_scratchpad}'
)
prompt = PromptTemplate.from_template(template)


# 定义工具（例如网络搜索）
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 创建 Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input":
                       """目前市场上玫瑰花的一般进货价格是多少？\n
                       如果我在此基础上加价5%，应该如何定价？"""})

