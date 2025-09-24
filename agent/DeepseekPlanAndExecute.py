from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI  # 可以使用 LangChain 的 OpenAI 兼容接口
from langchain_experimental.plan_and_execute import PlanAndExecute, load_chat_planner, load_agent_executor
import os

load_dotenv()


@tool
def check_inventory(flower_type: str) -> int:
    """
    :param flower_type:
    :return:
    """
    return 100


@tool
def calculate_price(base_price: float, mark_up: float) -> float:
    """
    :param base_price:
    :param mark_up:
    :return:
    """
    return base_price * (1 + mark_up)


@tool
def schedule_delivery(order_id: int, delivery_date: str):
    """
    :param order_id:
    :param delivery_date:
    :return:
    """

    return f"订单{order_id} 已经在{delivery_date}安排配送"


tools = [check_inventory, calculate_price, schedule_delivery]
deep_seek_key = os.environ["DEEPSEEK_API_KEY"]
llm = ChatOpenAI(
    temperature=0,
    base_url="https://api.deepseek.com/v1",
    api_key=deep_seek_key,
    model="deepseek-chat"
)

planer = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)
agent = PlanAndExecute(planner=planer, executor=executor, verbose=True)
# agent.run("查查玫瑰的库存然后给出出货方案")
agent.run("查查玫瑰的库存然后给出50朵玫瑰花的价格和当天的配送方案")
