import openai
import json
from typing import List, Dict, Any
import os
# 设置 DeepSeek API（如果提供 API，可类似 OpenAI 使用）
from dotenv import load_dotenv
load_dotenv()

deep_seek_key=os.environ["DEEPSEEK_API_KEY"]

# 设置 DeepSeek API
client = openai.OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=deep_seek_key,  # 替换为你的API密钥
)

def get_current_weather(location: str, unit: str = "celsius") -> str:
    """获取当前天气信息"""
    # 这里是模拟实现，实际可以接入天气API
    weather_data = {
        "beijing": {"temperature": 15, "unit": unit, "condition": "晴"},
        "shanghai": {"temperature": 18, "unit": unit, "condition": "多云"},
        "guangzhou": {"temperature": 22, "unit": unit, "condition": "雨"},
        "shenzhen": {"temperature": 25, "unit": unit, "condition": "晴"},
    }

    location_key = location.lower()
    if location_key in weather_data:
        data = weather_data[location_key]
        return f"{location}的天气: {data['condition']}, 温度: {data['temperature']}°{unit.upper()}"
    else:
        return f"找不到 {location} 的天气信息"

def get_stock_price(symbol: str) -> str:
    """获取股票价格"""
    # 模拟股票数据
    stock_data = {
        "AAPL": 185.25,
        "GOOGL": 145.80,
        "TSLA": 245.60,
        "MSFT": 405.30,
        "BABA": 85.45
    }

    if symbol.upper() in stock_data:
        return f"{symbol} 当前价格: ${stock_data[symbol.upper()]}"
    else:
        return f"找不到 {symbol} 的股票信息"

def calculate_math_expression(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全评估数学表达式
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "错误: 表达式包含非法字符"

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 定义可用的函数
functions = [
    {
        "name": "get_current_weather",
        "description": "获取指定城市的当前天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，如北京、上海"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位，摄氏度或华氏度"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_stock_price",
        "description": "获取股票的当前价格",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码，如AAPL、GOOGL"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "calculate_math_expression",
        "description": "计算数学表达式",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如'2+2'、'3*5'"
                }
            },
            "required": ["expression"]
        }
    }
]

def run_function_call(messages: List[Dict[str, Any]]) -> str:
    """执行函数调用流程"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=[{"type": "function", "function": func} for func in functions],
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"🔧 调用函数: {function_name}")
            print(f"📋 参数: {function_args}")

            # 调用对应的函数
            if function_name == "get_current_weather":
                function_response = get_current_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit", "celsius")
                )
            elif function_name == "get_stock_price":
                function_response = get_stock_price(
                    symbol=function_args.get("symbol")
                )
            elif function_name == "calculate_math_expression":
                function_response = calculate_math_expression(
                    expression=function_args.get("expression")
                )
            else:
                function_response = f"错误: 未知函数 {function_name}"

            print(f"✅ 函数返回: {function_response}")

            # 添加函数响应到消息历史
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

        # 获取最终回复
        second_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )

        return second_response.choices[0].message.content
    else:
        return response_message.content

# 测试函数调用
def test_function_calling():
    """测试函数调用功能"""
    test_cases = [
        "今天北京的天气怎么样？",
        "苹果公司的股票价格是多少？",
        "计算一下(1250+378)*2.5等于多少",
        "今天深圳的天气，用华氏度显示",
        "谷歌的股票价格",
        "2的8次方乘以3.14是多少"
    ]

    for query in test_cases:
        print(f"\n{'='*60}")
        print(f"👤 用户: {query}")

        messages = [{"role": "user", "content": query}]
        response = run_function_call(messages)

        print(f"🤖 AI: {response}")

if __name__ == "__main__":

    test_function_calling()