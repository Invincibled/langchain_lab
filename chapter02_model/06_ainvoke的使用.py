"""
06 ainvoke 的使用 —— 体现"不阻塞"功能

ainvoke 是 LangChain 提供的异步调用方法，基于 asyncio 实现。
与 invoke（同步阻塞）不同，ainvoke 返回一个可等待的协程（coroutine），
配合 asyncio.gather 等工具可以在等待一个请求的同时发起/处理其他请求，
从而实现并发执行、互不阻塞。

本例使用智谱 GLM 演示：
  1. 基础用法：单个 ainvoke 调用
  2. 并发用法：asyncio.gather 同时发起多个 ainvoke，
     总耗时远小于逐个串行 invoke，体现"不阻塞"特性
"""

import asyncio
import os
import time

from langchain_community.chat_models import ChatZhipuAI
from dotenv import load_dotenv

load_dotenv(override=True)

ZHIPU_API_KEY = os.getenv("ZHIPUAI_API_KEY")
ZHIPU_BASE_URL = os.getenv("ZHIPUAI_BASE_URL")

zhipuai_chat = ChatZhipuAI(
    model="glm-5.2",
    api_key=ZHIPU_API_KEY,
    api_base=ZHIPU_BASE_URL,
)

# 翻译助手：将中文翻译为英文
messages_translate = [
    ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
    ("human", "我喜欢编程。"),
]

# 总结助手：做文本总结
messages_summary = [
    ("system", "你是一名专业的编辑，擅长对长文本进行简洁总结。"),
    ("human", "请用一句话总结：人工智能正在改变医疗、教育、交通等多个行业。"),
]

# 提问助手：回答常识问题
messages_qa = [
    ("system", "你是知识渊博的助手。"),
    ("human", "Python 和 Java 的主要区别是什么？"),
]


async def example_basic() -> None:
    """基础用法：单个 ainvoke 调用，await 等待结果。"""
    print("===== 示例1：单个 ainvoke 调用 =====")
    resp = await zhipuai_chat.ainvoke(messages_translate)
    print(f"模型返回：{resp.text}")
    print()


async def concurrent_task(name: str, messages: list) -> str:
    """异步任务：调用 ainvoke 并返回带标签的结果。

    这是演示"不阻塞"的核心——每个协程在 await ainvoke 时会
    "让出"控制权给事件循环，从而允许其他协程同时运行。
    """
    print(f"[{name}] 开始请求 ...")
    start = time.perf_counter()
    resp = await zhipuai_chat.ainvoke(messages)
    cost = time.perf_counter() - start
    print(f"[{name}] 完成，耗时 {cost:.2f}s，结果：{resp.text}")
    return resp.text


async def example_concurrent() -> None:
    """并发用法：用 asyncio.gather 同时发起多个 ainvoke。

    注意观察耗时：3 个请求并发执行的总耗时约等于最慢那一个的耗时，
    而不是 3 倍串行耗时——这正是"不阻塞"的意义所在。
    """
    print("===== 示例2：ainvoke 并发执行（不阻塞） =====")
    start = time.perf_counter()

    # 同时发起 3 个异步请求，事件循环会交错调度，互不阻塞
    results = await asyncio.gather(
        concurrent_task("翻译", messages_translate),
        concurrent_task("总结", messages_summary),
        concurrent_task("问答", messages_qa),
    )

    total = time.perf_counter() - start
    print(f"3 个并发请求总耗时：{total:.2f}s")
    print(f"返回结果数量：{len(results)} 条")
    print()


async def main() -> None:
    await example_basic()
    await example_concurrent()


if __name__ == "__main__":
    asyncio.run(main())
