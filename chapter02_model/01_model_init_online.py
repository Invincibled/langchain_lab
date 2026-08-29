from langchain_community.chat_models import ChatZhipuAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)

ZHIPU_API_KEY= os.getenv("ZHIPUAI_API_KEY")
ZHIPU_BASE_URL = os.getenv("ZHIPUAI_BASE_URL")

zhipuai_chat= ChatZhipuAI(
    model="glm-5.2",
    api_key=ZHIPU_API_KEY,
    api_base=ZHIPU_BASE_URL
)
messages = [
            ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
            ("human", "我喜欢编程。"),
        ]

resp = zhipuai_chat.invoke(messages)
print(resp.text)