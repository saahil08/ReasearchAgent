
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

model = ChatNVIDIA(
    model="nvidia/nemotron-3.5-super-vl-120b-a12b",
    nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
    max_completion_tokens=128,
)

response = model.invoke("Reply with only: Hello!")
print(response.content)
