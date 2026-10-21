
# Import libraries
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import InMemorySaver
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# Load API keys
load_dotenv()

# Initialize Langfuse tracing
langfuse = get_client()
langfuse_handler = CallbackHandler()

# Define the response format
class Answer(BaseModel):
    answer: str
    confidence: float

# Initialize Tavily
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Create the web search tool
@tool
def tavily_web_search(query: str) -> dict:
    """Search the web for current information."""
    return tavily.search(query=query, search_depth="advanced")

# Define agent instructions
system_prompt = """You are a helpful research assistant.
Use Tavily to research factual questions.
Give a concise answer and a confidence score between 0 and 1.
"""

# Initialize NVIDIA model
model = ChatNVIDIA(
    model="nvidia/nemotron-3.5-super-vl-120b-a12b",
    nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.1,
    max_completion_tokens=2048,
)

# Set up short-term memory
memory = InMemorySaver()

# Configure conversation and tracing
config = {
    "configurable": {"thread_id": "1"},
    "callbacks": [langfuse_handler],
}

# Create the agent
agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=[tavily_web_search],
    checkpointer=memory,
    response_format=Answer,
)

# Ask the agent a question
try:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the capital of india?"}]},
        config=config,
    )
    print("Agent Response:", result["structured_response"])
finally:
    langfuse.flush()
