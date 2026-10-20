
import os
from dotenv import load_dotenv
from tavily import TavilyClient
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langfuse.langchain import CallbackHandler

# 1. Load environment variables
load_dotenv()

# 2. Initialize Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
)

# 3. Initialize Tavily
tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# 4. Define web search tool
@tool
def web_search(query: str) -> str:
    """Search the web for current or factual information."""
    results = tavily.search(query=query, max_results=3)
    return str(results)

# 5. Define structured output
class Answer(BaseModel):
    answer: str = Field(
        description="The final answer to the user's question."
    )
    confidence: float = Field(
        description="Estimated confidence from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )

# 6. Create the agent
agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=(
        "You are a helpful research assistant. "
        "Use web search when needed. "
        "Return a clear answer and a confidence score "
        "between 0.0 and 1.0. Base confidence on the "
        "reliability of the available evidence."
    ),
    response_format=ToolStrategy(Answer),
)

# 7. Run the agent
if __name__ == "__main__":
    question = input("Ask your question: ")

    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        },
        config={"callbacks": [CallbackHandler()]},
    )

    # 8. Extract structured output
    result = response["structured_response"]

    print("\n--- Agent Response ---")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")