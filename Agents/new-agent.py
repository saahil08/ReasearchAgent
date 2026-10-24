# we need in environment 
# we need langfuse
# we need langchain Agent
# we need to create tools
# Tavily Tool call.
# we need dynamodb connector with langchain WE CAN  also use posgres
# Middleware \ HITL \ gaurdrails
# Multi Agent creation \ MCP


import os
import boto3
from dotenv import load_dotenv

load_dotenv()

from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph_checkpoint_aws import DynamoDBSaver
from tavily import TavilyClient
from langchain_nvidia_ai_endpoints import ChatNVIDIA
table_name = "checkPointers"
region_name = "ap-south-1"
langfuse_handler = CallbackHandler()

tavilyclient=TavilyClient()

checkpointer=DynamoDBSaver(
table_name="checkPointers",
region_name="ap-southeast-2",
)

model = ChatNVIDIA(
    model="nvidia/nemotron-3.5-super-vl-120b-a12b",
    nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
    max_completion_tokens=1024,
)

# Research Agent
system_prompt_research_agent = "Research the user's topic and provide a concise, factual answer."
research_agent = create_agent(
    model=model,
    system_prompt=system_prompt_research_agent,
)

# Calculator Agent
system_prompt_calculator_agent = "Calculate the result of the user's mathematical expression."
calculator_agent = create_agent(
    model=model,
    system_prompt=system_prompt_calculator_agent,
)


@tool
def web_search(query: str) -> str:
    """Search the web for a given query."""
    return str(tavilyclient.search(query))


@tool
def research(query: str) -> str:
    """Research the given topic."""
    response = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return response["messages"][-1].content


@tool
def calculator(expression: str) -> str:
    """Calculate the given expression."""
    response = calculator_agent.invoke({
        "messages": [{"role": "user", "content": expression}]
    })
    return response["messages"][-1].content



def main():
    config = {
        "configurable": {"thread_id": "check-01"},
        "callbacks": [langfuse_handler],
    }

    dynamodb = boto3.client("dynamodb", region_name="ap-southeast-2")
    print(f"List of dynamo Tables {dynamodb}")

    agent = create_agent(
        model=model,
        system_prompt="you are helfull assistant you will help users with there queries",
        checkpointer=checkpointer,
        tools=[web_search],
    )

    agent_response = agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": "My Name is saahil i create Ai Agents",
            }]
        },
        config=config,
    )

    agent_response2 = agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": "what is my name and what i do?",
            }]
        },
        config=config,
    )

    print(f"agent response 2 {agent_response2['messages'][-1].content}")


if __name__ == "__main__":
    main()


# uv run python Agents/new-agent.py