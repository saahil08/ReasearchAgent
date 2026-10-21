
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler

# Load .env from the project root
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env", override=True)

# Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your project-root .env file."
    )

# Initialize Langfuse
langfuse_handler = CallbackHandler()

# Initialize Gemini, not OpenAI
model = init_chat_model(
    model="gemini-3.1-flash-lite",
    model_provider="google_genai",
    api_key=api_key,
    temperature=0.1,
    max_tokens=50,
)

response = model.invoke(
    "You have beautiful eyes!",
    config={"callbacks": [langfuse_handler]},
)

print("Response from Gemini:", response.content)
#uv run python Models/Models.py