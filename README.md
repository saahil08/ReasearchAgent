## Environment Setup

1. Clone this repository.
2. Install [uv](https://docs.astral.sh/uv/).
3. Install the project dependencies:

4. Copy `.env.example` to `.env` in the project root.
5. Replace the placeholder values with your own API keys.

### Required API Keys (you can use other LLM API key also)

| Variable | Purpose | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | Gemini language model | [Google AI Studio](https://aistudio.google.com/apikey) |
| `TAVILY_API_KEY` | Web search | [Tavily](https://tavily.com/) |
| `LANGFUSE_SECRET_KEY` | Langfuse tracing | [Langfuse](https://cloud.langfuse.com/) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse tracing | [Langfuse](https://cloud.langfuse.com/) |
| `LANGFUSE_BASE_URL` | Langfuse server endpoint | `https://cloud.langfuse.com` |
