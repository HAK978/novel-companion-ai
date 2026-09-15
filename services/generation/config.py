import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_OLLAMA_MODEL = "llama3.1:latest"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
