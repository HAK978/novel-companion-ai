import os

# Any OpenAI-compatible chat-completions endpoint: vLLM (the default, served locally),
# OpenAI, Groq, Together, Ollama. Switching providers is a matter of these settings.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8004/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "mistralai/Mistral-Nemo-Instruct-2407")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "180"))
