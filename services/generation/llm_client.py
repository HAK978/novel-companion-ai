"""Model access for the generation service.

One client for any OpenAI-compatible chat-completions endpoint. There used to be three
paths: vLLM over HTTP, the OpenAI SDK (the same protocol again), and a fallback that loaded
a second 24 GB copy of the model in-process whenever vLLM was unreachable, silently taking
a GPU on a shared machine. An unreachable endpoint is now reported as unavailable instead.

Construction does no I/O. The old handler probed the endpoint (and could load a model) at
import time, which is why this service had no tests.
"""

import httpx
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT

MAX_TOKENS = 800  # hard cap above the prompt's soft target so answers don't truncate
TEMPERATURE = 0.3


class LLMError(RuntimeError):
    """Generation failed.

    Raised rather than returned as text: callers once stored "Error: ..." strings as real
    answers and cached them.
    """


def build_prompt(query: str, context_chunks: list[str], conversation_context: str = "",
                 current_chapter: int | None = None) -> str:
    parts = []
    if conversation_context:
        parts.append(conversation_context)

    context = "\n\n".join(f"Passage {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks))
    spoiler_rules = _spoiler_rules(current_chapter)
    parts.append(f"""Based on the following passages, answer this question: {query}

Context:
{context}

Instructions:
{spoiler_rules}
- Match the depth of the answer to the question: typically 100-400 words
- If this is a follow-up question, use the previous conversation context
- Passages are labeled with their chapter number; pay attention to it
- Passages from far-apart chapters describe different points in a long story; do NOT merge them into one narrative. If the question concerns recent or specific events, prefer the passages whose chapters match that timeframe and say which chapters your answer draws from
- Include specific details, character names, and events
- Do not pad the answer with filler or a summary conclusion

Answer:""")
    return "\n".join(parts)


def system_prompt(current_chapter: int | None) -> str:
    """The companion's standing rules, sent as the system message.

    Measured on a well-known book (question at chapter 5: who "turns out to be the villain
    at the end"): with the rules only inside the user prompt, the model named the culprit
    from memory in 4 of 10 tries; with them also as the system message, 0 of 10.
    """
    if current_chapter is None:
        position, limit = "", "never reveal anything the passages do not contain"
    else:
        position = (f"The reader has read up to chapter {current_chapter} of this book and "
                    f"nothing beyond it. ")
        limit = f"never reveal, hint at, or confirm anything after chapter {current_chapter}"
    return (
        "You are a spoiler-free reading companion. " + position
        + "Answer only from the passages you are given. You may recognize this book: never use "
        f"that knowledge, and {limit}. If the passages do not answer a question, say the story "
        "has not revealed that yet."
    )


def _spoiler_rules(current_chapter: int | None) -> str:
    # Retrieval only hands the model chapters the reader has read, but a model may already
    # know a well-known book: asked at chapter 5 who "turns out to be the villain", it named
    # the culprit from memory in 5 of 5 tries. So the prompt has to rule that out too.
    rules = []
    if current_chapter is not None:
        rules.append(
            f"- The reader has read up to chapter {current_chapter}. Never reveal, hint at, or "
            f"confirm anything that happens after chapter {current_chapter}."
        )
    rules.append(
        "- Use ONLY the passages above. Do not use anything you may know about this book from "
        "anywhere else, even if you recognize it."
    )
    rules.append(
        "- If the passages do not answer the question, say the story has not revealed that "
        "yet. Do not guess or speculate about what happens later."
    )
    return "\n".join(rules)


class LLMClient:
    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        api_key: str = LLM_API_KEY,
        timeout: float = LLM_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = MAX_TOKENS,
                 temperature: float = TEMPERATURE) -> str:
        """Send a prompt exactly as given (plus an optional system message); return the text."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = httpx.post(f"{self.base_url}/chat/completions", json=payload,
                              headers=self._headers, timeout=self._timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"{self.base_url}: {exc}") from exc
        if not content or not content.strip():
            raise LLMError(f"{self.base_url}: empty completion")
        return content.strip()

    def answer(self, query: str, context_chunks: list[str], conversation_context: str = "",
               current_chapter: int | None = None) -> str:
        """Answer a question from retrieved passages, for a reader at `current_chapter`."""
        return self.complete(
            build_prompt(query, context_chunks, conversation_context, current_chapter),
            system=system_prompt(current_chapter),
        )

    def status(self) -> dict:
        """Whether the endpoint is reachable and serving the configured model. Never raises."""
        try:
            # short: this backs /health, which container probes give 4 s
            resp = httpx.get(f"{self.base_url}/models", headers=self._headers, timeout=2)
            resp.raise_for_status()
            served = [m.get("id") for m in resp.json().get("data", [])]
        except (httpx.HTTPError, ValueError) as exc:
            return {"status": "degraded", "endpoint": self.base_url, "model": self.model,
                    "error": f"endpoint unreachable: {exc}"}
        if served and self.model not in served:
            return {"status": "degraded", "endpoint": self.base_url, "model": self.model,
                    "error": f"model not served; endpoint has {served}"}
        return {"status": "ok", "endpoint": self.base_url, "model": self.model}
