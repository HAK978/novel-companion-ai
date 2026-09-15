import os

import httpx
from config import OPENAI_API_KEY

# Model config
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-Nemo-Instruct-2407")
HF_GPU = int(os.environ.get("HF_GPU", "0"))  # Which GPU to use (transformers fallback)
# vLLM OpenAI-compatible server (scripts/start_services.sh launches it on GPU 1)
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8004/v1")

class LLMError(RuntimeError):
    """Generation failed.

    Raised instead of returning an error string: callers were storing
    "Error: ..." text as if it were a real answer, and caching it.
    """


MAX_TOKENS = 800  # hard cap above the prompt's soft target so answers don't truncate
TEMPERATURE = 0.3


class LLMHandler:
    def __init__(self):
        self.backend = "none"
        self.model = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._detect_backend()

    def _detect_backend(self):
        # Prefer vLLM: paged KV-cache, continuous batching, CUDA graphs.
        # Probing it first also means we skip loading a second 24GB copy of
        # the model via transformers.
        try:
            resp = httpx.get(f"{VLLM_URL}/models", timeout=5)
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            if models:
                self.backend = "vllm"
                self.model = models[0]
                print(f"Using vLLM backend at {VLLM_URL}: {self.model}")
                return
        except Exception as e:
            print(f"vLLM not reachable ({e}), falling back to transformers")

        # Fall back to loading via transformers in-process
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading {HF_MODEL} on GPU {HF_GPU}...")
            self._hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL,
                dtype=torch.float16,
                device_map={"": f"cuda:{HF_GPU}"},
                trust_remote_code=True,
            )
            self._hf_model.eval()
            self.backend = "transformers"
            self.model = HF_MODEL
            print(f"Model loaded: {HF_MODEL}")
            return
        except Exception as e:
            print(f"Failed to load HF model: {e}")

        # Fall back to OpenAI
        if OPENAI_API_KEY:
            self.backend = "openai"
            self.model = "gpt-4o-mini"
            return

        self.backend = "none"

    def generate(self, query: str, context_chunks: list[str],
                 conversation_context: str = "") -> str | None:
        if self.backend == "none":
            return None

        prompt = self._build_prompt(query, context_chunks, conversation_context)

        if self.backend == "vllm":
            return self._vllm_generate(prompt)
        elif self.backend == "transformers":
            return self._transformers_generate(prompt)
        elif self.backend == "openai":
            return self._openai_generate(prompt)
        return None

    def _build_prompt(self, query: str, context_chunks: list[str],
                      conversation_context: str = "") -> str:
        parts = []

        if conversation_context:
            parts.append(conversation_context)

        context = "\n\n".join(
            f"Passage {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)
        )

        parts.append(f"""Based on the following passages, answer this question: {query}

Context:
{context}

Instructions:
- Match the depth of the answer to the question: typically 100-400 words
- If this is a follow-up question, use the previous conversation context
- Passages are labeled with their chapter number; pay attention to it
- Passages from far-apart chapters describe different points in a long story; do NOT merge them into one narrative. If the question concerns recent or specific events, prefer the passages whose chapters match that timeframe and say which chapters your answer draws from
- Include specific details, character names, and events
- Do not pad the answer with filler or a summary conclusion

Answer:""")

        return "\n".join(parts)

    def _vllm_generate(self, prompt: str) -> str:
        try:
            resp = httpx.post(
                f"{VLLM_URL}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise LLMError(f"vLLM generation failed: {e}") from e

    def _transformers_generate(self, prompt: str) -> str:
        try:
            import torch
            messages = [{"role": "user", "content": prompt}]
            text = self._hf_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._hf_tokenizer(text, return_tensors="pt").to(self._hf_model.device)

            with torch.no_grad():
                outputs = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self._hf_tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            return self._hf_tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as e:
            raise LLMError(f"transformers generation failed: {e}") from e

    def _openai_generate(self, prompt: str) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable assistant. Provide detailed answers based on the given context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI generation failed: {e}") from e

    def get_available_models(self) -> dict:
        return {
            "backend": self.backend,
            "current_model": self.model,
        }
