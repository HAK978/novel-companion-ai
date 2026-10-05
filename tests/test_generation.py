"""The generation service and its single LLM client, with the model endpoint stubbed.

This service had no tests before: importing it probed the model endpoint and could load a
24 GB model. Now construction does no I/O, so it can be imported and exercised directly.
"""

from pathlib import Path

import httpx
import pytest
from conftest import SERVICES, load_module
from fastapi.testclient import TestClient

MODEL = "mistralai/Mistral-Nemo-Instruct-2407"


def completion(text):
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture
def endpoint(monkeypatch):
    """Stub the model endpoint; record what was sent to it."""
    sent = []
    replies = {"post": completion("An answer."), "status": 200, "models": [MODEL]}

    def post(url, json=None, headers=None, timeout=None):
        sent.append({"url": url, "json": json, "headers": headers or {}})
        if isinstance(replies["post"], Exception):
            raise replies["post"]
        return httpx.Response(replies["status"], json=replies["post"], request=httpx.Request("POST", url))

    def get(url, headers=None, timeout=None):
        if isinstance(replies["models"], Exception):
            raise replies["models"]
        return httpx.Response(200, json={"data": [{"id": m} for m in replies["models"]]},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", get)
    return sent, replies


@pytest.fixture
def service(endpoint):
    module = load_module("generation", "main.py", alias="generation_main")
    return TestClient(module.app)


def test_importing_the_service_does_no_io(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network call at import time")

    monkeypatch.setattr(httpx, "post", refuse)
    monkeypatch.setattr(httpx, "get", refuse)

    load_module("generation", "main.py", alias="generation_main_import_check")


def test_answers_come_back_with_the_model_name(service, endpoint):
    endpoint[1]["post"] = completion("  Elena drew her sword.  ")

    body = service.post("/generate", json={"query": "q", "context_chunks": ["c"]}).json()

    assert body == {"answer": "Elena drew her sword.", "model_used": MODEL}


def test_the_prompt_carries_chapter_labelled_passages(service, endpoint):
    sent, _ = endpoint
    service.post("/generate", json={
        "query": "What did Elena do?",
        "context_chunks": ["[Chapter 12: The Gate]\nElena drew her sword."],
    })

    prompt = sent[0]["json"]["messages"][-1]["content"]
    assert "[Chapter 12: The Gate]" in prompt
    assert "What did Elena do?" in prompt
    assert sent[0]["json"]["model"] == MODEL


def test_an_unreachable_model_is_a_502_not_an_answer(service, endpoint):
    endpoint[1]["post"] = httpx.ConnectError("connection refused")

    response = service.post("/generate", json={"query": "q", "context_chunks": ["c"]})

    assert response.status_code == 502
    assert "answer" not in response.json()


def test_a_provider_error_is_a_502(service, endpoint):
    endpoint[1]["status"] = 500

    assert service.post("/generate", json={"query": "q", "context_chunks": []}).status_code == 502


def test_an_empty_completion_is_a_502(service, endpoint):
    endpoint[1]["post"] = completion("   ")

    assert service.post("/generate", json={"query": "q", "context_chunks": []}).status_code == 502


def test_extraction_prompt_is_sent_as_written(service, endpoint):
    # It used to go through the Q&A template, which appended "write 100-400 words and cite
    # chapters" after "return ONLY a JSON array"
    sent, replies = endpoint
    replies["post"] = completion("[]")

    service.post("/extract-entities", json={"chapter_text": "Elena drew her sword.", "chapter_number": 3})

    prompt = sent[0]["json"]["messages"][0]["content"]
    assert prompt.startswith("You are a precise entity extractor")
    assert "Based on the following passages" not in prompt
    assert "100-400 words" not in prompt


def test_extraction_reads_json_inside_a_code_fence(service, endpoint):
    endpoint[1]["post"] = completion('```json\n[{"name": "Elena", "aliases": []}]\n```')

    body = service.post("/extract-entities", json={"chapter_text": "t", "chapter_number": 1}).json()

    assert body["characters"] == [{"name": "Elena", "aliases": []}]


def test_api_key_is_sent_as_a_bearer_token(endpoint):
    client_module = load_module("generation", "llm_client.py", alias="generation_llm_client")
    client = client_module.LLMClient(base_url="https://api.example.com/v1", model="m", api_key="secret")

    client.complete("hi")

    assert endpoint[0][0]["headers"] == {"Authorization": "Bearer secret"}
    assert endpoint[0][0]["url"] == "https://api.example.com/v1/chat/completions"


def test_health_is_ok_when_the_model_is_served(service):
    assert service.get("/health").json()["status"] == "ok"


def test_health_reports_an_unreachable_endpoint(service, endpoint):
    # it used to report "ok" unconditionally
    endpoint[1]["models"] = httpx.ConnectError("connection refused")

    body = service.get("/health").json()

    assert body["status"] == "degraded"
    assert "unreachable" in body["error"]


def test_health_reports_a_model_the_endpoint_does_not_serve(service, endpoint):
    endpoint[1]["models"] = ["some-other-model"]

    assert service.get("/health").json()["status"] == "degraded"


def test_no_in_process_model_loading_remains():
    # The fallback loaded a second 24 GB copy onto a shared GPU when vLLM was down
    source = "\n".join(p.read_text() for p in (SERVICES / "generation").glob("*.py"))
    for marker in ("import torch", "transformers", "AutoModelForCausalLM", "cuda"):
        assert marker not in source, marker


def test_requirements_carry_no_gpu_stack():
    reqs = Path(SERVICES / "generation" / "requirements.txt").read_text().lower()
    assert "torch" not in reqs and "transformers" not in reqs


def test_prompt_forbids_outside_knowledge_and_knows_the_reading_position(service, endpoint):
    # Asked at chapter 5 who "turns out to be the villain", the model named a famous book's
    # culprit from memory; retrieval scoping cannot stop that, the prompt has to
    sent, _ = endpoint
    service.post("/generate", json={"query": "q", "context_chunks": ["c"], "current_chapter": 5})

    prompt = sent[0]["json"]["messages"][-1]["content"]
    assert "read up to chapter 5" in prompt
    assert "after chapter 5" in prompt
    assert "Use ONLY the passages" in prompt
    assert "not revealed that yet" in prompt


def test_prompt_without_a_position_still_forbids_outside_knowledge(service, endpoint):
    sent, _ = endpoint
    service.post("/generate", json={"query": "q", "context_chunks": ["c"]})

    prompt = sent[0]["json"]["messages"][-1]["content"]
    assert "Use ONLY the passages" in prompt
    assert "read up to chapter" not in prompt


def test_answers_carry_the_spoiler_rules_as_a_system_message(service, endpoint):
    # with the rules only in the user prompt, a recognized book's ending leaked in 4/10
    # tries; adding them as the system message brought that to 0/10
    sent, _ = endpoint
    service.post("/generate", json={"query": "q", "context_chunks": ["c"], "current_chapter": 5})

    system, user = sent[0]["json"]["messages"]
    assert system["role"] == "system" and user["role"] == "user"
    assert "read up to chapter 5" in system["content"]
    assert "never use that knowledge" in system["content"]


def test_extraction_needs_no_reader_position(service, endpoint):
    sent, replies = endpoint
    replies["post"] = completion("[]")

    service.post("/extract-entities", json={"chapter_text": "t", "chapter_number": 3})

    assert [m["role"] for m in sent[0]["json"]["messages"]] == ["user"]


def test_health_probe_of_the_model_is_short(monkeypatch):
    # /health backs the container health check (4 s); a hanging model endpoint must not
    # make the generation container look dead
    seen = {}

    def get(url, headers=None, timeout=None):
        seen["timeout"] = timeout
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(httpx, "get", get)
    client_module = load_module("generation", "llm_client.py", alias="generation_llm_client_probe")

    assert client_module.LLMClient().status()["status"] == "degraded"
    assert seen["timeout"] <= 2
