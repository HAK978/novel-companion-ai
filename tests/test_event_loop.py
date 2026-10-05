"""Blocking calls must stay off the event loop.

An `async def` handler runs on the event loop, so a synchronous database, vector-search or
Celery call inside it stalls every other request on that service until it returns, health
checks included. Sync work belongs in a plain `def` handler (FastAPI threadpools those) or
behind run_in_threadpool.
"""

import ast
import asyncio
import threading
import time

import httpx
import pytest
from conftest import SERVICES

# Calls in this codebase that block: SQLAlchemy connections, Chroma-backed search, and
# Celery's enqueue (a synchronous round trip to Redis).
BLOCKING = {"connect", "search_chunks", "delay", "get_collection", "_get_collection"}


def blocking_calls_in_async_handlers(path):
    tree = ast.parse(path.read_text())
    found = []

    def visit(node, in_async):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AsyncFunctionDef):
                visit(child, True)
            elif isinstance(child, (ast.FunctionDef, ast.Lambda)):
                visit(child, False)  # a nested sync helper is fine: it runs in a thread
            else:
                if in_async and isinstance(child, ast.Call):
                    func = child.func
                    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                    if name in BLOCKING:
                        found.append((child.lineno, name))
                visit(child, in_async)

    visit(tree, False)
    return found


@pytest.mark.parametrize(
    "path", sorted(SERVICES.glob("*/main.py")), ids=lambda p: p.parent.name
)
def test_no_async_handler_makes_blocking_calls(path):
    assert blocking_calls_in_async_handlers(path) == []


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *args, **kwargs):
        return type("R", (), {"fetchall": lambda self: [], "fetchone": lambda self: None})()


def test_a_slow_search_does_not_stall_other_requests(retrieval_main, monkeypatch):
    entered = threading.Event()

    def slow_search(**kwargs):
        entered.set()
        time.sleep(1.0)
        return []

    monkeypatch.setattr(retrieval_main, "search_chunks", slow_search)
    monkeypatch.setattr(retrieval_main, "engine",
                        type("E", (), {"connect": lambda self: FakeConnection()})())

    async def scenario():
        transport = httpx.ASGITransport(app=retrieval_main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # The clock starts before either request runs. If the slow handler blocks the
            # event loop, everything after this point waits behind it, including the
            # moment the fast request can even be sent.
            start = time.perf_counter()
            slow = asyncio.create_task(client.post("/search", json={
                "query": "q", "current_chapter": 10, "collection_name": "novel_1"}))
            while not entered.is_set():  # the slow search is now inside its handler
                await asyncio.sleep(0.005)
            fast = await client.get("/characters", params={"novel_id": 1, "current_chapter": 10})
            fast_done = time.perf_counter() - start
            await slow
            return fast.status_code, fast_done

    status, fast_done = asyncio.run(scenario())
    assert status == 200
    assert fast_done < 0.5, f"fast request finished {fast_done:.2f}s in, behind the slow search"
