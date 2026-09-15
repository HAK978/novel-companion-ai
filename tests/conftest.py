"""Shared fixtures and a loader for the service modules.

Each service is a self-contained directory that imports its own modules by flat
name (`from config import ...`), and several services define same-named modules
(`config.py`, `main.py`). Importing them normally would collide, so each load
puts that service's directory first on sys.path and evicts cached flat names
before executing the target file under a unique module name.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVICES = ROOT / "services"

# Module names services import from their own directory, which must not leak
# between loads.
_FLAT_NAMES = ("config", "main", "search", "chunking", "tasks", "llm_handler")


def load_module(service: str, filename: str, alias: str | None = None):
    """Import `services/<service>/<filename>` under a collision-free name."""
    service_dir = SERVICES / service
    path = service_dir / filename
    name = alias or f"{service}_{path.stem}"

    for flat in _FLAT_NAMES:
        sys.modules.pop(flat, None)

    sys.path.insert(0, str(service_dir))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(service_dir))


@pytest.fixture(scope="session")
def chunking():
    return load_module("ingestion", "chunking.py")


@pytest.fixture(scope="session")
def mcp_server():
    return load_module("mcp", "server.py", alias="novel_mcp_server")


@pytest.fixture(scope="session")
def retrieval_search():
    return load_module("retrieval", "search.py")


@pytest.fixture(scope="session")
def gateway_main():
    return load_module("gateway", "main.py")


class FakeResponse:
    """Stands in for an httpx.Response."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeAsyncClient:
    """Async httpx client stub that serves canned responses by URL substring.

    Records every call so tests can assert on what a service sent downstream.
    """

    def __init__(self, routes, calls):
        self._routes = routes
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def _resolve(self, method, url, payload):
        self.calls.append({"method": method, "url": url, "payload": payload})
        for fragment, response in self._routes.items():
            if fragment in url:
                return FakeResponse(response)
        raise AssertionError(f"unexpected {method} to {url}")

    async def post(self, url, json=None, **kwargs):
        return self._resolve("POST", url, json)

    async def get(self, url, params=None, **kwargs):
        return self._resolve("GET", url, params)


@pytest.fixture
def fake_async_client():
    """Build a patchable AsyncClient factory plus the list of recorded calls."""

    def _build(routes):
        calls = []

        def factory(*args, **kwargs):
            return FakeAsyncClient(routes, calls)

        return factory, calls

    return _build
