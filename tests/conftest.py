"""Shared fixtures and a loader for the service modules.

Each service is a self-contained directory that imports its own modules by flat
name (`from config import ...`), and several services define same-named modules
(`config.py`, `main.py`). Importing them normally would collide, so each load
puts that service's directory first on sys.path and evicts cached flat names
before executing the target file under a unique module name.
"""

import importlib.util
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVICES = ROOT / "services"

# Module names services import from their own directory, which must not leak
# between loads.
_FLAT_NAMES = ("config", "main", "search", "chunking", "tasks", "llm_handler", "vector_store")


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


@pytest.fixture(scope="session")
def retrieval_main():
    return load_module("retrieval", "main.py")


@pytest.fixture(scope="session")
def ingestion_tasks():
    return load_module("ingestion", "tasks.py")


def load_script(filename: str, alias: str):
    """Import a script from scripts/ under a collision-free name."""
    spec = importlib.util.spec_from_file_location(alias, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_migrate():
    """The migration runner in scripts/, so tests build schemas exactly the way deploys do."""
    return load_script("migrate.py", "novel_migrate")


class FakeEmbedding:
    """Deterministic stand-in for the embedding model, for tests where vector values do not
    matter. Real embeddings would download the ONNX model on first use."""

    def __call__(self, input):
        return [[float(len(d) % 97), float(sum(map(ord, d)) % 89), 1.0] for d in input]

    @staticmethod
    def name():
        return "fake_for_tests"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbedding()

    def is_legacy(self):
        return False


# --- real-database tier ------------------------------------------------------
#
# Stubs record SQL without running it, so they cannot check what PostgreSQL actually returns.
# Tests marked `db` run against a real, disposable database. Locally they use the Compose
# Postgres; without one they are skipped, unless REQUIRE_DB=1 (as in CI), where an
# unreachable database is a failure rather than a silent skip.

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://novel:novel@localhost:5432/novel_companion_test"
)


class Postgres:
    """Creates and drops throwaway databases on the test server."""

    def __init__(self, base_url: str):
        self.base = urlparse(base_url)

    def url(self, dbname: str) -> str:
        return self.base._replace(path=f"/{dbname}").geturl()

    def _admin(self):
        import psycopg2

        conn = psycopg2.connect(self.url("postgres"), connect_timeout=3)
        conn.autocommit = True
        return conn

    def recreate(self, dbname: str) -> str:
        # Never drop anything that is not unmistakably a test database
        if not dbname.endswith("_test"):
            raise RuntimeError(f"refusing to recreate {dbname!r}: name must end in _test")
        conn = self._admin()
        try:
            with conn.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{dbname}" WITH (FORCE)')
                cur.execute(f'CREATE DATABASE "{dbname}"')
        finally:
            conn.close()
        return self.url(dbname)

    def drop(self, dbname: str) -> None:
        if not dbname.endswith("_test"):
            raise RuntimeError(f"refusing to drop {dbname!r}: name must end in _test")
        conn = self._admin()
        try:
            with conn.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{dbname}" WITH (FORCE)')
        finally:
            conn.close()


@pytest.fixture(scope="session")
def postgres():
    import psycopg2

    server = Postgres(TEST_DATABASE_URL)
    try:
        server._admin().close()
    except psycopg2.OperationalError as exc:
        if os.environ.get("REQUIRE_DB") == "1":
            pytest.fail(f"REQUIRE_DB=1 but no Postgres is reachable: {exc}")
        pytest.skip("no Postgres reachable for the db test tier")
    return server


@pytest.fixture(scope="session")
def pg_url(postgres):
    """A fresh test database with every migration applied by the real runner."""
    dbname = urlparse(TEST_DATABASE_URL).path.lstrip("/")
    url = postgres.recreate(dbname)
    load_migrate().migrate(url, out=lambda _: None)
    return url


@pytest.fixture
def db_engine(pg_url):
    """SQLAlchemy engine on the test database, emptied before each test."""
    from sqlalchemy import create_engine, text

    engine = create_engine(pg_url)
    with engine.begin() as conn:
        tables = conn.execute(text(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename <> 'schema_migrations'"
        )).scalars().all()
        conn.execute(text(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE"))
    yield engine
    engine.dispose()


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
                # a route may supply a FakeResponse directly to set a status code
                return response if isinstance(response, FakeResponse) else FakeResponse(response)
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
