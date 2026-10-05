import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CHROMADB_URL = os.environ.get("CHROMADB_URL", "http://localhost:8005")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://novel:novel@localhost:5432/novel_companion"
)
GENERATION_SERVICE_URL = os.environ.get("GENERATION_SERVICE_URL", "http://localhost:8003")
CHUNK_SIZE = 400
