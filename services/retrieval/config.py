import os

CHROMADB_URL = os.environ.get("CHROMADB_URL", "http://localhost:8005")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://novel:novel@localhost:5432/novel_companion"
)
COLLECTION_NAME = "shadow_slave"
