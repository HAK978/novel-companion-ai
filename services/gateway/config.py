import os

INGESTION_SERVICE_URL = os.environ.get("INGESTION_SERVICE_URL", "http://localhost:8001")
RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://localhost:8002")
GENERATION_SERVICE_URL = os.environ.get("GENERATION_SERVICE_URL", "http://localhost:8003")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://novel:novel@localhost:5432/novel_companion"
)
