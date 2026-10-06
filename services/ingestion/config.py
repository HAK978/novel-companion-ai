import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CHROMADB_URL = os.environ.get("CHROMADB_URL", "http://localhost:8005")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://novel:novel@localhost:5432/novel_companion"
)
GENERATION_SERVICE_URL = os.environ.get("GENERATION_SERVICE_URL", "http://localhost:8003")
# Passages the model answering a question reads, in words
CHUNK_SIZE = 400
# The embedding model reads only the first 256 tokens of a text, about half of a chunk, so
# chunks are searched through smaller windows: 170 words always fit (measured on the Hound
# and Shadow Slave 1-300). Windows are 2.7x as many vectors as chunks.
WINDOW_SIZE = 170

# Chapter summaries are written in the background, only as far as readers have got plus this
# many chapters: work past every reader's position is wasted until someone reaches it.
SUMMARY_LOOKAHEAD = int(os.environ.get("SUMMARY_LOOKAHEAD", "100"))
SUMMARY_BATCH = 50  # chapters per task
# model calls in flight per task; the server batches them. 8 took ~2.4 s per chapter on
# one A6000, 32 about 0.6 s; more slows any question asked meanwhile
SUMMARY_PARALLEL = int(os.environ.get("SUMMARY_PARALLEL", "16"))
