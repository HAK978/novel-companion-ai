
from config import DATABASE_URL
from fastapi import FastAPI
from pydantic import BaseModel
from search import delete_collection, search_chunks
from sqlalchemy import create_engine
from sqlalchemy import text as sa_text

app = FastAPI(title="Retrieval Service")
engine = create_engine(DATABASE_URL)


class SearchRequest(BaseModel):
    query: str
    current_chapter: int
    n_results: int = 5
    collection_name: str = "shadow_slave"
    min_chapter: int | None = None


class ChunkResult(BaseModel):
    text: str
    chapter_number: int
    chapter_title: str = ""
    relevance_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    count: int


class DeleteCollectionRequest(BaseModel):
    novel_id: int


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    results = search_chunks(
        query=request.query,
        current_chapter=request.current_chapter,
        n_results=request.n_results,
        collection_name=request.collection_name,
        min_chapter=request.min_chapter,
    )
    return SearchResponse(
        query=request.query,
        results=results,
        count=len(results),
    )


@app.get("/characters/{name}")
async def character_recall(name: str, novel_id: int, current_chapter: int = 9999):
    """Look up a character by name, scoped to reading progress."""
    with engine.connect() as conn:
        # Find character by name or alias
        row = conn.execute(
            sa_text("""
                SELECT id, name, aliases, first_appearance, description
                FROM characters
                WHERE novel_id = :nid
                  AND (LOWER(name) = LOWER(:name) OR LOWER(:name) = ANY(SELECT LOWER(unnest(aliases))))
                  AND first_appearance <= :ch
                LIMIT 1
            """),
            {"nid": novel_id, "name": name, "ch": current_chapter},
        ).fetchone()

        if not row:
            return {"error": f"Character '{name}' not found (up to chapter {current_chapter})"}

        char_id = row[0]

        # Get mentions up to current chapter
        mentions = conn.execute(
            sa_text("""
                SELECT chapter_number, context
                FROM character_mentions
                WHERE character_id = :cid AND chapter_number <= :ch
                ORDER BY chapter_number
            """),
            {"cid": char_id, "ch": current_chapter},
        ).fetchall()

        # Get relationships up to current chapter
        relationships = conn.execute(
            sa_text("""
                SELECT c2.name, cr.relationship_type, cr.first_chapter
                FROM character_relationships cr
                JOIN characters c2 ON cr.character_b_id = c2.id
                WHERE cr.character_a_id = :cid
                  AND cr.first_chapter <= :ch
                ORDER BY cr.first_chapter
            """),
            {"cid": char_id, "ch": current_chapter},
        ).fetchall()

    # Also search ChromaDB for passages mentioning this character
    collection_name = f"novel_{novel_id}"
    passages = search_chunks(
        query=name,
        current_chapter=current_chapter,
        n_results=3,
        collection_name=collection_name,
    )

    return {
        "name": row[1],
        "aliases": row[2] or [],
        "first_appearance": row[3],
        "description": row[4],
        "mentions": [
            {"chapter": m[0], "context": m[1]} for m in mentions
        ],
        "relationships": [
            {"character": r[0], "type": r[1], "since_chapter": r[2]} for r in relationships
        ],
        "relevant_passages": passages,
    }


@app.get("/characters")
async def list_characters(novel_id: int, current_chapter: int = 9999):
    """List all characters discovered up to current reading position."""
    with engine.connect() as conn:
        rows = conn.execute(
            sa_text("""
                SELECT name, aliases, first_appearance, description
                FROM characters
                WHERE novel_id = :nid AND first_appearance <= :ch
                ORDER BY first_appearance
            """),
            {"nid": novel_id, "ch": current_chapter},
        ).fetchall()

    return [
        {
            "name": r[0],
            "aliases": r[1] or [],
            "first_appearance": r[2],
            "description": r[3],
        }
        for r in rows
    ]


@app.post("/delete-collection")
async def delete_col(request: DeleteCollectionRequest):
    name = f"novel_{request.novel_id}"
    success = delete_collection(name)
    return {"collection": name, "deleted": success}


@app.get("/health")
async def health():
    try:
        from search import _get_client
        client = _get_client()
        counts = {c.name: client.get_collection(c.name).count()
                  for c in client.list_collections()}
        return {"status": "ok", "chunks_indexed": sum(counts.values()), "collections": counts}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
