from config import DATABASE_URL
from fastapi import FastAPI
from names import withhold
from pydantic import BaseModel, Field
from search import delete_collection, search_chunks
from sqlalchemy import create_engine
from sqlalchemy import text as sa_text
from words import CAPITALIZED, WORD, stem

app = FastAPI(title="Retrieval Service")
engine = create_engine(DATABASE_URL)

# Handlers are plain `def`: Chroma and SQLAlchemy calls here are synchronous, and FastAPI
# runs sync handlers in a threadpool. Declared `async def`, each one blocked the event loop
# for the whole call, so one slow search stalled every other request, health checks included.


class SearchRequest(BaseModel):
    query: str
    current_chapter: int
    collection_name: str
    n_results: int = Field(5, ge=1, le=20)
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
def search(request: SearchRequest):
    results = search_chunks(
        query=request.query,
        current_chapter=request.current_chapter,
        n_results=request.n_results,
        collection_name=request.collection_name,
        min_chapter=request.min_chapter,
    )
    return SearchResponse(query=request.query, results=results, count=len(results))


# `current_chapter` is required on every character endpoint. It used to default to 9999,
# so a caller that forgot it got every character, alias and relationship: the spoiler
# filter failed open.


@app.get("/characters/{name}")
def character_recall(name: str, novel_id: int, current_chapter: int):
    """Look up a character by name or alias, as known at the reader's current chapter.

    An alias counts only from the chapter that revealed it: matching on, or returning, a
    later alias would confirm an identity the reader has not reached yet.
    """
    with engine.connect() as conn:
        row = conn.execute(
            sa_text("""
                SELECT c.id, c.name, c.first_appearance, c.description
                FROM characters c
                WHERE c.novel_id = :nid
                  AND c.first_appearance <= :ch
                  AND (
                      LOWER(c.name) = LOWER(:name)
                      OR EXISTS (
                          SELECT 1 FROM character_aliases a
                          WHERE a.character_id = c.id
                            AND LOWER(a.alias) = LOWER(:name)
                            AND a.first_chapter <= :ch
                      )
                  )
                -- an exact name beats an alias match
                ORDER BY LOWER(c.name) = LOWER(:name) DESC, c.first_appearance, c.id
                LIMIT 1
            """),
            {"nid": novel_id, "name": name, "ch": current_chapter},
        ).fetchone()

        if not row:
            return {"error": f"Character '{name}' not found (up to chapter {current_chapter})"}

        char_id = row[0]

        aliases = conn.execute(
            sa_text("""
                SELECT alias FROM character_aliases
                WHERE character_id = :cid AND first_chapter <= :ch
                ORDER BY first_chapter, alias
            """),
            {"cid": char_id, "ch": current_chapter},
        ).scalars().all()

        mentions = conn.execute(
            sa_text("""
                SELECT chapter_number, context
                FROM character_mentions
                WHERE character_id = :cid AND chapter_number <= :ch
                ORDER BY chapter_number
            """),
            {"cid": char_id, "ch": current_chapter},
        ).fetchall()

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

    passages = search_chunks(
        query=name,
        current_chapter=current_chapter,
        n_results=3,
        collection_name=f"novel_{novel_id}",
    )

    return {
        "name": row[1],
        "aliases": list(aliases),
        "first_appearance": row[2],
        "description": row[3],
        "mentions": [{"chapter": m[0], "context": m[1]} for m in mentions],
        "relationships": [
            {"character": r[0], "type": r[1], "since_chapter": r[2]} for r in relationships
        ],
        "relevant_passages": passages,
    }


@app.get("/characters")
def list_characters(novel_id: int, current_chapter: int):
    """List the characters the reader has met, with only the aliases revealed so far."""
    with engine.connect() as conn:
        rows = conn.execute(
            sa_text("""
                SELECT c.name, c.first_appearance, c.description,
                       COALESCE(
                           array_agg(a.alias ORDER BY a.first_chapter, a.alias)
                               FILTER (WHERE a.alias IS NOT NULL),
                           '{}'::text[]
                       ) AS aliases
                FROM characters c
                LEFT JOIN character_aliases a
                       ON a.character_id = c.id AND a.first_chapter <= :ch
                WHERE c.novel_id = :nid AND c.first_appearance <= :ch
                GROUP BY c.id
                ORDER BY c.first_appearance, c.name
            """),
            {"nid": novel_id, "ch": current_chapter},
        ).fetchall()

    return [
        {
            "name": r[0],
            "aliases": list(r[3]),
            "first_appearance": r[1],
            "description": r[2],
        }
        for r in rows
    ]


class WithholdRequest(BaseModel):
    novel_id: int
    current_chapter: int
    text: str
    # what the reader wrote (their question): repeating their own words back is no leak
    ignore: str = ""


@app.post("/withhold-unread-names")
def withhold_unread_names(request: WithholdRequest):
    """`text` without the names the reader has not reached: words the book only ever writes
    capitalized, first used after `current_chapter`. A model that knows the book uses them
    even when every passage it was given comes from chapters the reader has read: at chapter
    14 of the Hound it called Stapleton's wife "Beryl Garcia", a name the book first uses in
    chapter 15. `withheld` counts the names taken out (never listed: they are the spoiler)."""
    ignore = {stem(w) for w in WORD.findall(request.ignore)}
    capitalized = {stem(w) for w in CAPITALIZED.findall(request.text)}
    names: dict[str, int] = {}  # the capitalized words that are names: first chapter
    if capitalized:
        with engine.connect() as conn:
            names = dict(conn.execute(
                sa_text("""
                    SELECT word, first_chapter FROM book_words
                    WHERE novel_id = :nid AND word = ANY(:words)
                      AND first_lowercase_chapter IS NULL
                """),
                {"nid": request.novel_id, "words": sorted(capitalized)},
            ).all())
    unread = {w for w, first in names.items()
              if first > request.current_chapter and w not in ignore}
    if not unread:
        return {"text": request.text, "withheld": 0}
    return {"text": withhold(request.text, unread, set(names)), "withheld": len(unread)}


@app.post("/delete-collection")
def delete_col(request: DeleteCollectionRequest):
    name = f"novel_{request.novel_id}"
    success = delete_collection(name)
    return {"collection": name, "deleted": success}


@app.get("/health")
def health():
    try:
        from search import _get_client
        client = _get_client()
        counts = {c.name: client.get_collection(c.name).count()
                  for c in client.list_collections()}
        return {"status": "ok", "chunks_indexed": sum(counts.values()), "collections": counts}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
