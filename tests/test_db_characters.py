"""The character graph's spoiler rules, against real PostgreSQL.

Story used throughout (invented, not from any real novel): Elena first appears in chapter 5.
A masked figure, "the Grey Wanderer", is revealed as Elena in chapter 1500.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

pytestmark = pytest.mark.db


@pytest.fixture
def store(ingestion_tasks, db_engine, monkeypatch):
    """The real ingestion write path, pointed at the test database."""
    monkeypatch.setattr(ingestion_tasks, "engine", db_engine)
    with db_engine.begin() as conn:
        conn.execute(text("INSERT INTO novels (id, title) VALUES (1, 'Main'), (2, 'Other')"))

    def _store(chapter, characters, novel_id=1):
        ingestion_tasks._store_entities(novel_id, chapter, characters)

    return _store


@pytest.fixture
def api(retrieval_main, db_engine, monkeypatch):
    """The real retrieval endpoints on the test database; vector search stubbed out."""
    monkeypatch.setattr(retrieval_main, "engine", db_engine)
    monkeypatch.setattr(retrieval_main, "search_chunks", lambda **kwargs: [])
    client = TestClient(retrieval_main.app)

    def recall(name, at, novel_id=1):
        return client.get(
            f"/characters/{name}", params={"novel_id": novel_id, "current_chapter": at}
        ).json()

    def roster(at, novel_id=1):
        return client.get("/characters", params={"novel_id": novel_id, "current_chapter": at}).json()

    return type("Api", (), {"recall": staticmethod(recall), "roster": staticmethod(roster),
                            "client": client})


def elena(aliases=(), role=None, relationships=()):
    return {"name": "Elena", "aliases": list(aliases), "role": role,
            "relationships": list(relationships)}


@pytest.fixture
def revealed_at_1500(store):
    store(5, [elena(role="a travelling healer")])
    store(1500, [elena(aliases=["Grey Wanderer"], role="unmasked")])


def test_alias_is_hidden_before_its_reveal(api, revealed_at_1500):
    assert api.recall("Elena", at=300)["aliases"] == []


def test_alias_cannot_find_the_character_before_its_reveal(api, revealed_at_1500):
    # Matching here would tell a chapter-300 reader who the Grey Wanderer is
    assert "error" in api.recall("Grey Wanderer", at=300)


def test_alias_is_visible_and_searchable_after_its_reveal(api, revealed_at_1500):
    assert api.recall("Elena", at=1600)["aliases"] == ["Grey Wanderer"]
    assert api.recall("grey wanderer", at=1600)["name"] == "Elena"


def test_roster_shows_only_revealed_aliases(api, revealed_at_1500):
    assert api.roster(at=300) == [
        {"name": "Elena", "aliases": [], "first_appearance": 5,
         "description": "a travelling healer"}
    ]
    assert api.roster(at=1600)[0]["aliases"] == ["Grey Wanderer"]


def test_out_of_order_extraction_keeps_the_earliest_alias_reveal(api, store):
    # Parallel workers can finish chapter 1500 before chapter 900
    store(1500, [elena(aliases=["Grey Wanderer"])])
    store(900, [elena(aliases=["Grey Wanderer"])])

    assert api.recall("Elena", at=950)["aliases"] == ["Grey Wanderer"]


def test_out_of_order_extraction_keeps_the_earliest_appearance(api, store):
    store(1500, [elena()])
    store(5, [elena()])

    assert api.recall("Elena", at=10)["first_appearance"] == 5


def test_description_comes_from_the_earliest_chapter(api, store):
    # A description written from chapter 1500 can carry chapter 1500's reveal
    store(1500, [elena(role="unmasked as the Grey Wanderer")])
    store(5, [elena(role="a travelling healer")])

    assert api.recall("Elena", at=10)["description"] == "a travelling healer"


def test_relationship_mention_cannot_surface_a_later_description(api, store):
    # The Queen is extracted at chapter 800 with a revealing description; chapter 100,
    # processed later, mentions her only as Elena's relative. A chapter-150 reader must not
    # see what chapter 800 said about her.
    store(800, [{"name": "The Queen", "role": "secretly the villain", "aliases": []}])
    store(100, [elena(relationships=[{"character": "The Queen", "type": "family"}])])

    queen = api.recall("The Queen", at=150)
    assert queen["first_appearance"] == 100
    assert queen["description"] is None


def test_relationships_appear_from_their_earliest_chapter(api, store):
    store(1200, [elena(relationships=[{"character": "Marcus", "type": "ally"}])])
    store(40, [elena(relationships=[{"character": "Marcus", "type": "ally"}])])

    assert api.recall("Elena", at=50)["relationships"] == [
        {"character": "Marcus", "type": "ally", "since_chapter": 40}
    ]


def test_aliases_are_stored_once(store, db_engine):
    store(1500, [elena(aliases=["Grey Wanderer", "Grey Wanderer", " Grey Wanderer "])])
    store(1501, [elena(aliases=["Grey Wanderer"])])

    with db_engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM character_aliases")).scalar()
    assert count == 1


def test_a_name_never_becomes_its_own_alias(store, db_engine):
    store(5, [elena(aliases=["Elena", "elena"])])

    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM character_aliases")).scalar() == 0


def test_aliases_do_not_cross_novels(api, store):
    store(1, [elena(aliases=["Grey Wanderer"])], novel_id=2)
    store(5, [elena()], novel_id=1)

    assert "error" in api.recall("Grey Wanderer", at=300, novel_id=1)


def test_an_exact_name_beats_an_alias(api, store):
    store(5, [elena(aliases=["Wanderer"])])
    store(7, [{"name": "Wanderer", "role": "a different person", "aliases": []}])

    assert api.recall("Wanderer", at=10)["description"] == "a different person"


@pytest.mark.parametrize("path", ["/characters/Elena", "/characters"])
def test_reading_position_is_required(api, path):
    # Defaulting to chapter 9999 made a forgotten parameter show everything
    response = api.client.get(path, params={"novel_id": 1})

    assert response.status_code == 422
