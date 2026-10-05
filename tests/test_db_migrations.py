"""Migrations against a real PostgreSQL server.

The static schema test checks that upserts have matching constraints on paper; these check
that the migrations actually run, including on databases shaped by this project's history.
"""

from pathlib import Path

import pytest
from conftest import load_migrate
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.db

MIGRATIONS = sorted((Path(__file__).resolve().parent.parent / "migrations").glob("*.sql"))
NOVEL_SCOPED_TABLES = (
    "chapters", "characters", "character_mentions", "character_relationships",
    "chapter_summaries", "range_summaries", "reading_progress", "search_history",
)


@pytest.fixture
def scratch(postgres):
    """An empty, migration-free database for replaying a particular history."""
    name = "novel_companion_scratch_test"
    url = postgres.recreate(name)
    engine = create_engine(url)
    yield url, engine
    engine.dispose()
    postgres.drop(name)


def run_sql(engine, sql: str) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def test_fresh_database_records_every_migration(pg_url):
    engine = create_engine(pg_url)
    with engine.connect() as conn:
        recorded = conn.execute(text("SELECT filename FROM schema_migrations")).scalars().all()
    engine.dispose()

    assert sorted(recorded) == [m.name for m in MIGRATIONS]


def test_every_novel_scoped_table_requires_a_novel(db_engine):
    with db_engine.connect() as conn:
        nullable = conn.execute(text(
            "SELECT table_name FROM information_schema.columns "
            "WHERE column_name = 'novel_id' AND is_nullable = 'YES'"
        )).scalars().all()

    assert nullable == []
    assert set(NOVEL_SCOPED_TABLES) <= {
        row for row in db_engine.connect().execute(text(
            "SELECT table_name FROM information_schema.columns WHERE column_name = 'novel_id'"
        )).scalars()
    }


def test_runner_is_a_noop_when_up_to_date(pg_url):
    assert load_migrate().migrate(pg_url, out=lambda _: None) == []


def test_runner_refuses_a_schema_it_has_no_record_of(scratch):
    # A database built by the old docker-entrypoint mount has tables but no record;
    # replaying 001 onward against it would fail partway.
    url, engine = scratch
    run_sql(engine, MIGRATIONS[0].read_text())
    migrate = load_migrate()

    with pytest.raises(migrate.MigrationError, match="--baseline"):
        migrate.migrate(url, out=lambda _: None)


def test_003_applies_to_the_hand_patched_dev_database(scratch):
    # The development database was built from 001 and 002, then patched by hand: the
    # name-only unique constraint dropped and a unique index created under the very name
    # 003 later used for its constraint. 003 as first written failed on this.
    url, engine = scratch
    run_sql(engine, MIGRATIONS[0].read_text())
    run_sql(engine, MIGRATIONS[1].read_text())
    run_sql(engine, """
        ALTER TABLE characters DROP CONSTRAINT IF EXISTS characters_name_key;
        CREATE UNIQUE INDEX characters_novel_name_unique ON characters (novel_id, name);
    """)
    migrate = load_migrate()
    assert migrate.migrate(url, baseline=MIGRATIONS[1].name, out=lambda _: None) == []

    applied = migrate.migrate(url, out=lambda _: None)

    assert applied == [m.name for m in MIGRATIONS[2:]]
    with engine.connect() as conn:
        constraint = conn.execute(text(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conname = 'characters_novel_name_unique'"
        )).scalar()
    assert constraint == "UNIQUE (novel_id, name)"


def test_004_repairs_what_it_can_and_removes_true_orphans(scratch):
    url, engine = scratch
    for migration in MIGRATIONS[:3]:
        run_sql(engine, migration.read_text())
    run_sql(engine, """
        INSERT INTO novels (id, title) VALUES (1, 'Kept');
        INSERT INTO chapters (novel_id, chapter_number, title) VALUES (NULL, 1, 'orphan');
        INSERT INTO chapters (novel_id, chapter_number, title) VALUES (1, 1, 'kept');
        INSERT INTO characters (id, novel_id, name, first_appearance) VALUES (10, 1, 'Elena', 5);
        -- written by old ingestion code that never set novel_id on mentions
        INSERT INTO character_mentions (character_id, chapter_number, context, novel_id)
            VALUES (10, 5, 'appears', NULL);
    """)
    migrate = load_migrate()
    migrate.migrate(url, baseline=MIGRATIONS[2].name, out=lambda _: None)
    migrate.migrate(url, out=lambda _: None)

    with engine.connect() as conn:
        chapters = conn.execute(text("SELECT title FROM chapters")).scalars().all()
        mention_novel = conn.execute(text("SELECT novel_id FROM character_mentions")).scalar()

    assert chapters == ["kept"]
    assert mention_novel == 1  # inferred from its character, not deleted


def test_baseline_records_history_without_changing_the_schema(scratch):
    url, engine = scratch
    run_sql(engine, MIGRATIONS[0].read_text())
    run_sql(engine, MIGRATIONS[1].read_text())

    load_migrate().migrate(url, baseline=MIGRATIONS[1].name, out=lambda _: None)

    with engine.connect() as conn:
        recorded = conn.execute(text("SELECT filename FROM schema_migrations")).scalars().all()
        aliases_table = conn.execute(text("SELECT to_regclass('character_aliases')")).scalar()
    assert sorted(recorded) == [m.name for m in MIGRATIONS[:2]]
    assert aliases_table is None  # 004 not applied


def test_unused_conversations_table_is_gone(db_engine):
    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('conversations')")).scalar() is None


def test_006_keeps_cached_range_summaries_and_frees_the_name(scratch):
    # 006 renames the range cache so that 007 can create chapter_summaries for single
    # chapters; the rename must carry the table's constraints, index and sequence names
    url, engine = scratch
    for migration in MIGRATIONS[:5]:
        run_sql(engine, migration.read_text())
    run_sql(engine, """
        INSERT INTO novels (id, title) VALUES (1, 'Kept');
        INSERT INTO chapter_summaries (novel_id, start_chapter, end_chapter, summary)
            VALUES (1, 1, 10, 'a cached recap');
    """)
    migrate = load_migrate()
    migrate.migrate(url, baseline=MIGRATIONS[4].name, out=lambda _: None)

    applied = migrate.migrate(url, out=lambda _: None)

    with engine.connect() as conn:
        recap = conn.execute(text("SELECT summary FROM range_summaries")).scalar()
        leftover = conn.execute(text(
            "SELECT conname FROM pg_constraint WHERE conrelid = 'range_summaries'::regclass "
            "AND conname NOT LIKE 'range_summaries%'")).scalars().all()
        per_chapter = conn.execute(text("SELECT count(*) FROM chapter_summaries")).scalar()
    assert applied == [m.name for m in MIGRATIONS[5:]]
    assert recap == "a cached recap"
    assert leftover == []
    assert per_chapter == 0
