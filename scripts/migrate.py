"""Apply pending SQL migrations in order, recording each one in `schema_migrations`.

Without a record of what has been applied, a migration can be committed and never reach the
database that matters; 003 sat unapplied on the development database for three weeks.

Usage:
    python scripts/migrate.py                 # apply pending migrations
    python scripts/migrate.py --status        # list applied and pending migrations
    python scripts/migrate.py --baseline 002_multi_novel.sql
        # for a database built before this runner existed: record every migration up to
        # and including the named file as applied, without running them. Applies nothing;
        # run again without --baseline to apply the rest.

The target database comes from DATABASE_URL.
"""

import argparse
import os
from collections.abc import Callable
from pathlib import Path

import psycopg2

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
DEFAULT_URL = "postgresql://novel:novel@localhost:5432/novel_companion"

# Any of these existing without a tracking record means the schema was built some other way
# (by hand, or by an older docker-entrypoint mount), so blindly replaying 001 onward is unsafe.
_KNOWN_TABLES = ("novels", "chapters")


class MigrationError(RuntimeError):
    pass


def migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def _applied(cur) -> set[str]:
    cur.execute("SELECT filename FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def migrate(
    database_url: str,
    baseline: str | None = None,
    status_only: bool = False,
    out: Callable[[str], None] = print,
) -> list[str]:
    """Apply pending migrations; return the names of those applied."""
    files = migration_files()
    names = [f.name for f in files]
    conn = psycopg2.connect(database_url)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.schema_migrations') IS NOT NULL"
            )
            tracker_existed = cur.fetchone()[0]
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            if baseline is not None:
                if baseline not in names:
                    raise MigrationError(f"unknown migration {baseline!r}; have {names}")
                for name in names[: names.index(baseline) + 1]:
                    cur.execute(
                        "INSERT INTO schema_migrations (filename) VALUES (%s) "
                        "ON CONFLICT (filename) DO NOTHING",
                        (name,),
                    )
                # Bookkeeping only: recording history must not also change the schema.
                # Run again without --baseline to apply what is still pending.
                out(f"baselined through {baseline}; run again to apply pending migrations")
                return []
            elif not tracker_existed:
                cur.execute(
                    "SELECT count(*) FROM unnest(%s::text[]) AS t(name) "
                    "WHERE to_regclass('public.' || t.name) IS NOT NULL",
                    (list(_KNOWN_TABLES),),
                )
                if cur.fetchone()[0]:
                    raise MigrationError(
                        "this database has tables but no migration record. Re-running from "
                        "001 is unsafe; record what it already has with --baseline <file>."
                    )

            done = _applied(cur)

        pending = [f for f in files if f.name not in done]

        if status_only:
            for name in names:
                out(f"  {'applied' if name in done else 'pending'}  {name}")
            return []

        applied_now = []
        for path in pending:
            # one transaction per file: PostgreSQL DDL is transactional, so a failing
            # migration leaves no partial schema behind
            with conn, conn.cursor() as cur:
                cur.execute(path.read_text())
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,)
                )
            applied_now.append(path.name)
            out(f"applied {path.name}")

        if not applied_now:
            out("up to date")
        return applied_now
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", action="store_true", help="list migrations and exit")
    parser.add_argument("--baseline", metavar="FILE", help="mark migrations through FILE as applied")
    args = parser.parse_args()

    url = os.environ.get("DATABASE_URL", DEFAULT_URL)
    try:
        migrate(url, baseline=args.baseline, status_only=args.status)
    except MigrationError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
