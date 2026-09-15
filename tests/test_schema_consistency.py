"""Every ON CONFLICT target must have a matching UNIQUE constraint.

Database stubs never execute SQL, so they cannot catch a mismatch between what
the code upserts on and what the migrations actually create. That gap let
`ON CONFLICT (novel_id, name)` ship against a `characters` table that was still
unique by name alone: the insert worked on the hand-patched development
database and failed on any fresh one.

This replays the migrations' constraint changes in order and checks the result
against the upserts in the service code.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = sorted((ROOT / "migrations").glob("*.sql"))
SERVICES = ROOT / "services"


def _columns(text: str) -> tuple[str, ...]:
    return tuple(c.strip().lower() for c in text.split(",") if c.strip())


def _implicit_name(table: str, columns: tuple[str, ...]) -> str:
    """PostgreSQL's generated name for an unnamed UNIQUE constraint."""
    return f"{table}_{'_'.join(columns)}_key"


def unique_constraints() -> dict[str, set[tuple[str, ...]]]:
    """Replay migrations and return {table: {(column, ...), ...}}."""
    constraints: dict[str, set[tuple[str, ...]]] = {}
    named: dict[str, tuple[str, tuple[str, ...]]] = {}

    for migration in MIGRATIONS:
        sql = migration.read_text()

        for table, body in re.findall(
            r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+(\w+)\s*\((.*?)\n\);", sql, re.S | re.I
        ):
            table = table.lower()
            found = constraints.setdefault(table, set())
            for cols in re.findall(r"\bUNIQUE\s*\(([^)]+)\)", body, re.I):
                found.add(_columns(cols))
            for line in body.splitlines():
                stripped = line.strip().rstrip(",")
                if "(" in stripped:
                    continue
                if re.match(r"^\w+\s+[\w\[\]]+.*\bUNIQUE\b", stripped, re.I):
                    found.add((stripped.split()[0].lower(),))

        # Applied in document order: a migration commonly drops a constraint
        # and re-adds it under the same name, so ordering decides the outcome.
        for alter in re.finditer(
            r"ALTER TABLE\s+(\w+)\s+(?:"
            r"ADD CONSTRAINT\s+(\w+)\s+UNIQUE\s*\(([^)]+)\)"
            r"|DROP CONSTRAINT(?:\s+IF EXISTS)?\s+(\w+))",
            sql,
            re.I,
        ):
            table = alter.group(1).lower()

            if alter.group(2):
                name, columns = alter.group(2).lower(), _columns(alter.group(3))
                constraints.setdefault(table, set()).add(columns)
                named[name] = (table, columns)
                continue

            name = alter.group(4).lower()
            explicit = named.pop(name, None)
            if explicit:
                constraints.get(explicit[0], set()).discard(explicit[1])
                continue
            for columns in list(constraints.get(table, set())):
                if _implicit_name(table, columns) == name:
                    constraints[table].discard(columns)

    return constraints


def upserts() -> list[tuple[Path, str, tuple[str, ...]]]:
    """Find (file, table, conflict columns) for every ON CONFLICT in the code."""
    found = []
    for path in sorted(SERVICES.rglob("*.py")):
        text = path.read_text()
        inserts = list(re.finditer(r"INSERT INTO\s+(\w+)", text, re.I))
        for i, match in enumerate(inserts):
            # only look as far as the next INSERT, so a statement without an
            # ON CONFLICT cannot borrow the next statement's
            end = inserts[i + 1].start() if i + 1 < len(inserts) else len(text)
            conflict = re.search(r"ON CONFLICT\s*\(([^)]+)\)", text[match.end():end], re.I)
            if conflict:
                found.append((path, match.group(1).lower(), _columns(conflict.group(1))))
    return found


def test_migrations_define_constraints():
    assert unique_constraints(), "no UNIQUE constraints parsed from migrations"


def test_code_contains_upserts():
    assert upserts(), "no ON CONFLICT statements found to verify"


def test_superseded_constraints_are_dropped():
    # 001 made characters unique by name alone, which would reject the same
    # character appearing in two novels.
    assert ("name",) not in unique_constraints().get("characters", set())


@pytest.mark.parametrize(
    "path,table,columns",
    [pytest.param(p, t, c, id=f"{t}[{'+'.join(c)}]") for p, t, c in upserts()],
)
def test_every_upsert_has_a_matching_constraint(path, table, columns):
    available = unique_constraints().get(table, set())
    wanted = set(columns)

    assert any(set(c) == wanted for c in available), (
        f"{path.relative_to(ROOT)} upserts into {table} "
        f"ON CONFLICT ({', '.join(columns)}), but the migrations define "
        f"{sorted(sorted(c) for c in available) or 'no unique constraint'} there. "
        f"This fails on a fresh database."
    )
