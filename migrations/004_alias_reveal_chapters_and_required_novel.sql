-- Per-alias reveal chapters, and every row must belong to a novel.

-- 1. Aliases
--
-- Aliases were a single array per character, appended from every chapter with no record of
-- when each was revealed, and returned unfiltered. A reader at chapter 100 could see, and
-- search by, a title first revealed at chapter 1500. Each alias now records the chapter it
-- first appeared in, and reads filter on it.

CREATE TABLE IF NOT EXISTS character_aliases (
    id SERIAL PRIMARY KEY,
    character_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    first_chapter INTEGER NOT NULL,
    CONSTRAINT character_aliases_character_alias_unique UNIQUE (character_id, alias)
);

CREATE INDEX IF NOT EXISTS character_aliases_lower_alias_idx
    ON character_aliases (LOWER(alias));

-- Existing array aliases are dropped rather than carried over: the chapter that revealed
-- each one was never recorded, so any date assigned now would be a guess, and a wrong guess
-- is a spoiler. Re-running extraction repopulates them with real reveal chapters.
ALTER TABLE characters DROP COLUMN IF EXISTS aliases;

-- 2. Rows without a novel
--
-- These predate multi-novel support. Every read path filters by novel_id, so they are
-- unreachable, and deleting a novel can never remove them. Rows whose novel can be inferred
-- from their character are repaired first; the rest are removed.

UPDATE character_mentions m SET novel_id = c.novel_id
FROM characters c
WHERE m.character_id = c.id AND m.novel_id IS NULL AND c.novel_id IS NOT NULL;

UPDATE character_relationships r SET novel_id = c.novel_id
FROM characters c
WHERE r.character_a_id = c.id AND r.novel_id IS NULL AND c.novel_id IS NOT NULL;

DELETE FROM character_mentions
WHERE novel_id IS NULL
   OR character_id IN (SELECT id FROM characters WHERE novel_id IS NULL);

DELETE FROM character_relationships
WHERE novel_id IS NULL
   OR character_a_id IN (SELECT id FROM characters WHERE novel_id IS NULL)
   OR character_b_id IN (SELECT id FROM characters WHERE novel_id IS NULL);

DELETE FROM characters WHERE novel_id IS NULL;
DELETE FROM chapters WHERE novel_id IS NULL;
DELETE FROM chapter_summaries WHERE novel_id IS NULL;
DELETE FROM reading_progress WHERE novel_id IS NULL;
DELETE FROM search_history WHERE novel_id IS NULL;
DELETE FROM conversations WHERE novel_id IS NULL;

-- 3. Enforce it, so the database rejects an orphan instead of storing one.

ALTER TABLE chapters ALTER COLUMN novel_id SET NOT NULL;
ALTER TABLE characters ALTER COLUMN novel_id SET NOT NULL;
ALTER TABLE character_mentions ALTER COLUMN novel_id SET NOT NULL;
ALTER TABLE character_relationships ALTER COLUMN novel_id SET NOT NULL;
ALTER TABLE chapter_summaries ALTER COLUMN novel_id SET NOT NULL;
ALTER TABLE reading_progress ALTER COLUMN novel_id SET NOT NULL;
ALTER TABLE search_history ALTER COLUMN novel_id SET NOT NULL;
ALTER TABLE conversations ALTER COLUMN novel_id SET NOT NULL;
