-- Scope uniqueness to the novel.
--
-- 002 added novel_id columns and re-scoped `chapters` and `reading_progress`,
-- but left `characters` unique by name alone and `chapter_summaries` unique by
-- chapter range alone. Two consequences:
--
--   * ingestion upserts with ON CONFLICT (novel_id, name), which has no
--     matching constraint on a fresh database, so character storage fails;
--   * summarizing chapters 1-10 of one novel overwrites the cached summary for
--     chapters 1-10 of every other novel.

ALTER TABLE characters DROP CONSTRAINT IF EXISTS characters_name_key;
ALTER TABLE characters DROP CONSTRAINT IF EXISTS characters_novel_name_unique;
ALTER TABLE characters ADD CONSTRAINT characters_novel_name_unique UNIQUE (novel_id, name);

ALTER TABLE chapter_summaries
    DROP CONSTRAINT IF EXISTS chapter_summaries_start_chapter_end_chapter_key;
ALTER TABLE chapter_summaries
    DROP CONSTRAINT IF EXISTS chapter_summaries_novel_range_unique;
ALTER TABLE chapter_summaries
    ADD CONSTRAINT chapter_summaries_novel_range_unique
    UNIQUE (novel_id, start_chapter, end_chapter);
