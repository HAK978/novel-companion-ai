-- chapter_summaries has always cached /summarize answers for chapter *ranges*. Its name now
-- goes to summaries of single chapters (007). A renamed table keeps the names of its
-- sequence, indexes and constraints, so they are renamed with it; otherwise the range table
-- would keep chapter_summaries_pkey and 007's table would get chapter_summaries_pkey1.
ALTER TABLE chapter_summaries RENAME TO range_summaries;

DO $$
DECLARE
    item record;
BEGIN
    FOR item IN SELECT conname AS name FROM pg_constraint
                WHERE conrelid = 'range_summaries'::regclass
                  AND conname LIKE 'chapter\_summaries%'
    LOOP
        -- renaming a constraint that is backed by an index renames the index as well
        EXECUTE format('ALTER TABLE range_summaries RENAME CONSTRAINT %I TO %I',
                       item.name, replace(item.name, 'chapter_summaries', 'range_summaries'));
    END LOOP;
    FOR item IN SELECT indexname AS name FROM pg_indexes
                WHERE tablename = 'range_summaries' AND indexname LIKE 'chapter\_summaries%'
    LOOP
        EXECUTE format('ALTER INDEX %I RENAME TO %I',
                       item.name, replace(item.name, 'chapter_summaries', 'range_summaries'));
    END LOOP;
END $$;

ALTER SEQUENCE IF EXISTS chapter_summaries_id_seq RENAME TO range_summaries_id_seq;
