-- One summary per chapter, written from that chapter's text alone, so that broad questions
-- ("catch me up") can be answered without reading every chapter again. A reader is only
-- shown summaries of chapters they have read, which makes them spoiler-safe by construction.
-- unverified_names: names a summary uses that its chapter never mentions, which may have
-- come from the model's memory of the book rather than the text.
-- ON DELETE CASCADE: summaries are written in the background, and one that lands while its
-- novel is being deleted must not block the delete.
CREATE TABLE IF NOT EXISTS chapter_summaries (
    id SERIAL PRIMARY KEY,
    novel_id INTEGER NOT NULL REFERENCES novels(id) ON DELETE CASCADE,
    chapter_number INTEGER NOT NULL,
    summary TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version INTEGER NOT NULL,
    unverified_names TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chapter_summaries_novel_chapter_unique UNIQUE (novel_id, chapter_number)
);
