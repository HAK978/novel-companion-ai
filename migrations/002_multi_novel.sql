-- Multi-novel support

CREATE TABLE IF NOT EXISTS novels (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    source_type TEXT,
    source_url TEXT,
    total_chapters INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add novel_id to existing tables
ALTER TABLE chapters ADD COLUMN IF NOT EXISTS novel_id INTEGER REFERENCES novels(id);
ALTER TABLE characters ADD COLUMN IF NOT EXISTS novel_id INTEGER REFERENCES novels(id);
ALTER TABLE character_relationships ADD COLUMN IF NOT EXISTS novel_id INTEGER REFERENCES novels(id);
ALTER TABLE character_mentions ADD COLUMN IF NOT EXISTS novel_id INTEGER REFERENCES novels(id);
ALTER TABLE chapter_summaries ADD COLUMN IF NOT EXISTS novel_id INTEGER REFERENCES novels(id);
ALTER TABLE reading_progress ADD COLUMN IF NOT EXISTS novel_id INTEGER REFERENCES novels(id);
ALTER TABLE search_history ADD COLUMN IF NOT EXISTS novel_id INTEGER REFERENCES novels(id);
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS novel_id INTEGER REFERENCES novels(id);

-- Update unique constraints to include novel_id
ALTER TABLE chapters DROP CONSTRAINT IF EXISTS chapters_chapter_number_key;
ALTER TABLE chapters ADD CONSTRAINT chapters_novel_chapter_unique UNIQUE (novel_id, chapter_number);

ALTER TABLE reading_progress DROP CONSTRAINT IF EXISTS reading_progress_user_id_key;
ALTER TABLE reading_progress ADD CONSTRAINT reading_progress_novel_user_unique UNIQUE (novel_id, user_id);
