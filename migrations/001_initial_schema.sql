-- Chapters metadata
CREATE TABLE IF NOT EXISTS chapters (
    id SERIAL PRIMARY KEY,
    chapter_number INTEGER UNIQUE NOT NULL,
    title TEXT,
    word_count INTEGER,
    volume INTEGER,
    ingestion_status TEXT DEFAULT 'pending',
    ingested_at TIMESTAMPTZ
);

-- Reading progress per user
CREATE TABLE IF NOT EXISTS reading_progress (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    current_chapter INTEGER NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id)
);

-- Characters extracted from chapters
CREATE TABLE IF NOT EXISTS characters (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    aliases TEXT[],
    first_appearance INTEGER,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Character relationships
CREATE TABLE IF NOT EXISTS character_relationships (
    id SERIAL PRIMARY KEY,
    character_a_id INTEGER REFERENCES characters(id),
    character_b_id INTEGER REFERENCES characters(id),
    relationship_type TEXT,
    description TEXT,
    first_chapter INTEGER,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(character_a_id, character_b_id, relationship_type)
);

-- Character mentions per chapter
CREATE TABLE IF NOT EXISTS character_mentions (
    id SERIAL PRIMARY KEY,
    character_id INTEGER REFERENCES characters(id),
    chapter_number INTEGER,
    context TEXT,
    UNIQUE(character_id, chapter_number)
);

-- Cached chapter summaries
CREATE TABLE IF NOT EXISTS chapter_summaries (
    id SERIAL PRIMARY KEY,
    start_chapter INTEGER NOT NULL,
    end_chapter INTEGER NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(start_chapter, end_chapter)
);

-- Search history
CREATE TABLE IF NOT EXISTS search_history (
    id SERIAL PRIMARY KEY,
    user_id TEXT DEFAULT 'default',
    query TEXT NOT NULL,
    results_count INTEGER,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Conversation logs
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT DEFAULT 'default',
    query TEXT,
    response TEXT,
    sources JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);
