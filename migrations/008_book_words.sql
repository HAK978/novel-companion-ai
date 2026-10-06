-- Where each word of a novel first appears, so text written for a reader (answers, chapter
-- summaries) can be checked for names the reader has not reached yet: the model may know the
-- book, and at chapter 14 of the Hound it called Stapleton's wife "Beryl Garcia", a name the
-- book first uses in chapter 15. Words are reduced as in words.py (lowercase, plurals and
-- possessives folded).
-- first_lowercase_chapter: the first chapter writing the word in lowercase. NULL means the
-- book only ever capitalizes it, which makes it a name.
CREATE TABLE IF NOT EXISTS book_words (
    novel_id INTEGER NOT NULL REFERENCES novels(id) ON DELETE CASCADE,
    word TEXT NOT NULL,
    first_chapter INTEGER NOT NULL,
    first_lowercase_chapter INTEGER,
    PRIMARY KEY (novel_id, word)
);
