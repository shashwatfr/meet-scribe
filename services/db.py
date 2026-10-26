"""The SQLite database: the connection and the table definitions (schema), nothing else.

store.py and admin.py open their connections here. Schema upgrades (migrations) run
automatically the first time the database is opened.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/meetscribe.db")
SCHEMA_VERSION = 2  # bump and add a step to MIGRATIONS when the schema changes

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('user', 'admin')),
    created_at    TEXT NOT NULL
);

-- status is checked in code (store.STATUSES), not here, so new statuses need no migration.
CREATE TABLE IF NOT EXISTS meetings (
    id               INTEGER PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title            TEXT NOT NULL,
    source_filename  TEXT NOT NULL,
    duration_seconds REAL,
    status           TEXT NOT NULL,
    error_message    TEXT,
    timings_json     TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL  -- last status change; used to find interrupted meetings
);
CREATE INDEX IF NOT EXISTS meetings_user ON meetings(user_id);

CREATE TABLE IF NOT EXISTS transcripts (
    meeting_id    INTEGER PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    language      TEXT,
    model         TEXT NOT NULL,
    segments_json TEXT NOT NULL  -- [{id, start, end, text}, ...]
);

CREATE TABLE IF NOT EXISTS analyses (
    meeting_id      INTEGER PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    summary_json    TEXT NOT NULL,
    extraction_json TEXT NOT NULL,
    model           TEXT NOT NULL,
    tokens_json     TEXT NOT NULL,  -- per step: seconds, calls, tokens, reasoning tokens
    created_at      TEXT NOT NULL
);
"""

# Version 2: transcript chunks with their embeddings, for Q&A. "end" is quoted (SQL keyword).
SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY,
    meeting_id      INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    idx             INTEGER NOT NULL,
    start           REAL NOT NULL,
    "end"           REAL NOT NULL,
    text            TEXT NOT NULL,
    embedding       BLOB NOT NULL,  -- float32 vector as raw bytes
    embedding_model TEXT NOT NULL   -- only rows of the current model are searched
);
CREATE INDEX IF NOT EXISTS chunks_meeting ON chunks(meeting_id, embedding_model);
"""

MIGRATIONS = {1: SCHEMA_V1, 2: SCHEMA_V2}  # version -> script that brings the file to it


def now() -> str:
    """Current UTC time as ISO text. Same format everywhere, so strings sort by time."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create or upgrade the schema. Each step only adds; existing data is kept."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= SCHEMA_VERSION:
        return
    for step in range(version + 1, SCHEMA_VERSION + 1):
        conn.executescript(MIGRATIONS[step])  # IF NOT EXISTS: safe to re-run
    if version == 0:
        conn.execute("PRAGMA journal_mode = WAL")  # stored in the file, so set once
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


@contextmanager
def connection():
    """One connection per operation: commits on success, rolls back on error, always closes."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5)  # wait up to 5 s if another writer holds the lock
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # off by default in SQLite, per connection
    try:
        _ensure_schema(conn)
        with conn:
            yield conn
    finally:
        conn.close()
