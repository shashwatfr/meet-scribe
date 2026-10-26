"""Admin-only database queries for the Admin page (pages/3_Admin.py).

Every function first checks IN THE DATABASE that the caller is an admin; the role is
never taken from the browser session. Password hashes become "(hidden)" in the query.
"""
from services import db

MAX_CELL_CHARS = 200

# One fixed query per table (table names never come from user input).
# Every table gets a username column; password hashes are replaced in the query itself.
TABLE_QUERIES = {
    "users": "SELECT id, username, '(hidden)' AS password_hash, role, created_at "
             "FROM users ORDER BY id",
    "meetings": "SELECT u.username, m.* FROM meetings m JOIN users u ON u.id = m.user_id "
                "ORDER BY m.id",
    "transcripts": "SELECT u.username, t.* FROM transcripts t "
                   "JOIN meetings m ON m.id = t.meeting_id JOIN users u ON u.id = m.user_id "
                   "ORDER BY t.meeting_id",
    "analyses": "SELECT u.username, a.* FROM analyses a "
                "JOIN meetings m ON m.id = a.meeting_id JOIN users u ON u.id = m.user_id "
                "ORDER BY a.meeting_id",
    # The vector itself is not shown, only its size and model, e.g. "(6144 bytes, text-embedding-3-small)".
    "chunks": 'SELECT u.username, c.id, c.meeting_id, c.idx, c.start, c."end" AS "end", c.text, '
              "'(' || length(c.embedding) || ' bytes, ' || c.embedding_model || ')' AS embedding "
              "FROM chunks c JOIN meetings m ON m.id = c.meeting_id JOIN users u ON u.id = m.user_id "
              "ORDER BY c.meeting_id, c.idx",
}


def require_admin(user_id: int) -> None:
    """Raise PermissionError unless this user id is an admin in the database."""
    with db.connection() as conn:
        row = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None or row["role"] != "admin":
        raise PermissionError("Admins only.")


def row_counts(user_id: int) -> dict[str, int]:
    require_admin(user_id)
    with db.connection() as conn:
        return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in TABLE_QUERIES}


def db_file_size(user_id: int) -> dict[str, int]:
    """Bytes on disk: the main file and the WAL file (recent writes not yet merged in)."""
    require_admin(user_id)
    wal = db.DB_PATH.with_name(db.DB_PATH.name + "-wal")
    return {"database": db.DB_PATH.stat().st_size if db.DB_PATH.exists() else 0,
            "wal": wal.stat().st_size if wal.exists() else 0}


def _shorten(value):
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return f"{value[:MAX_CELL_CHARS]}...({len(value):,} chars)"
    return value


def table_rows(user_id: int, table: str) -> list[dict]:
    """Rows of one table as stored, long values cut to 200 characters."""
    require_admin(user_id)
    if table not in TABLE_QUERIES:
        raise ValueError(f"Unknown table: {table}")
    with db.connection() as conn:
        rows = conn.execute(TABLE_QUERIES[table]).fetchall()
    return [{key: _shorten(row[key]) for key in row.keys()} for row in rows]
