"""All database queries for users, meetings, transcripts, analyses and chunks.

Every meeting query is filtered by user_id, so someone else's meeting looks exactly like
one that does not exist (None / False). Called by the pipeline, the pages and the CLIs.
"""
import json
from datetime import datetime, timedelta, timezone

from services import db

STATUSES = ("transcribing", "transcribed", "done", "analysis_failed", "failed")
INTERRUPTED_TRANSCRIBING = "Interrupted, please upload again."
INTERRUPTED_ANALYZING = "Interrupted during analysis. The transcript is saved; retry the analysis."
MEETING_COLUMNS = ("id, title, source_filename, duration_seconds, status, error_message, "
                   "timings_json, created_at, updated_at")


# ---------- users ----------

def get_user_by_username(username: str) -> dict | None:
    with db.connection() as conn:
        row = conn.execute("SELECT id, username, password_hash, role FROM users "
                           "WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def count_users() -> int:
    with db.connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def create_user(username: str, password_hash: str, role: str) -> bool:
    """Insert a user. Returns False if the username is already taken."""
    with db.connection() as conn:
        cur = conn.execute("INSERT OR IGNORE INTO users (username, password_hash, role, created_at) "
                           "VALUES (?, ?, ?, ?)", (username, password_hash, role, db.now()))
    return cur.rowcount == 1


# ---------- meetings ----------

def _merge_timings(conn, user_id: int, meeting_id: int, timings: dict | None) -> None:
    """Update only the given keys of timings_json, so a retry keeps the transcription times."""
    if not timings:
        return
    row = conn.execute("SELECT timings_json FROM meetings WHERE id = ? AND user_id = ?",
                       (meeting_id, user_id)).fetchone()
    merged = {**json.loads(row["timings_json"]), **timings}
    conn.execute("UPDATE meetings SET timings_json = ? WHERE id = ? AND user_id = ?",
                 (json.dumps(merged), meeting_id, user_id))


def create_meeting(user_id: int, title: str, source_filename: str) -> int:
    with db.connection() as conn:
        cur = conn.execute(
            "INSERT INTO meetings (user_id, title, source_filename, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'transcribing', ?, ?)",
            (user_id, title, source_filename, db.now(), db.now()))
    return cur.lastrowid


def update_status(user_id: int, meeting_id: int, status: str,
                  error_message: str | None = None) -> bool:
    """Set status (and error message). Returns False if no such meeting for this user."""
    if status not in STATUSES:
        raise ValueError(f"Unknown status: {status}")
    with db.connection() as conn:
        cur = conn.execute("UPDATE meetings SET status = ?, error_message = ?, updated_at = ? "
                           "WHERE id = ? AND user_id = ?",
                           (status, error_message, db.now(), meeting_id, user_id))
    return cur.rowcount == 1


def save_transcript(user_id: int, meeting_id: int, segments: list[dict], model: str,
                    duration_seconds: float, language: str | None = None,
                    timings: dict | None = None) -> bool:
    """Save the transcript and set status 'transcribed', in one transaction.

    Only id, start, end and text are kept for each segment (no token arrays).
    Returns False if no such meeting for this user.
    """
    slim = [{k: s[k] for k in ("id", "start", "end", "text")} for s in segments]
    with db.connection() as conn:
        cur = conn.execute("UPDATE meetings SET status = 'transcribed', error_message = NULL, "
                           "duration_seconds = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                           (duration_seconds, db.now(), meeting_id, user_id))
        if cur.rowcount == 0:
            return False
        conn.execute("INSERT OR REPLACE INTO transcripts (meeting_id, language, model, segments_json) "
                     "VALUES (?, ?, ?, ?)", (meeting_id, language, model, json.dumps(slim)))
        _merge_timings(conn, user_id, meeting_id, timings)
    return True


def save_analysis(user_id: int, meeting_id: int, result: dict,
                  timings: dict | None = None) -> bool:
    """Save an analyze() result and set status 'done', in one transaction.

    A new analysis replaces the old one. Returns False if no such meeting for this user.
    """
    with db.connection() as conn:
        cur = conn.execute("UPDATE meetings SET status = 'done', error_message = NULL, "
                           "updated_at = ? WHERE id = ? AND user_id = ?",
                           (db.now(), meeting_id, user_id))
        if cur.rowcount == 0:
            return False
        conn.execute(
            "INSERT OR REPLACE INTO analyses (meeting_id, summary_json, extraction_json, model, "
            "tokens_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (meeting_id, json.dumps(result["summary"]), json.dumps(result["extraction"]),
             result["model"], json.dumps(result["steps"]), db.now()))
        _merge_timings(conn, user_id, meeting_id, timings)
    return True


STALE_RULES = (  # (stuck status, minutes without progress, new status, message)
    ("transcribing", 10, "failed", INTERRUPTED_TRANSCRIBING),
    ("transcribed", 30, "analysis_failed", INTERRUPTED_ANALYZING),
)


def mark_stale(user_id: int) -> int:
    """Fail meetings stuck mid-pipeline (e.g. the page was closed). Returns how many changed.

    'transcribing' for 10+ minutes -> 'failed' (no transcript to keep);
    'transcribed' for 30+ minutes -> 'analysis_failed' (transcript saved, retry possible).
    """
    changed = 0
    with db.connection() as conn:
        for stuck, minutes, new, message in STALE_RULES:
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")
            cur = conn.execute("UPDATE meetings SET status = ?, error_message = ?, updated_at = ? "
                               "WHERE user_id = ? AND status = ? AND updated_at < ?",
                               (new, message, db.now(), user_id, stuck, cutoff))
            changed += cur.rowcount
    return changed


def list_meetings(user_id: int) -> list[dict]:
    """The user's meetings, newest first. Marks interrupted ones as failed first."""
    mark_stale(user_id)
    with db.connection() as conn:
        rows = conn.execute(f"SELECT {MEETING_COLUMNS} FROM meetings WHERE user_id = ? "
                            "ORDER BY created_at DESC, id DESC", (user_id,)).fetchall()
    return [_meeting_dict(r) for r in rows]


def _meeting_dict(row) -> dict:
    meeting = dict(row)
    meeting["timings"] = json.loads(meeting.pop("timings_json"))
    return meeting


def get_meeting(user_id: int, meeting_id: int) -> dict | None:
    """One meeting with its transcript segments and analysis (None where missing)."""
    with db.connection() as conn:
        row = conn.execute(f"SELECT {MEETING_COLUMNS} FROM meetings WHERE id = ? AND user_id = ?",
                           (meeting_id, user_id)).fetchone()
        if row is None:
            return None
        transcript = conn.execute("SELECT language, model, segments_json FROM transcripts "
                                  "WHERE meeting_id = ?", (meeting_id,)).fetchone()
        analysis = conn.execute("SELECT summary_json, extraction_json, model, tokens_json, "
                                "created_at FROM analyses WHERE meeting_id = ?",
                                (meeting_id,)).fetchone()
    meeting = _meeting_dict(row)
    meeting["transcript"] = None if transcript is None else {
        "language": transcript["language"], "model": transcript["model"],
        "segments": json.loads(transcript["segments_json"])}
    meeting["analysis"] = None if analysis is None else {
        "summary": json.loads(analysis["summary_json"]),
        "extraction": json.loads(analysis["extraction_json"]),
        "model": analysis["model"], "steps": json.loads(analysis["tokens_json"]),
        "created_at": analysis["created_at"]}
    return meeting


def get_segments(user_id: int, meeting_id: int) -> list[dict] | None:
    """Saved transcript segments, or None if not found / not this user's / no transcript."""
    with db.connection() as conn:
        row = conn.execute("SELECT t.segments_json FROM transcripts t "
                           "JOIN meetings m ON m.id = t.meeting_id "
                           "WHERE m.id = ? AND m.user_id = ?", (meeting_id, user_id)).fetchone()
    return json.loads(row["segments_json"]) if row else None


def update_timings(user_id: int, meeting_id: int, timings: dict) -> bool:
    """Merge step timings (e.g. {"index": 1.2}) into the meeting. False if not this user's."""
    with db.connection() as conn:
        exists = conn.execute("SELECT 1 FROM meetings WHERE id = ? AND user_id = ?",
                              (meeting_id, user_id)).fetchone()
        if exists:
            _merge_timings(conn, user_id, meeting_id, timings)
    return exists is not None


# ---------- chunks (Q&A index) ----------

def replace_chunks(user_id: int, meeting_id: int, chunks: list[dict], model: str) -> bool:
    """Replace all chunks of a meeting, in one transaction. False if not this user's meeting.

    Each chunk: {start, end, text, embedding (bytes)}.
    """
    with db.connection() as conn:
        exists = conn.execute("SELECT 1 FROM meetings WHERE id = ? AND user_id = ?",
                              (meeting_id, user_id)).fetchone()
        if exists is None:
            return False
        conn.execute("DELETE FROM chunks WHERE meeting_id = ?", (meeting_id,))
        conn.executemany(
            'INSERT INTO chunks (meeting_id, idx, start, "end", text, embedding, embedding_model) '
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(meeting_id, i, c["start"], c["end"], c["text"], c["embedding"], model)
             for i, c in enumerate(chunks)])
    return True


def load_chunks(user_id: int, model: str, meeting_id: int | None = None) -> list[dict]:
    """This user's chunks made with `model`: one meeting, or all their meetings."""
    sql = ('SELECT c.id AS chunk_id, c.meeting_id, m.title, c.idx, c.start, c."end" AS "end", '
           "c.text, c.embedding FROM chunks c JOIN meetings m ON m.id = c.meeting_id "
           "WHERE m.user_id = ? AND c.embedding_model = ?")
    params = [user_id, model]
    if meeting_id is not None:
        sql += " AND m.id = ?"
        params.append(meeting_id)
    with db.connection() as conn:
        rows = conn.execute(sql + " ORDER BY c.meeting_id, c.idx", params).fetchall()
    return [dict(r) for r in rows]


def transcript_meeting_ids(user_id: int) -> list[int]:
    """The user's meetings that have a saved transcript, oldest first."""
    with db.connection() as conn:
        rows = conn.execute("SELECT m.id FROM meetings m JOIN transcripts t ON t.meeting_id = m.id "
                            "WHERE m.user_id = ? ORDER BY m.id", (user_id,)).fetchall()
    return [r["id"] for r in rows]


def indexed_meeting_ids(user_id: int, model: str) -> set[int]:
    """The user's meetings that have at least one chunk made with `model`."""
    with db.connection() as conn:
        rows = conn.execute("SELECT DISTINCT c.meeting_id FROM chunks c "
                            "JOIN meetings m ON m.id = c.meeting_id "
                            "WHERE m.user_id = ? AND c.embedding_model = ?",
                            (user_id, model)).fetchall()
    return {r["meeting_id"] for r in rows}


def delete_meeting(user_id: int, meeting_id: int) -> bool:
    """Delete a meeting and (by cascade) its transcript, analysis and chunks."""
    with db.connection() as conn:
        cur = conn.execute("DELETE FROM meetings WHERE id = ? AND user_id = ?",
                           (meeting_id, user_id))
    return cur.rowcount == 1
