"""Database scoping, round trips and stale-meeting handling. Temp database, no API calls.
Run by pytest (uv run pytest).
"""
import pytest

from services import db, store

SEGMENTS = [{"id": 0, "start": 0.0, "end": 2.0, "text": "Hello.", "tokens": [1, 2],
             "avg_logprob": -0.1, "text_groq": "Hello"}]
RESULT = {"model": "m", "summary": {"overview": "o", "key_topics": ["t"]},
          "extraction": {"decisions": [], "action_items": [], "open_topics": ["x"]},
          "steps": [{"step": "summary", "seconds": 1.0, "calls": 1, "total_tokens": 10}]}


@pytest.fixture
def users(temp_db):
    store.create_user("alice", "hash-a", "user")
    store.create_user("bob", "hash-b", "user")
    return store.get_user_by_username("alice")["id"], store.get_user_by_username("bob")["id"]


def test_schema_settings(temp_db):
    store.get_user_by_username("nobody")  # first connection creates the schema
    with db.connection() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_round_trip(users):
    alice, _ = users
    mid = store.create_meeting(alice, "Standup", "standup.mp4")
    assert store.save_transcript(alice, mid, SEGMENTS, "whisper", 2.0,
                                 timings={"extract": 1.0, "transcribe": 2.0})
    assert store.save_analysis(alice, mid, RESULT, timings={"summary": 3.0, "extraction": 4.0})
    m = store.get_meeting(alice, mid)
    assert m["status"] == "done" and m["duration_seconds"] == 2.0
    assert m["transcript"]["segments"] == [{"id": 0, "start": 0.0, "end": 2.0, "text": "Hello."}]
    assert m["analysis"]["summary"] == RESULT["summary"]
    assert m["analysis"]["extraction"] == RESULT["extraction"]
    assert m["analysis"]["steps"] == RESULT["steps"]
    assert m["timings"] == {"extract": 1.0, "transcribe": 2.0, "summary": 3.0, "extraction": 4.0}


def test_retry_updates_only_its_own_timing_keys(users):
    alice, _ = users
    mid = store.create_meeting(alice, "t", "f.mp4")
    store.save_transcript(alice, mid, SEGMENTS, "whisper", 2.0, timings={"extract": 1.0, "transcribe": 2.0})
    store.save_analysis(alice, mid, RESULT, timings={"summary": 3.0, "extraction": 4.0})
    store.save_analysis(alice, mid, RESULT, timings={"summary": 5.0, "extraction": 6.0})
    assert store.get_meeting(alice, mid)["timings"] == {
        "extract": 1.0, "transcribe": 2.0, "summary": 5.0, "extraction": 6.0}


def test_other_user_cannot_read_list_or_delete(users):
    alice, bob = users
    mid = store.create_meeting(alice, "Secret", "s.mp4")
    store.save_transcript(alice, mid, SEGMENTS, "whisper", 2.0)
    assert store.get_meeting(bob, mid) is None
    assert store.get_meeting(bob, 99999) is None  # same answer as a missing id
    assert store.get_segments(bob, mid) is None
    assert store.list_meetings(bob) == []
    assert store.delete_meeting(bob, mid) is False
    assert store.get_meeting(alice, mid) is not None  # still there


def test_other_user_cannot_write(users):
    alice, bob = users
    mid = store.create_meeting(alice, "Secret", "s.mp4")
    assert store.update_status(bob, mid, "failed", "x") is False
    assert store.save_transcript(bob, mid, SEGMENTS, "whisper", 2.0) is False
    assert store.save_analysis(bob, mid, RESULT) is False
    m = store.get_meeting(alice, mid)
    assert m["status"] == "transcribing" and m["transcript"] is None and m["analysis"] is None


def test_unknown_status_rejected(users):
    alice, _ = users
    mid = store.create_meeting(alice, "t", "f.mp4")
    with pytest.raises(ValueError):
        store.update_status(alice, mid, "finished")


def test_delete_removes_transcript_and_analysis(users):
    alice, _ = users
    mid = store.create_meeting(alice, "t", "f.mp4")
    store.save_transcript(alice, mid, SEGMENTS, "whisper", 2.0)
    store.save_analysis(alice, mid, RESULT)
    assert store.delete_meeting(alice, mid) is True
    with db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 0


def _age(meeting_id: int, minutes: int) -> None:
    """Test helper: pretend the meeting's last status change was long ago."""
    from datetime import datetime, timedelta, timezone
    old = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")
    with db.connection() as conn:
        conn.execute("UPDATE meetings SET updated_at = ? WHERE id = ?", (old, meeting_id))


def test_stale_meetings_marked_failed_by_list(users):
    alice, bob = users
    stuck_transcribing = store.create_meeting(alice, "a", "a.mp4")
    stuck_analyzing = store.create_meeting(alice, "b", "b.mp4")
    store.save_transcript(alice, stuck_analyzing, SEGMENTS, "whisper", 2.0)
    fresh = store.create_meeting(alice, "c", "c.mp4")
    bobs = store.create_meeting(bob, "d", "d.mp4")
    for mid in (stuck_transcribing, stuck_analyzing, bobs):
        _age(mid, 31)

    statuses = {m["id"]: (m["status"], m["error_message"]) for m in store.list_meetings(alice)}
    assert statuses[stuck_transcribing] == ("failed", store.INTERRUPTED_TRANSCRIBING)
    assert statuses[stuck_analyzing] == ("analysis_failed", store.INTERRUPTED_ANALYZING)
    assert statuses[fresh] == ("transcribing", None)
    assert store.get_meeting(bob, bobs)["status"] == "transcribing"  # only alice's were touched
    assert store.get_segments(alice, stuck_analyzing) is not None  # transcript kept for retry


def test_mark_stale_leaves_recent_and_finished_meetings(users):
    alice, _ = users
    recent = store.create_meeting(alice, "a", "a.mp4")
    _age(recent, 5)
    done = store.create_meeting(alice, "b", "b.mp4")
    store.save_analysis(alice, done, RESULT)
    _age(done, 120)
    assert store.mark_stale(alice) == 0


def test_transcribing_goes_stale_after_10_minutes_transcribed_after_30(users):
    alice, _ = users
    transcribing = store.create_meeting(alice, "a", "a.mp4")
    transcribed = store.create_meeting(alice, "b", "b.mp4")
    store.save_transcript(alice, transcribed, SEGMENTS, "whisper", 2.0)
    _age(transcribing, 11)
    _age(transcribed, 11)
    assert store.mark_stale(alice) == 1
    assert store.get_meeting(alice, transcribing)["status"] == "failed"
    assert store.get_meeting(alice, transcribed)["status"] == "transcribed"  # 30-minute rule
