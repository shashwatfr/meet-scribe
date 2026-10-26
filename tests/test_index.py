"""Chunking, vector storage, indexing and the version 1 -> 2 migration. No API calls.
Run by pytest (uv run pytest).
"""
import sqlite3

import numpy as np
import pytest

from services import db, index, store
from services.index import MAX_WORDS, MIN_WORDS, from_blob, make_chunks, to_blob


def seg(i: int, words: int, start: float | None = None) -> dict:
    start = i * 10.0 if start is None else start
    return {"id": i, "start": start, "end": start + 9, "text": " ".join(f"w{i}x{j}" for j in range(words))}


def words(chunk: dict) -> int:
    return len(chunk["text"].split())


# ---------- chunking ----------

def test_chunks_are_150_to_250_words_with_one_segment_overlap():
    segments = [seg(i, 40) for i in range(20)]  # 800 words
    chunks = make_chunks(segments)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert MIN_WORDS <= words(chunk) <= MAX_WORDS
    for a, b in zip(chunks, chunks[1:]):
        last_segment_of_a = a["text"].split()[-40:]
        assert b["text"].split()[:40] == last_segment_of_a  # one segment of overlap
        assert b["start"] == a["end"] - 9  # b starts at the overlapping segment's start


def test_chunk_times_come_from_their_segments():
    segments = [seg(i, 60) for i in range(6)]
    chunks = make_chunks(segments)
    assert chunks[0]["start"] == 0.0
    assert chunks[-1]["end"] == segments[-1]["end"]
    assert all(c["start"] < c["end"] for c in chunks)


def test_long_segment_is_its_own_chunk_and_never_cut():
    segments = [seg(0, 100), seg(1, 400), seg(2, 100), seg(3, 100)]
    chunks = make_chunks(segments)
    long_chunks = [c for c in chunks if "w1x0" in c["text"]]
    assert any(words(c) >= 400 for c in long_chunks)
    assert all(c["text"].count("w1x399") <= 1 for c in chunks)
    assert "w3x99" in chunks[-1]["text"]  # nothing lost at the end
    assert len(chunks) < 10  # no endless loop of tiny chunks


def test_short_meeting_is_one_chunk_and_empty_is_none():
    assert len(make_chunks([seg(0, 20), seg(1, 30)])) == 1
    assert make_chunks([]) == []
    assert make_chunks([{"id": 0, "start": 0, "end": 1, "text": "  "}]) == []


def test_every_word_is_in_some_chunk():
    segments = [seg(i, n) for i, n in enumerate([30, 200, 10, 90, 260, 5, 120, 80])]
    text = " ".join(c["text"] for c in make_chunks(segments))
    for s in segments:
        assert s["text"] in text


# ---------- vectors ----------

def test_float32_blob_round_trip():
    vector = np.random.default_rng(1).standard_normal(1536).astype(np.float32)
    blob = to_blob(vector)
    assert len(blob) == 1536 * 4
    assert np.array_equal(from_blob(blob), vector)


# ---------- indexing ----------

@pytest.fixture
def users(temp_db):
    store.create_user("alice", "h", "user")
    store.create_user("bob", "h", "user")
    return store.get_user_by_username("alice")["id"], store.get_user_by_username("bob")["id"]


def meeting_with_transcript(user_id: int, segments=None) -> int:
    mid = store.create_meeting(user_id, "m", "m.mp4")
    store.save_transcript(user_id, mid, segments or [seg(i, 60) for i in range(8)], "whisper", 80.0)
    return mid


def test_index_meeting_replaces_old_chunks(users):
    alice, _ = users
    mid = meeting_with_transcript(alice)
    first = index.index_meeting(alice, mid)
    second = index.index_meeting(alice, mid)
    assert first == second == len(store.load_chunks(alice, index.EMBEDDING_MODEL, mid))
    assert mid in index.indexed_ids(alice)


def test_index_is_scoped(users):
    alice, bob = users
    mid = meeting_with_transcript(alice)
    assert index.index_meeting(bob, mid) is None
    assert store.replace_chunks(bob, mid, [], index.EMBEDDING_MODEL) is False
    assert index.indexed_ids(bob) == set()


def test_other_model_counts_as_not_indexed(users):
    alice, _ = users
    mid = meeting_with_transcript(alice)
    store.replace_chunks(alice, mid, [{"start": 0, "end": 1, "text": "x", "embedding": to_blob([1, 0])}],
                         "some-old-model")
    assert mid not in index.indexed_ids(alice)
    assert store.load_chunks(alice, index.EMBEDDING_MODEL) == []
    assert index.unindexed_ids(alice) == [mid]
    count = len(make_chunks(store.get_segments(alice, mid)))
    assert index.index_missing(alice) == [(mid, f"{count} chunks")]
    assert mid in index.indexed_ids(alice)


def test_failed_embedding_keeps_old_chunks(users, monkeypatch):
    alice, _ = users
    mid = meeting_with_transcript(alice)
    index.index_meeting(alice, mid)
    before = store.load_chunks(alice, index.EMBEDDING_MODEL, mid)

    def broken(texts):
        raise index.EmbeddingError("OpenAI rate limit hit. Try again in a minute.")

    monkeypatch.setattr(index, "embed_texts", broken)
    with pytest.raises(index.EmbeddingError):
        index.index_meeting(alice, mid)
    assert store.load_chunks(alice, index.EMBEDDING_MODEL, mid) == before
    assert index.index_missing(alice) == []  # already indexed, nothing to do


def test_delete_meeting_removes_chunks(users):
    alice, _ = users
    mid = meeting_with_transcript(alice)
    index.index_meeting(alice, mid)
    store.delete_meeting(alice, mid)
    with db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


# ---------- migration ----------

def test_migration_from_version_1_keeps_meetings(temp_db):
    conn = sqlite3.connect(temp_db)  # build a version-1 file by hand, as Phase 3 left it
    conn.executescript(db.SCHEMA_V1)
    conn.execute("INSERT INTO users (username, password_hash, role, created_at) "
                 "VALUES ('alice', 'h', 'user', '2026-10-25T10:00:00+00:00')")
    conn.execute("INSERT INTO meetings (user_id, title, source_filename, status, created_at, updated_at) "
                 "VALUES (1, 'Old meeting', 'old.mp4', 'done', '2026-10-25T10:00:00+00:00', "
                 "'2026-10-25T10:00:00+00:00')")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    assert store.get_meeting(1, 1)["title"] == "Old meeting"  # first connection migrates
    with db.connection() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "chunks" in tables
    assert index.unindexed_ids(1) == []  # 'done' but no transcript row: nothing to index
