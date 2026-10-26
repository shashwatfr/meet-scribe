"""Pipeline status flow and file handling, with fake transcription and analysis.
Run by pytest (uv run pytest).
"""
import os

import pytest

from services import index, pipeline, store
from services.analyze import AnalyzeError
from services.audio import AudioError
from services.transcribe import TranscribeError

SEGMENTS = [{"id": 0, "start": 0.0, "end": 3.0, "text": "We agree.", "tokens": [5]}]
RESULT = {"model": "m", "summary": {"overview": "o", "key_topics": []},
          "extraction": {"decisions": [], "action_items": [], "open_topics": []},
          "steps": [{"step": "summary", "seconds": 1.5}, {"step": "extraction", "seconds": 2.5}]}


@pytest.fixture
def setup(temp_db, tmp_path, monkeypatch):
    """A user, an original recording, a temp uploads folder, and a fake transcriber."""
    monkeypatch.setattr(pipeline, "UPLOAD_DIR", tmp_path / "uploads")
    store.create_user("alice", "h", "user")
    original = tmp_path / "samples" / "meeting.mp4"
    original.parent.mkdir()
    original.write_bytes(b"original recording bytes")
    seen = {}

    def fake_transcribe(path, on_progress=None):
        seen["copy"] = path
        seen["copy_existed"] = path.exists()
        seen["copy_bytes"] = path.read_bytes()
        if seen.get("fail"):
            raise seen["fail"]
        return SEGMENTS, 3.0, {"extract": 0.1, "transcribe": 0.2}, "English"

    monkeypatch.setattr(pipeline, "_transcribe_file", fake_transcribe)
    monkeypatch.setattr(pipeline, "analyze", lambda segments, on_progress=None: RESULT)
    return store.get_user_by_username("alice")["id"], original, seen


def test_success_saves_everything_and_deletes_the_copy(setup):
    user, original, seen = setup
    stat_before = os.stat(original)
    m = pipeline.process_recording(original, user, "Weekly")
    assert m["status"] == "done" and m["title"] == "Weekly"
    assert m["transcript"]["segments"][0]["text"] == "We agree."
    assert m["transcript"]["language"] == "English"
    timings = dict(m["timings"])
    assert isinstance(timings.pop("index"), float)  # indexing time is measured, so not fixed
    assert timings == {"extract": 0.1, "transcribe": 0.2, "summary": 1.5, "extraction": 2.5}
    assert m["id"] in index.indexed_ids(user)
    # worked on a copy in uploads/, which is gone now
    assert seen["copy_existed"] and seen["copy"] != original
    assert seen["copy"].parent == pipeline.UPLOAD_DIR
    assert seen["copy_bytes"] == b"original recording bytes"
    assert not seen["copy"].exists()
    # the user's file is untouched
    assert original.read_bytes() == b"original recording bytes"
    assert os.stat(original).st_mtime == stat_before.st_mtime


def test_transcription_failure_sets_failed_and_deletes_the_copy(setup):
    user, original, seen = setup
    seen["fail"] = TranscribeError("Groq rate limit hit. Try again in about 5 minutes.")
    m = pipeline.process_recording(original, user)
    assert m["status"] == "failed" and "rate limit" in m["error_message"]
    assert m["transcript"] is None
    assert not seen["copy"].exists()
    assert original.exists()


def test_unexpected_transcription_error_is_saved_then_raised(setup):
    user, original, seen = setup
    seen["fail"] = RuntimeError("bug")
    with pytest.raises(pipeline.PipelineError) as err:
        pipeline.process_recording(original, user)
    assert isinstance(err.value.__cause__, RuntimeError)
    [m] = store.list_meetings(user)
    assert err.value.meeting_id == m["id"]
    assert m["status"] == "failed" and m["error_message"] == pipeline.UNEXPECTED_TRANSCRIBE
    assert not seen["copy"].exists()


def test_analysis_failure_keeps_transcript_and_retry_works(setup, monkeypatch):
    user, original, _ = setup

    def failing_analyze(segments, on_progress=None):
        raise AnalyzeError("Groq rate limit hit (asked to wait 600 s).")

    monkeypatch.setattr(pipeline, "analyze", failing_analyze)
    m = pipeline.process_recording(original, user)
    assert m["status"] == "analysis_failed" and "rate limit" in m["error_message"]
    assert m["transcript"]["segments"][0]["text"] == "We agree."
    assert m["analysis"] is None

    monkeypatch.setattr(pipeline, "analyze", lambda segments, on_progress=None: RESULT)
    monkeypatch.setattr(pipeline, "_transcribe_file", lambda *a: pytest.fail("must not re-transcribe"))
    m = pipeline.retry_analysis(user, m["id"])
    assert m["status"] == "done" and m["error_message"] is None and m["analysis"] is not None
    assert m["timings"]["transcribe"] == 0.2  # transcription timings kept


def test_retry_is_scoped_to_the_owner(setup):
    user, original, _ = setup
    m = pipeline.process_recording(original, user)
    store.create_user("bob", "h", "user")
    bob = store.get_user_by_username("bob")["id"]
    assert pipeline.retry_analysis(bob, m["id"]) is None


def test_bad_input_is_rejected_before_saving(setup, tmp_path):
    user, _, _ = setup
    with pytest.raises(AudioError, match="File not found"):
        pipeline.process_recording(tmp_path / "missing.mp4", user)
    notes = tmp_path / "notes.txt"
    notes.write_text("hi")
    with pytest.raises(AudioError, match="Unsupported file type"):
        pipeline.process_recording(notes, user)
    assert store.list_meetings(user) == []


def test_upload_copy_deleted_after_success(setup):
    import io
    user, _, seen = setup
    m = pipeline.process_upload(io.BytesIO(b"uploaded bytes"), "../../Team call.MP4", user)
    assert m["status"] == "done" and m["title"] == "Team call" and m["source_filename"] == "Team call.MP4"
    assert seen["copy_existed"] and seen["copy_bytes"] == b"uploaded bytes"
    assert seen["copy"].parent == pipeline.UPLOAD_DIR and seen["copy"].suffix == ".mp4"
    assert not seen["copy"].exists()


def test_upload_copy_deleted_after_failure(setup):
    import io
    user, _, seen = setup
    seen["fail"] = AudioError("This file has no audio.")
    m = pipeline.process_upload(io.BytesIO(b"x"), "silent.mp4", user)
    assert m["status"] == "failed" and m["error_message"] == "This file has no audio."
    assert not seen["copy"].exists()
    assert list(pipeline.UPLOAD_DIR.iterdir()) == []


def test_upload_rejects_bad_type_before_saving(setup):
    import io
    user, _, _ = setup
    with pytest.raises(AudioError, match="Unsupported file type"):
        pipeline.process_upload(io.BytesIO(b"x"), "notes.txt", user)
    assert store.list_meetings(user) == []


def test_progress_messages(setup, monkeypatch):
    user, original, _ = setup
    messages = []

    def fake_analyze(segments, on_progress=None):
        on_progress("Rate limited by Groq, waiting 3 s...")
        return RESULT

    monkeypatch.setattr(pipeline, "analyze", fake_analyze)
    pipeline.process_recording(original, user, on_progress=messages.append)
    assert "Rate limited by Groq, waiting 3 s..." in messages and messages[-1] == "Done."


def test_indexing_failure_keeps_status_and_retry_works(setup, monkeypatch):
    user, original, _ = setup
    messages = []

    def broken_embed(texts):
        raise index.EmbeddingError("The OpenAI quota is used up. Check billing on the OpenAI account.")

    monkeypatch.setattr(index, "embed_texts", broken_embed)
    m = pipeline.process_recording(original, user, on_progress=messages.append)
    assert m["status"] == "done"  # indexing failure does not fail the meeting
    assert m["id"] not in index.indexed_ids(user)
    assert any(msg.startswith("Indexing failed: The OpenAI quota") for msg in messages)

    monkeypatch.setattr(index, "embed_texts", lambda texts: __import__("numpy").ones((len(texts), 4)))
    assert pipeline.retry_index(user, m["id"]) == 1
    assert m["id"] in index.indexed_ids(user)
    assert "index" in store.get_meeting(user, m["id"])["timings"]


def test_unexpected_indexing_error_is_logged_not_raised(setup, monkeypatch):
    user, original, _ = setup
    monkeypatch.setattr(index, "embed_texts", lambda texts: 1 / 0)
    m = pipeline.process_recording(original, user)
    assert m["status"] == "done" and m["id"] not in index.indexed_ids(user)


def test_retry_index_is_scoped(setup):
    user, original, _ = setup
    m = pipeline.process_recording(original, user)
    store.create_user("bob", "h", "user")
    bob = store.get_user_by_username("bob")["id"]
    assert pipeline.retry_index(bob, m["id"]) is None
