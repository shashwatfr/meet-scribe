"""Runs the whole job for one recording: copy, transcribe, save, index, analyse, save,
updating the meeting's status at each step.

Called by the Upload page (process_upload) and process_cli.py (process_recording).

Status flow:  transcribing -> transcribed -> done
              transcribing -> failed            (transcription failed)
              transcribed  -> analysis_failed   (transcript kept; retry_analysis() can redo it)
"""
import logging
import shutil
import time
import uuid
from pathlib import Path

from services import store
from services.analyze import AnalyzeError, analyze
from services.audio import ALLOWED_EXTENSIONS, AudioError, prepare_chunks
from services.index import EmbeddingError, index_meeting
from services.transcribe import MODEL as WHISPER_MODEL, TranscribeError, transcribe

log = logging.getLogger("meetscribe")

UPLOAD_DIR = Path("uploads")  # git-ignored; holds temporary working copies only
UNEXPECTED_TRANSCRIBE = "Unexpected error during transcription. Please try again."
UNEXPECTED_ANALYZE = "Unexpected error during analysis. The transcript is saved; retry the analysis."


class PipelineError(Exception):
    """An unexpected error after a meeting was created. The meeting is already marked failed."""

    def __init__(self, meeting_id: int):
        super().__init__(f"Unexpected error while processing meeting {meeting_id}")
        self.meeting_id = meeting_id


def _quiet(message: str) -> None:
    pass


def _check_name(filename: str) -> str:
    """Return the lowercase extension, or raise AudioError if the type is not allowed."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise AudioError(f"Unsupported file type: {ext or 'no extension'}. "
                         f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    return ext


def _transcribe_file(path: Path, on_progress=_quiet) -> tuple[list[dict], float, dict, str | None]:
    """Returns (segments, duration in seconds, {"extract": s, "transcribe": s}, language)."""
    on_progress("Extracting the audio...")
    t0 = time.perf_counter()
    with prepare_chunks(path) as chunks:
        t1 = time.perf_counter()
        on_progress(f"Transcribing ({len(chunks)} chunk(s))...")
        segments, language = transcribe(chunks)
        t2 = time.perf_counter()
        duration = sum(d for _, d in chunks)
    timings = {"extract": round(t1 - t0, 1), "transcribe": round(t2 - t1, 1)}
    return segments, duration, timings, language


def _analyze_and_save(user_id: int, meeting_id: int, segments: list[dict],
                      on_progress=_quiet) -> None:
    try:
        result = analyze(segments, on_progress=on_progress)
    except AnalyzeError as err:
        store.update_status(user_id, meeting_id, "analysis_failed", str(err))
        return
    except Exception as err:
        store.update_status(user_id, meeting_id, "analysis_failed", UNEXPECTED_ANALYZE)
        raise PipelineError(meeting_id) from err
    seconds = {s["step"]: s["seconds"] for s in result["steps"]}
    timings = {"summary": round(seconds.get("summary", 0) + seconds.get("merge summaries", 0), 1),
               "extraction": seconds.get("extraction", 0)}
    store.save_analysis(user_id, meeting_id, result, timings)


def retry_index(user_id: int, meeting_id: int) -> int | None:
    """(Re)build the Q&A index of one meeting. Returns the chunk count, or None if not found.

    Raises EmbeddingError with a friendly message if OpenAI fails.
    """
    t0 = time.perf_counter()
    count = index_meeting(user_id, meeting_id)
    if count is not None:
        store.update_timings(user_id, meeting_id, {"index": round(time.perf_counter() - t0, 1)})
    return count


def _index_quietly(user_id: int, meeting_id: int, on_progress) -> None:
    """Index for Q&A. A failure never changes the meeting's status; it just stays unindexed."""
    on_progress("Indexing for questions...")
    try:
        retry_index(user_id, meeting_id)
    except EmbeddingError as err:
        on_progress(f"Indexing failed: {err} You can retry it from your meetings.")
    except Exception:
        log.exception("Indexing failed for meeting %s", meeting_id)
        on_progress("Indexing failed. You can retry it from your meetings.")


def _run(user_id: int, meeting_id: int, write_copy, copy: Path, on_progress) -> dict:
    """Write the working copy, transcribe it, delete it, then analyse.

    The copy is deleted right after transcription, whether it worked or not.
    """
    try:
        write_copy(copy)
        segments, duration, timings, language = _transcribe_file(copy, on_progress)
    except (AudioError, TranscribeError) as err:
        store.update_status(user_id, meeting_id, "failed", str(err))
        return store.get_meeting(user_id, meeting_id)
    except Exception as err:
        store.update_status(user_id, meeting_id, "failed", UNEXPECTED_TRANSCRIBE)
        raise PipelineError(meeting_id) from err
    finally:
        copy.unlink(missing_ok=True)

    store.save_transcript(user_id, meeting_id, segments, WHISPER_MODEL, duration,
                          language=language, timings=timings)
    _index_quietly(user_id, meeting_id, on_progress)  # before analysis, so Q&A works even if it fails
    _analyze_and_save(user_id, meeting_id, segments, on_progress)
    on_progress("Done.")
    return store.get_meeting(user_id, meeting_id)


def _new_copy_path(ext: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"  # random name; the user's filename is never a path


def process_recording(path, user_id: int, title: str | None = None,
                      on_progress=_quiet) -> dict:
    """Run the pipeline on a file on disk (CLI). Works on a copy; the file itself is only read.

    Bad input (missing file, wrong type) raises AudioError before anything is saved.
    An unexpected error raises PipelineError after marking the meeting failed.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"File not found: {path.name}")
    ext = _check_name(path.name)
    meeting_id = store.create_meeting(user_id, title or path.stem, path.name)
    return _run(user_id, meeting_id, lambda copy: shutil.copyfile(path, copy),
                _new_copy_path(ext), on_progress)


def process_upload(fileobj, filename: str, user_id: int, title: str | None = None,
                   on_progress=_quiet) -> dict:
    """Run the pipeline on an uploaded file object (UI).

    The upload is written once to uploads/ and that file is the working copy,
    deleted right after transcription. filename is only used for display.
    """
    ext = _check_name(filename)
    meeting_id = store.create_meeting(user_id, title or Path(filename).stem, Path(filename).name)

    def write_copy(copy: Path) -> None:
        fileobj.seek(0)
        with open(copy, "wb") as out:
            shutil.copyfileobj(fileobj, out)

    return _run(user_id, meeting_id, write_copy, _new_copy_path(ext), on_progress)


def retry_analysis(user_id: int, meeting_id: int, on_progress=_quiet) -> dict | None:
    """Re-run only the analysis from the saved transcript (no new transcription).

    Returns the updated meeting, or None if not found / not this user's / no transcript.
    """
    segments = store.get_segments(user_id, meeting_id)
    if segments is None:
        return None
    store.update_status(user_id, meeting_id, "transcribed")  # clears the old error
    _analyze_and_save(user_id, meeting_id, segments, on_progress)
    on_progress("Done.")
    return store.get_meeting(user_id, meeting_id)
