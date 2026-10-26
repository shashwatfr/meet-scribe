"""Prepares a recording for Whisper: checks it, extracts the audio with ffmpeg, and
makes small mono MP3 chunk(s). Temp files are deleted automatically.

Called by services/pipeline.py and transcribe_cli.py.
"""
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

# Single source of truth: the Streamlit uploader will reuse this list.
ALLOWED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".mkv", ".mov")
MIN_SECONDS = 1.0
# Groq's free-tier upload limit is 25 MB; 20 MB leaves headroom. At 32 kbps that is ~87 minutes.
SINGLE_CHUNK_MAX_BYTES = 20 * 1024 * 1024
AUTO_CHUNK_SECONDS = 30 * 60


class AudioError(Exception):
    """A problem with the input file. The message is safe to show the user."""


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")


def _duration(path: Path) -> float:
    """Length of a media file in seconds, read with ffprobe."""
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(path)])
    return float(out.stdout.strip())


def _check_input(path: Path) -> None:
    """Raise AudioError if the file type is not allowed, unreadable, or has no audio."""
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise AudioError(f"Unsupported file type: {ext or 'no extension'}. "
                         f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    # List the audio streams. ffprobe fails on unreadable files, prints nothing if there is no audio.
    probe = _run(["ffprobe", "-v", "error", "-select_streams", "a",
                  "-show_entries", "stream=index", "-of", "csv=p=0", str(path)])
    if probe.returncode != 0:
        raise AudioError(f"Could not read this {ext} file. It may be damaged or not a real {ext} file.")
    if not probe.stdout.strip():
        raise AudioError("This file has no audio.")


@contextmanager
def prepare_chunks(path, chunk_seconds: float | None = None):
    """Yield [(chunk_path, duration_seconds), ...] in order.

    By default the whole recording is one chunk if its MP3 is under 20 MB,
    otherwise it is split into 30-minute chunks. Passing chunk_seconds forces
    splitting at that length (used to test chunking on short files).

    Audio and video inputs take the same path: ffmpeg keeps only the first
    audio stream. All temp files are deleted when the `with` block ends.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"File not found: {path}")
    _check_input(path)

    with tempfile.TemporaryDirectory(prefix="meetscribe_") as tmp:
        tmp = Path(tmp)
        audio = tmp / "audio.mp3"
        # Mono 16 kHz is what Whisper uses internally; 32 kbps keeps a 10 min chunk ~2.4 MB.
        result = _run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:a:0",
                       "-ac", "1", "-ar", "16000", "-b:a", "32k", str(audio)])
        if result.returncode != 0:
            raise AudioError(f"Could not read the audio in this {path.suffix.lower()} file.")
        duration = _duration(audio)
        if duration < MIN_SECONDS:
            raise AudioError("The audio is shorter than 1 second.")

        if chunk_seconds is None:
            if audio.stat().st_size < SINGLE_CHUNK_MAX_BYTES:
                yield [(audio, duration)]  # one chunk = no boundaries to cut words at
                return
            chunk_seconds = AUTO_CHUNK_SECONDS

        # shortcut: chunks are cut with no overlap, so a word on a boundary can be garbled or
        # dropped. By default this only happens for files over ~80 minutes (or when
        # --chunk-minutes forces splitting); add ~2 s overlap plus de-duplication if it matters.
        result = _run(["ffmpeg", "-v", "error", "-i", str(audio), "-f", "segment",
                       "-segment_time", str(chunk_seconds), "-c", "copy",
                       str(tmp / "chunk_%03d.mp3")])
        if result.returncode != 0:
            raise AudioError("Could not split the audio into chunks.")

        # Use each chunk's measured length so timestamp offsets never drift.
        yield [(chunk, _duration(chunk)) for chunk in sorted(tmp.glob("chunk_*.mp3"))]
