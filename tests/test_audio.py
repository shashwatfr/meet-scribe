"""Edge cases for services/audio.py. Test files are generated with ffmpeg; no API calls.
Run by pytest (uv run pytest).
"""
import shutil
import subprocess

import pytest

from services import audio
from services.audio import AudioError, prepare_chunks

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def test_video_without_audio(tmp_path):
    video = tmp_path / "silent.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5", "-pix_fmt", "yuv420p", video)
    with pytest.raises(AudioError, match="This file has no audio."):
        with prepare_chunks(video):
            pass


def test_unreadable_file_names_type(tmp_path):
    fake = tmp_path / "broken.mp4"
    fake.write_bytes(b"this is not a video" * 100)
    with pytest.raises(AudioError, match=r"Could not read this \.mp4 file"):
        with prepare_chunks(fake):
            pass


def test_too_short(tmp_path):
    wav = tmp_path / "short.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=duration=0.5", wav)
    with pytest.raises(AudioError, match="shorter than 1 second"):
        with prepare_chunks(wav):
            pass


def test_unsupported_extension(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with pytest.raises(AudioError, match=r"Unsupported file type: \.txt"):
        with prepare_chunks(txt):
            pass


def test_small_file_is_one_chunk(tmp_path):
    wav = tmp_path / "tone.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=duration=5", wav)
    with prepare_chunks(wav) as chunks:
        assert len(chunks) == 1
        assert chunks[0][1] == pytest.approx(5, abs=0.2)
        path = chunks[0][0]
    assert not path.exists()  # temp file deleted


def test_large_file_is_split_automatically(tmp_path, monkeypatch):
    # Pretend every file is "large" and use 2 s chunks instead of 30 min.
    monkeypatch.setattr(audio, "SINGLE_CHUNK_MAX_BYTES", 1)
    monkeypatch.setattr(audio, "AUTO_CHUNK_SECONDS", 2)
    wav = tmp_path / "tone.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=duration=5", wav)
    with prepare_chunks(wav) as chunks:
        assert len(chunks) == 3


def test_forced_split_covers_whole_file_and_is_cleaned_up(tmp_path):
    wav = tmp_path / "tone.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=duration=5", wav)
    with prepare_chunks(wav, chunk_seconds=2) as chunks:
        paths = [path for path, _ in chunks]
        assert len(chunks) == 3
        assert sum(d for _, d in chunks) == pytest.approx(5, abs=0.2)
    assert not any(p.exists() for p in paths)  # temp files deleted


def test_video_with_audio_takes_same_path(tmp_path):
    video = tmp_path / "clip.mkv"
    ffmpeg("-f", "lavfi", "-i", "testsrc=duration=3:size=64x64:rate=5",
           "-f", "lavfi", "-i", "sine=duration=3", "-pix_fmt", "yuv420p", "-shortest", video)
    with prepare_chunks(video) as chunks:
        assert len(chunks) == 1
        assert chunks[0][1] == pytest.approx(3, abs=0.2)
