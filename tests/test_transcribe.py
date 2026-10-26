"""Timestamp stitching and rate-limit messages. No API calls.
Run by pytest (uv run pytest).
"""
from types import SimpleNamespace

import pytest

from services.transcribe import _rate_limit_message, offset_segments


def seg(start, end, text="x", **extra):
    return {"id": 0, "start": start, "end": end, "text": text, **extra}


def test_offsets_run_continuously_across_chunks():
    merged = offset_segments([
        (120.0, [seg(0, 5), seg(115, 120)]),
        (120.0, [seg(5, 10)]),
        (60.0, [seg(30, 40)]),
    ])
    assert [(s["start"], s["end"]) for s in merged] == [(0, 5), (115, 120), (125, 130), (270, 280)]
    assert [s["id"] for s in merged] == [0, 1, 2, 3]


def test_offsets_use_measured_not_nominal_durations():
    merged = offset_segments([(119.98, [seg(0, 1)]), (120.04, [seg(0, 1)]), (10.0, [seg(2, 3)])])
    assert merged[1]["start"] == pytest.approx(119.98)
    assert merged[2]["start"] == pytest.approx(119.98 + 120.04 + 2)


def test_extra_segment_fields_are_kept():
    merged = offset_segments([(60.0, [seg(1, 2, avg_logprob=-0.2, no_speech_prob=0.01)])])
    assert merged[0]["avg_logprob"] == -0.2
    assert merged[0]["no_speech_prob"] == 0.01


def test_transcribe_returns_detected_language(monkeypatch):
    from services import transcribe as t
    replies = iter([([seg(0, 5)], "English"), ([seg(1, 2)], "English")])
    monkeypatch.setattr(t, "_client", lambda: None)
    monkeypatch.setattr(t, "transcribe_chunk", lambda client, path, language: next(replies))
    segments, language = t.transcribe([("c0.mp3", 60.0), ("c1.mp3", 30.0)])
    assert language == "English"
    assert [s["start"] for s in segments] == [0, 61]


def fake_429(retry_after):
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return SimpleNamespace(response=SimpleNamespace(headers=headers))


@pytest.mark.parametrize("retry_after, expected", [
    ("20", "in about 20 seconds"),
    ("600", "in about 10 minutes"),
    (None, "in a few minutes"),
    ("not-a-number", "in a few minutes"),
])
def test_rate_limit_message(retry_after, expected):
    message = _rate_limit_message(fake_429(retry_after))
    assert "rate limit" in message and expected in message
