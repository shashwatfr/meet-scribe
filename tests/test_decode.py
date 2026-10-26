"""Rebuilding segment text from Whisper tokens. No API calls.
Run by pytest (uv run pytest).
"""
from services.transcribe import FIRST_SPECIAL_TOKEN, _vocab, fix_segment_text

TIMESTAMP = 50365  # a Whisper timestamp token (<|0.02|>)


def ids(*pieces: str) -> list[int]:
    """Look up the token id of each text piece in the real vocabulary file."""
    by_bytes = {b: i for i, b in _vocab().items()}
    return [by_bytes[p.encode("utf-8")] for p in pieces]


def test_broken_text_is_rebuilt_from_tokens():
    tokens = [TIMESTAMP, *ids(" We", " don", "'t", " have", " it", "."), TIMESTAMP]
    seg = fix_segment_text({"text": " We don have it", "tokens": tokens})
    assert seg["text"] == "We don't have it."
    assert seg["text_groq"] == " We don have it"


def test_special_and_timestamp_ids_are_skipped():
    tokens = [FIRST_SPECIAL_TOKEN, TIMESTAMP, *ids(" Hi", "."), 51000, FIRST_SPECIAL_TOKEN + 1]
    assert fix_segment_text({"text": " Hi", "tokens": tokens})["text"] == "Hi."


def test_matching_segment_is_unchanged():
    seg = fix_segment_text({"text": " Okay, fine.", "tokens": ids(" Okay", ",", " fine", ".")})
    assert seg["text"] == seg["text_groq"].strip() == "Okay, fine."


def test_missing_tokens_falls_back():
    seg = fix_segment_text({"text": " Groq text"})
    assert seg["text"] == seg["text_groq"] == " Groq text"


def test_unknown_token_falls_back():
    seg = fix_segment_text({"text": " Groq text", "tokens": [-1]})  # not in the vocabulary
    assert seg["text"] == seg["text_groq"] == " Groq text"


def test_only_timestamps_falls_back():
    seg = fix_segment_text({"text": " Groq text", "tokens": [TIMESTAMP, TIMESTAMP]})
    assert seg["text"] == seg["text_groq"] == " Groq text"


def test_invalid_utf8_falls_back():
    # Some tokens are only part of a multi-byte character; one on its own is not valid UTF-8.
    partial = next(i for i, b in _vocab().items() if not _is_utf8(b))
    seg = fix_segment_text({"text": " Groq text", "tokens": [partial]})
    assert seg["text"] == seg["text_groq"] == " Groq text"


def _is_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False
