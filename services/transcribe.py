"""Sends audio chunks to Groq Whisper, repairs each segment's text from its tokens, and
joins the chunks' timestamps into one timeline.

Called by services/pipeline.py and transcribe_cli.py.
"""
import base64
import functools
import math
from pathlib import Path

import groq

from config import load_config

MODEL = "whisper-large-v3"  # more accurate than -turbo (WER 10.3% vs 12%), ~13% slower
TIMEOUT_SECONDS = 300  # a single ~80-minute chunk can take a while to upload and process
MAX_RETRIES = 2
VOCAB_PATH = Path(__file__).parent / "assets" / "multilingual.tiktoken"
FIRST_SPECIAL_TOKEN = 50257  # ids from here up are Whisper control/timestamp tokens, not text


class TranscribeError(Exception):
    """A transcription problem. The message is safe to show the user."""


def _client() -> groq.Groq:
    # The SDK itself retries 429s (waiting for retry-after) and timeouts, MAX_RETRIES times.
    return groq.Groq(api_key=load_config()["GROQ_API_KEY"],
                     timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)


def _rate_limit_message(err) -> str:
    """Turn a 429 into a friendly message, using Groq's retry-after header if present."""
    try:
        wait = float(err.response.headers.get("retry-after"))
    except (TypeError, ValueError):
        wait = None
    if wait is None:
        when = "in a few minutes"
    elif wait < 90:
        when = f"in about {math.ceil(wait)} seconds"
    else:
        when = f"in about {math.ceil(wait / 60)} minutes"
    return f"Groq rate limit hit (still limited after {MAX_RETRIES} retries). Try again {when}."


@functools.cache
def _vocab() -> dict[int, bytes]:
    """Whisper's token table, id -> bytes. Each line of the file is '<base64 bytes> <id>'."""
    vocab = {}
    with open(VOCAB_PATH, encoding="ascii") as f:
        for line in f:
            if line.strip():
                token, token_id = line.split()
                vocab[int(token_id)] = base64.b64decode(token)
    return vocab


def decode_tokens(tokens: list[int]) -> str:
    """Turn Whisper token ids back into text, skipping control and timestamp tokens."""
    vocab = _vocab()
    raw = b"".join(vocab[t] for t in tokens if t < FIRST_SPECIAL_TOKEN)
    return raw.decode("utf-8").strip()


def fix_segment_text(seg: dict) -> dict:
    """Rebuild seg["text"] from its tokens; keep Groq's string as seg["text_groq"].

    Groq's `text` sometimes drops punctuation, apostrophes and parts of numbers
    ("we don have" for "we don't have") while the tokens are correct. Seen in
    about 1 in 25 segments on whisper-large-v3. Never raises: if decoding fails
    or gives nothing, Groq's text is kept.
    """
    seg["text_groq"] = seg.get("text", "")
    try:
        decoded = decode_tokens(seg["tokens"])
    except Exception:  # missing/invalid tokens, unreadable vocab file, bad UTF-8
        decoded = ""
    if decoded:
        seg["text"] = decoded
    return seg


def transcribe_chunk(client: groq.Groq, path,
                     language: str | None = None) -> tuple[list[dict], str | None]:
    """Transcribe one chunk.

    Returns (segments with times relative to the chunk, detected language such as "English").
    """
    extra = {"language": language} if language else {}  # omitted = auto-detect
    try:
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=f, model=MODEL, response_format="verbose_json", temperature=0, **extra)
    except groq.RateLimitError as err:
        raise TranscribeError(_rate_limit_message(err)) from err
    except groq.APIConnectionError as err:  # also covers timeouts
        raise TranscribeError(f"Could not reach Groq (no answer within {TIMEOUT_SECONDS} s "
                              "or no connection). Try again.") from err
    # The SDK model only declares `text`; verbose_json's other fields arrive as extras.
    extra_fields = result.model_extra
    segments = [fix_segment_text(seg) for seg in extra_fields["segments"]]
    return segments, extra_fields.get("language")


def offset_segments(chunk_results: list[tuple[float, list[dict]]]) -> list[dict]:
    """Merge per-chunk segments into one list timed from the start of the recording.

    chunk_results is [(chunk_duration, segments), ...] in order. Each chunk is
    shifted by the total length of the chunks before it. All other segment
    fields are kept as-is (note `seek` stays relative to its chunk).
    """
    merged, offset = [], 0.0
    for duration, segments in chunk_results:
        for seg in segments:
            merged.append({**seg, "id": len(merged),
                           "start": seg["start"] + offset, "end": seg["end"] + offset})
        offset += duration
    return merged


def transcribe(chunks: list[tuple], language: str | None = None) -> tuple[list[dict], str | None]:
    """Transcribe [(chunk_path, duration), ...] in order.

    Returns (continuous segments, detected language of the first chunk).
    """
    client = _client()
    results, detected = [], None
    for path, duration in chunks:
        segments, chunk_language = transcribe_chunk(client, path, language)
        results.append((duration, segments))
        detected = detected or chunk_language
    return offset_segments(results), detected
