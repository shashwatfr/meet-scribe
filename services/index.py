"""Prepares meetings for questions: splits transcripts into chunks and stores an OpenAI
embedding (a list of numbers that captures the meaning) for each chunk.

Called by services/pipeline.py after transcription; services/qa.py searches the result.
"""
import numpy as np
import openai

from config import load_config
from services import store

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, $0.02 per 1M input tokens
BATCH_SIZE = 100  # texts per OpenAI request (the API allows far more)
TIMEOUT_SECONDS = 60
MIN_WORDS, MAX_WORDS = 150, 250


class EmbeddingError(Exception):
    """An embedding problem. The message is safe to show the user."""


# ---------- chunking ----------

def _word_count(segments: list[dict]) -> int:
    return sum(len(s["text"].split()) for s in segments)


def _chunk(segments: list[dict]) -> dict:
    return {"start": segments[0]["start"], "end": segments[-1]["end"],
            "text": " ".join(s["text"].strip() for s in segments)}


def make_chunks(segments: list[dict]) -> list[dict]:
    """Group whole segments into chunks of about 150-250 words, with one segment of overlap.

    A chunk closes when it has at least 150 words and the next segment would take it
    past 250. The next chunk starts with the previous chunk's last segment (overlap),
    unless that would push it past 250 on its own. A single segment longer than 250
    words becomes its own chunk (segments are never cut, so times stay exact).
    """
    segments = [s for s in segments if s["text"].strip()]
    chunks, current, new_count = [], [], 0  # new_count: segments not already in the last chunk
    for seg in segments:
        words = len(seg["text"].split())
        if new_count and _word_count(current) >= MIN_WORDS and _word_count(current) + words > MAX_WORDS:
            chunks.append(_chunk(current))
            overlap = [current[-1]] if len(current) > 1 else []
            # keep the overlap segment unless it would push the new chunk past MAX_WORDS
            current = overlap if _word_count(overlap) + words <= MAX_WORDS else []
            new_count = 0
        current.append(seg)
        new_count += 1
    if new_count:
        chunks.append(_chunk(current))
    return chunks


# ---------- embeddings ----------

def to_blob(vector) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Thin wrapper around OpenAI embeddings. Returns one float32 row per text."""
    client = openai.OpenAI(api_key=load_config()["OPENAI_API_KEY"], timeout=TIMEOUT_SECONDS,
                           max_retries=2)
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[i:i + BATCH_SIZE])
        except openai.AuthenticationError as err:
            raise EmbeddingError("OpenAI rejected the API key. Check OPENAI_API_KEY in .env.") from err
        except openai.RateLimitError as err:
            if "insufficient_quota" in str(err):
                raise EmbeddingError("The OpenAI quota is used up. Check billing on the "
                                     "OpenAI account.") from err
            raise EmbeddingError("OpenAI rate limit hit. Try again in a minute.") from err
        except openai.APIConnectionError as err:  # also covers timeouts
            raise EmbeddingError("Could not reach OpenAI. Check the connection and try again.") from err
        except openai.APIStatusError as err:
            raise EmbeddingError(f"OpenAI returned an error ({err.status_code}). "
                                 "Try again later.") from err
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return np.asarray(vectors, dtype=np.float32)


# ---------- indexing meetings ----------

def index_meeting(user_id: int, meeting_id: int) -> int | None:
    """Chunk, embed and store a meeting's transcript, replacing old chunks.

    Returns the number of chunks, or None if not found / not this user's / no transcript.
    Raises EmbeddingError if OpenAI fails (old chunks are then left as they were).
    """
    segments = store.get_segments(user_id, meeting_id)
    if segments is None:
        return None
    chunks = make_chunks(segments)
    if chunks:
        for chunk, vector in zip(chunks, embed_texts([c["text"] for c in chunks])):
            chunk["embedding"] = to_blob(vector)
    store.replace_chunks(user_id, meeting_id, chunks, EMBEDDING_MODEL)
    return len(chunks)


def indexed_ids(user_id: int) -> set[int]:
    """The user's meetings indexed with the CURRENT model (other models count as not indexed)."""
    return store.indexed_meeting_ids(user_id, EMBEDDING_MODEL)


def unindexed_ids(user_id: int) -> list[int]:
    """The user's meetings that have a transcript but no chunks for the current model."""
    done = indexed_ids(user_id)
    return [mid for mid in store.transcript_meeting_ids(user_id) if mid not in done]


def index_missing(user_id: int) -> list[tuple[int, str]]:
    """Index every meeting of this user that is not indexed yet. Returns [(id, outcome)]."""
    results = []
    for meeting_id in unindexed_ids(user_id):
        try:
            results.append((meeting_id, f"{index_meeting(user_id, meeting_id)} chunks"))
        except EmbeddingError as err:
            results.append((meeting_id, f"failed: {err}"))
    return results
