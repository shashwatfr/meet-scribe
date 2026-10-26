"""Shared test setup that pytest loads automatically for every test.

It blocks real network access, replaces OpenAI embeddings with a fake, and offers a
temporary database (temp_db) so no test touches the real one.
"""

import socket
import zlib

import numpy as np
import pytest

import config
from services import db, index

FAKE_DIMENSIONS = 16
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic stand-in for OpenAI: a bag-of-words vector (same words -> similar vectors)."""
    vectors = np.zeros((len(texts), FAKE_DIMENSIONS), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.lower().split():
            vectors[row, zlib.crc32(word.strip(".,?!").encode()) % FAKE_DIMENSIONS] += 1
    return vectors + 1e-3  # never an all-zero vector


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that opens a network connection or looks up a host (localhost is fine).

    Libraries often turn connection errors into their own errors, which our code turns into
    friendly messages, so the test could still pass. Every attempt is therefore recorded,
    and the test fails at the end if there was one.
    """
    attempts = []
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def is_local(address) -> bool:
        return not isinstance(address, tuple) or address[0] in LOCAL_HOSTS  # non-tuple: Unix socket

    def connect(self, address):
        if not is_local(address):
            attempts.append(address)
            raise OSError(f"network blocked in tests: {address}")
        return real_connect(self, address)

    def connect_ex(self, address):
        if not is_local(address):
            attempts.append(address)
            raise OSError(f"network blocked in tests: {address}")
        return real_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if host not in LOCAL_HOSTS and host is not None:
            attempts.append(host)
            raise socket.gaierror(f"network blocked in tests: {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    yield attempts
    if attempts:
        pytest.fail(f"Test tried to reach the network: {attempts}")


@pytest.fixture(autouse=True)
def no_real_settings(monkeypatch):
    """Tests never read the real .env or a real Streamlit secrets file; they set what they need."""
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.setattr(config, "_streamlit_secret", lambda name: None)
    for name in ("DEMO_PASSWORD_ADMIN", "DEMO_PASSWORD_ALICE", "DEMO_PASSWORD_BOB"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_real_embeddings(monkeypatch):
    """No test may call OpenAI. Tests that need specific vectors patch embed_texts again."""
    monkeypatch.setattr(index, "embed_texts", fake_embed)
    import services.qa as qa  # imported here so the patch also covers qa's own reference
    monkeypatch.setattr(qa, "embed_texts", fake_embed)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the app at an empty database file in a temp folder."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    return tmp_path / "test.db"
