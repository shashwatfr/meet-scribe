"""The conftest network guard really blocks outside connections (and allows localhost).
Run by pytest (uv run pytest).
"""
import socket

import groq
import pytest

from services import analyze


def test_outside_connection_is_blocked_and_recorded(no_network):
    with pytest.raises(OSError):
        socket.create_connection(("203.0.113.10", 443), timeout=1)  # blocked at the lookup
    with pytest.raises(OSError):
        socket.getaddrinfo("api.groq.com", 443)
    raw = socket.socket()
    with pytest.raises(OSError):
        raw.connect(("203.0.113.10", 443))  # blocked at connect (no lookup needed)
    raw.close()
    assert no_network == ["203.0.113.10", "api.groq.com", ("203.0.113.10", 443)]
    no_network.clear()  # this test caused the attempts on purpose


def test_real_sdk_call_is_blocked(no_network):
    """An unfaked Groq call must not get out (the SDK turns it into a connection error)."""
    with pytest.raises((groq.APIConnectionError, analyze.AnalyzeError)):
        groq.Groq(api_key="test", max_retries=0, timeout=2).chat.completions.create(
            model="x", messages=[{"role": "user", "content": "hi"}])
    assert no_network  # the attempt was recorded, so a careless test would fail
    no_network.clear()


def test_localhost_is_allowed():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname(), timeout=2)
    client.close()
    server.close()
