"""Tests for config.py: missing keys are named, and key values never appear in errors.
Run by pytest; no API calls.
"""

import pytest

import config

FAKE_GROQ = "gsk_fake_secret_value_123"


@pytest.fixture
def env(monkeypatch):
    # Stop load_dotenv from reading the real .env during tests.
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_missing_keys_lists_missing(env):
    assert config.missing_keys() == ["OPENAI_API_KEY"]


def test_load_config_raises_clear_error_without_values(env):
    with pytest.raises(RuntimeError) as err:
        config.load_config()
    assert "OPENAI_API_KEY" in str(err.value)
    assert FAKE_GROQ not in str(err.value)


def test_load_config_returns_keys_when_all_present(env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk_fake")
    assert config.load_config() == {"GROQ_API_KEY": FAKE_GROQ, "OPENAI_API_KEY": "sk_fake"}
