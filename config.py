"""Reads the API keys (and the hosted demo passwords) from .env or Streamlit secrets.

get_value() is the only place settings and secrets are read: the environment / .env
first, then Streamlit secrets. missing_keys() is used by check_setup.py; load_config() by
services/transcribe.py, analyze.py and index.py. Errors name keys but never show values.
"""
import os

from dotenv import load_dotenv

REQUIRED_KEYS = ("GROQ_API_KEY", "OPENAI_API_KEY")


def _streamlit_secret(name: str) -> str | None:
    """A value from Streamlit secrets (.streamlit/secrets.toml, or the Cloud's Secrets settings)."""
    try:
        import streamlit as st  # imported here: the command-line tools usually don't need it

        value = st.secrets[name]
    except (FileNotFoundError, KeyError):  # no secrets file at all / no such secret
        return None
    return str(value) if value else None


def get_value(name: str) -> str | None:
    """A setting from the environment (.env), or else from Streamlit secrets; None if neither."""
    load_dotenv()
    return os.getenv(name) or _streamlit_secret(name)


def missing_keys() -> list[str]:
    """Return the names of required keys that are missing or empty."""
    return [name for name in REQUIRED_KEYS if not get_value(name)]


def load_config() -> dict[str, str]:
    """Return {key name: value}, or raise if any key is missing."""
    missing = missing_keys()
    if missing:
        raise RuntimeError(f"Missing in .env or Streamlit secrets: {', '.join(missing)}")
    return {name: get_value(name) for name in REQUIRED_KEYS}
