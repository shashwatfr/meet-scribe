"""Hosting support: secrets fallback, hosted demo users, folders created at runtime.
Run by pytest; no API calls.
"""
import io

import pytest

import config
from services import auth, db, pipeline, store
from services.auth import ensure_demo_users, verify_login

REAL_SECRET_LOOKUP = config._streamlit_secret  # taken before conftest swaps it for each test
SECRETS = {"OPENAI_API_KEY": "secret-openai", "DEMO_PASSWORD_ADMIN": "a" * 32,
           "DEMO_PASSWORD_ALICE": "b" * 32, "DEMO_PASSWORD_BOB": "c" * 32}


@pytest.fixture
def fake_secrets(monkeypatch):
    """Pretend a Streamlit secrets file holds SECRETS (the environment has no keys)."""
    for name in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_streamlit_secret", lambda name: SECRETS.get(name))


# ---------- config: environment first, then Streamlit secrets ----------

def test_key_only_in_secrets_is_found(fake_secrets):
    assert config.get_value("OPENAI_API_KEY") == "secret-openai"
    assert config.missing_keys() == ["GROQ_API_KEY"]  # still named when missing everywhere


def test_environment_wins_over_secrets(fake_secrets, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert config.get_value("OPENAI_API_KEY") == "from-env"


def test_no_secrets_file_means_missing_not_a_crash(monkeypatch, tmp_path):
    """The real st.secrets lookup, in a folder with no secrets file: just None."""
    monkeypatch.setattr(config, "_streamlit_secret", REAL_SECRET_LOOKUP)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert config._streamlit_secret("GROQ_API_KEY") is None
    assert config.get_value("GROQ_API_KEY") is None


# ---------- hosted demo users ----------

def test_demo_users_created_from_secrets_once(temp_db, fake_secrets):
    assert ensure_demo_users() == ["admin", "alice", "bob"]
    assert verify_login("alice", "b" * 32)["role"] == "user"
    assert verify_login("admin", "a" * 32)["role"] == "admin"
    assert verify_login("alice", "alice-demo-2026") is None  # not the public local password
    hashes = {u: store.get_user_by_username(u)["password_hash"] for u in ("admin", "alice", "bob")}
    assert ensure_demo_users() == []  # second call: users exist, nothing changes
    assert {u: store.get_user_by_username(u)["password_hash"] for u in hashes} == hashes


def test_no_demo_users_without_all_three_secrets(temp_db, monkeypatch):
    monkeypatch.setattr(config, "_streamlit_secret", lambda name: None)
    assert ensure_demo_users() == [] and store.count_users() == 0
    partial = {"DEMO_PASSWORD_ADMIN": "x" * 32, "DEMO_PASSWORD_ALICE": "y" * 32}
    monkeypatch.setattr(config, "_streamlit_secret", lambda name: partial.get(name))
    assert ensure_demo_users() == [] and store.count_users() == 0


def test_existing_users_are_left_alone(temp_db, fake_secrets):
    store.create_user("carol", auth.hash_password("pw"), "user")
    assert ensure_demo_users() == []
    assert store.get_user_by_username("alice") is None


# ---------- folders that git does not carry ----------

def test_data_and_uploads_folders_are_created_at_runtime(tmp_path, monkeypatch):
    data_dir, uploads_dir = tmp_path / "fresh" / "data", tmp_path / "fresh" / "uploads"
    monkeypatch.setattr(db, "DB_PATH", data_dir / "meetscribe.db")
    monkeypatch.setattr(pipeline, "UPLOAD_DIR", uploads_dir)
    assert not data_dir.exists() and not uploads_dir.exists()

    store.create_user("alice", "h", "user")  # first database use
    assert (data_dir / "meetscribe.db").exists()

    user = store.get_user_by_username("alice")["id"]
    seen = []

    def fake_transcribe(path, on_progress=None):
        seen.append(path.parent.is_dir())  # the uploads folder existed for the copy
        return [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hi."}], 1.0, {}, None

    monkeypatch.setattr(pipeline, "_transcribe_file", fake_transcribe)
    monkeypatch.setattr(pipeline, "analyze", lambda segments, on_progress=None: {
        "model": "m", "summary": {"overview": "o", "key_topics": []},
        "extraction": {"decisions": [], "action_items": [], "open_topics": []}, "steps": []})
    meeting = pipeline.process_upload(io.BytesIO(b"x"), "call.mp3", user)
    assert seen == [True] and meeting["status"] == "done"
    assert uploads_dir.is_dir() and list(uploads_dir.iterdir()) == []  # created, copy deleted
