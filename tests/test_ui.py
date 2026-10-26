"""Streamlit pages via AppTest. Temp database, fake analysis, no API calls.
Run by pytest (uv run pytest).
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import admin, index, pipeline, store
from services.auth import hash_password
from ui import md

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["pages/1_Upload.py", "pages/2_History.py", "pages/3_Admin.py"]
SEGMENTS = [{"id": i, "start": i * 10.0, "end": i * 10.0 + 9, "text": f"Line {i}."} for i in range(8)]
RESULT = {"model": "m",
          "summary": {"overview": "They want $18,000 for the year, up from $15,000.", "key_topics": ["price"]},
          "extraction": {"decisions": [{"topic": "price", "text": "Cap is $16,000 (was $18,000).",
                                        "timestamp": 40}],
                         "action_items": [{"task": "Email the vendor", "owner": None,
                                           "due_date": "Nov 8", "timestamp": 50}],
                         "open_topics": []},
          "steps": [{"step": "summary", "seconds": 1.0}]}


def page(path: str, user: dict | None = None) -> AppTest:
    at = AppTest.from_file(str(ROOT / path), default_timeout=15)
    if user is not None:
        at.session_state["user"] = user
    return at.run()


@pytest.fixture
def people(temp_db):
    store.create_user("alice", hash_password("alice-pw"), "user")
    store.create_user("bob", hash_password("bob-pw"), "user")
    store.create_user("admin", hash_password("admin-pw"), "admin")
    return {name: {k: v for k, v in store.get_user_by_username(name).items() if k != "password_hash"}
            for name in ("alice", "bob", "admin")}


def add_meeting(user: dict, title: str, status: str) -> int:
    mid = store.create_meeting(user["id"], title, f"{title}.mp4")
    if status in ("transcribed", "done", "analysis_failed"):
        store.save_transcript(user["id"], mid, SEGMENTS, "whisper", 80.0, language="English")
    if status == "done":
        store.save_analysis(user["id"], mid, RESULT)
    elif status in ("failed", "analysis_failed"):
        store.update_status(user["id"], mid, status, f"{status} because of a test")
    return mid


def all_text(at: AppTest) -> str:
    parts = [e.value for kind in ("markdown", "caption", "error", "warning", "info", "title",
                                  "subheader") for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


# ---------- signed out ----------

def test_main_page_shows_login_form(temp_db):
    at = page("app.py")
    assert [t.label for t in at.text_input] == ["Username", "Password"]
    assert at.text_input[1].proto.type == at.text_input[1].proto.PASSWORD


@pytest.mark.parametrize("path", PAGES)
def test_every_page_stops_when_signed_out(temp_db, path):
    at = page(path)
    assert "Please log in" in at.warning[0].value
    assert len(at.title) == 0 and len(at.button) == 0 and len(at.dataframe) == 0


# ---------- login ----------

def login(at: AppTest, username: str, password: str) -> AppTest:
    at.text_input[0].input(username)
    at.text_input[1].input(password)
    return at.button[0].click().run()


def test_login_right_wrong_unknown(people):
    at = login(page("app.py"), "alice", "alice-pw")
    assert at.session_state["user"]["username"] == "alice"
    assert "password_hash" not in at.session_state["user"]

    for username, password in (("alice", "wrong"), ("mallory", "alice-pw")):
        at = login(page("app.py"), username, password)
        assert [e.value for e in at.error] == ["Wrong username or password."]
        assert "user" not in at.session_state


def test_logout_clears_session(people):
    at = page("app.py", people["alice"])
    [logout] = [b for b in at.button if b.label == "Log out"]
    at = logout.click().run()
    assert "user" not in at.session_state
    assert [t.label for t in at.text_input] == ["Username", "Password"]


# ---------- admin ----------

def test_normal_user_is_denied_admin_page(people):
    add_meeting(people["bob"], "Bob secret", "done")
    at = page("pages/3_Admin.py", people["alice"])
    assert [e.value for e in at.error] == ["Admins only."]
    assert len(at.dataframe) == 0 and "Bob secret" not in all_text(at)


def test_admin_functions_raise_for_normal_user(people):
    alice = people["alice"]["id"]
    for call in (lambda: admin.require_admin(alice), lambda: admin.row_counts(alice),
                 lambda: admin.db_file_size(alice), lambda: admin.table_rows(alice, "users"),
                 lambda: admin.require_admin(99999)):
        with pytest.raises(PermissionError):
            call()


def test_admin_sees_all_tables_without_hashes(people):
    add_meeting(people["bob"], "Bob secret", "done")
    at = page("pages/3_Admin.py", people["admin"])
    assert len(at.error) == 0
    assert len(at.dataframe) == 4
    users = at.dataframe[0].value
    assert set(users["password_hash"]) == {"(hidden)"}
    for frame in at.dataframe:
        assert "$2b$" not in frame.value.to_string()  # no bcrypt hash anywhere
    assert "bob" in at.dataframe[1].value["username"].tolist()


def test_admin_cuts_long_cells(people):
    long_segments = [{"id": 0, "start": 0, "end": 1, "text": "x" * 5000}]
    mid = store.create_meeting(people["alice"]["id"], "Long", "l.mp4")
    store.save_transcript(people["alice"]["id"], mid, long_segments, "whisper", 1.0)
    [row] = admin.table_rows(people["admin"]["id"], "transcripts")
    assert len(row["segments_json"]) < 250 and row["segments_json"].endswith(" chars)")


# ---------- history ----------

def test_history_shows_only_own_meetings(people):
    add_meeting(people["alice"], "Alice standup", "done")
    add_meeting(people["bob"], "Bob secret", "done")
    at = page("pages/2_History.py", people["alice"])
    options = at.selectbox[0].options
    assert any("Alice standup" in o for o in options)
    assert not any("Bob secret" in o for o in options)
    assert "Bob secret" not in all_text(at)


def test_retry_button_only_for_analysis_failed(people):
    ids = {status: add_meeting(people["alice"], f"m-{status}", status)
           for status in ("done", "failed", "analysis_failed")}
    at = page("pages/2_History.py", people["alice"])
    for status, mid in ids.items():
        at = at.selectbox[0].set_value(mid).run()
        has_retry = any(b.label == "Retry analysis" for b in at.button)
        assert has_retry == (status == "analysis_failed"), status


def test_word_download_only_with_a_transcript(people):
    ids = {status: add_meeting(people["alice"], f"m-{status}", status)
           for status in ("done", "failed", "analysis_failed")}
    at = page("pages/2_History.py", people["alice"])
    for status, mid in ids.items():
        at = at.selectbox[0].set_value(mid).run()
        buttons = at.get("download_button")
        assert len(buttons) == (0 if status == "failed" else 1), status
        if buttons:
            assert buttons[0].label == "Download as Word"


def test_retry_click_runs_analysis(people, monkeypatch):
    monkeypatch.setattr(pipeline, "analyze", lambda segments, on_progress=None: RESULT)
    mid = add_meeting(people["alice"], "Retry me", "analysis_failed")
    at = page("pages/2_History.py", people["alice"])
    at = at.selectbox[0].set_value(mid).run()
    [retry] = [b for b in at.button if b.label == "Retry analysis"]
    retry.click().run()
    assert store.get_meeting(people["alice"]["id"], mid)["status"] == "done"


def test_dollar_amounts_are_escaped(people):
    add_meeting(people["alice"], "Vendor", "done")
    at = page("pages/2_History.py", people["alice"])
    text = all_text(at)
    assert r"\$18,000 for the year, up from \$15,000" in text  # no LaTeX between the two $
    assert r"Cap is \$16,000" in text
    assert md("$18,000 ... $15,000") == r"\$18,000 \.\.\. \$15,000"


def test_history_shows_sources_language_and_caption(people):
    add_meeting(people["alice"], "Vendor", "done")
    at = page("pages/2_History.py", people["alice"])
    text = all_text(at)
    assert "Owner is shown only when the conversation names them." in text
    assert "language English" in text
    assert any(e.label == "Show the transcript lines around this" for e in at.expander)
    assert r"Line 4\." in text  # decision at 40 s: the source line is shown


def test_delete_needs_confirmation(people):
    mid = add_meeting(people["alice"], "Doomed", "done")
    at = page("pages/2_History.py", people["alice"])
    at = [b for b in at.button if b.label == "Delete meeting"][0].click().run()
    assert store.get_meeting(people["alice"]["id"], mid) is not None  # not yet
    [yes] = [b for b in at.button if b.label == "Yes, delete"]
    yes.click().run()
    assert store.get_meeting(people["alice"]["id"], mid) is None


# ---------- Q&A ----------

def test_ask_tab_offers_retry_indexing_when_not_indexed(people, monkeypatch):
    mid = add_meeting(people["alice"], "Vendor", "done")
    at = page("pages/2_History.py", people["alice"])
    assert "This meeting is not indexed for questions yet." in all_text(at)
    [retry] = [b for b in at.button if b.label == "Retry indexing"]
    at = retry.click().run()
    assert mid in index.indexed_ids(people["alice"]["id"])
    assert not any(b.label == "Retry indexing" for b in at.button)
    assert r"indexed for questions\: yes" in all_text(at)  # ":" is escaped for markdown


def test_ask_shows_answer_with_cited_time(people, monkeypatch):
    import json
    from services import analyze
    mid = add_meeting(people["alice"], "Vendor", "done")
    index.index_meeting(people["alice"]["id"], mid)

    def fake_llm(messages, on_progress=None, **kwargs):
        first_chunk = int(messages[1]["content"].split("[chunk ")[1].split(" ")[0])
        return json.dumps({"answerable": True, "answer": "The cap is $16,000 (was $18,000).",
                           "citations": [first_chunk]}), {"calls": 1}

    monkeypatch.setattr(analyze, "call_llm", fake_llm)
    at = page("pages/2_History.py", people["alice"])
    at = at.chat_input[0].set_value("What is the cap?").run()
    text = all_text(at)
    assert r"The cap is \$16,000 \(was \$18,000\)\." in text  # escaped, so no LaTeX
    assert "`[00:00]` Vendor" in text
    assert any(e.label == "Show the cited lines" for e in at.expander)


def test_admin_shows_embedding_size_not_vector(people):
    mid = add_meeting(people["alice"], "Vendor", "done")
    index.index_meeting(people["alice"]["id"], mid)
    [row, *_] = admin.table_rows(people["admin"]["id"], "chunks")
    assert row["embedding"] == f"(64 bytes, {index.EMBEDDING_MODEL})"  # 16 fake dims x 4 bytes
    assert row["username"] == "alice"
