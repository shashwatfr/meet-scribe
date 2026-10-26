"""The test-run scripts stop on Groq's daily limit, and --effort switches the Q&A setting.

Everything runs on temp files and fakes: no API calls, no real data folders.
Run by pytest (uv run pytest).
"""
import json
import sys
from types import SimpleNamespace

import httpx
import groq
import pytest

import run_qa_tests
import run_script_tests
import services.qa as qa
from services import analyze
from services.analyze import DailyLimitError

SEGMENTS = [{"id": 0, "start": 0.0, "end": 5.0, "text": "We agree on the launch date."}]


def daily_limit(*args, **kwargs):
    raise DailyLimitError("Groq rate limit hit (asked to wait 3600 s).")


def test_call_llm_raises_daily_limit_on_a_long_retry_after(monkeypatch):
    response = httpx.Response(429, headers={"retry-after": "3600"},
                              request=httpx.Request("POST", "https://api.groq.test"))
    error = groq.RateLimitError("limit", response=response, body=None)

    def create(**kwargs):
        raise error

    monkeypatch.setattr(analyze, "_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(analyze.time, "sleep", lambda s: pytest.fail("must not wait for an hour"))
    with pytest.raises(DailyLimitError, match="3600 s"):
        analyze.call_llm([{"role": "user", "content": "hi"}])
    assert issubclass(DailyLimitError, analyze.AnalyzeError)  # old handlers still catch it


@pytest.fixture
def script_files(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for x in "ab":
        (scripts / f"script_{x}_key.md").write_text(f"# key {x}\nSpeakers: A and B\n", encoding="utf-8")
        (tmp_path / f"script_{x}.json").write_text(json.dumps({"segments": SEGMENTS}), encoding="utf-8")
    monkeypatch.setattr(run_script_tests, "SCRIPTS_DIR", scripts)
    monkeypatch.setattr(run_script_tests, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(run_script_tests, "ERRORS_FILE", tmp_path / "errors.md")
    monkeypatch.setattr(run_script_tests, "INPUTS", {"real": str(tmp_path / "{name}.json")})
    return tmp_path


def test_run_script_tests_stops_on_daily_limit(script_files, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_script_tests, "analyze", lambda segs: calls.append(1) or daily_limit())
    monkeypatch.setattr(sys, "argv", ["run_script_tests.py", "--scripts", "a,b", "--inputs", "real",
                                      "--refresh"])
    with pytest.raises(SystemExit) as stop:
        run_script_tests.main()
    out = capsys.readouterr().out
    assert stop.value.code == 2 and len(calls) == 1  # script_b was never tried
    assert "STOPPED at script_a [real]" in out and "Tokens used this run" in out
    assert not (script_files / "out" / "script_a_real.json").exists()


@pytest.fixture
def qa_files(tmp_path, temp_db, monkeypatch):
    for x in "abcd":
        (tmp_path / f"script_{x}.mp4.json").write_text(json.dumps(
            {"model": "whisper", "language": "English", "segments": SEGMENTS}), encoding="utf-8")
        (tmp_path / f"qa_{x}.md").write_text(
            "# qa\n## 1. Answerable (this meeting)\nQuestion: When?\nScope: this meeting\n"
            "## 2. Cross (all my meetings)\nQuestion: Where?\nScope: all my meetings\n",
            encoding="utf-8")
    monkeypatch.setattr(run_qa_tests, "QA_DB", temp_db)
    monkeypatch.setattr(run_qa_tests, "TRANSCRIPT", str(tmp_path / "{title}.mp4.json"))
    monkeypatch.setattr(run_qa_tests, "QUESTIONS", str(tmp_path / "qa_{x}.md"))
    monkeypatch.setattr(qa, "QA_REASONING_EFFORT", qa.QA_REASONING_EFFORT)  # restored afterwards


def test_run_qa_tests_stops_on_daily_limit(qa_files, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(analyze, "call_llm", lambda *a, **k: calls.append(1) or daily_limit())
    monkeypatch.setattr(sys, "argv", ["run_qa_tests.py", "--scripts", "a,b"])
    with pytest.raises(SystemExit) as stop:
        run_qa_tests.main()
    out = capsys.readouterr().out
    assert stop.value.code == 2 and len(calls) == 1  # no further questions were asked
    assert "STOPPED at script_a '1. Answerable (this meeting)'" in out


@pytest.mark.parametrize("flag, expected", [("default", None), ("low", "low")])
def test_effort_flag_reaches_the_llm_and_prints_timing(qa_files, monkeypatch, capsys, flag, expected):
    seen = []

    def fake_llm(messages, on_progress=None, reasoning_effort=None):
        seen.append(reasoning_effort)
        chunk = int(messages[1]["content"].split("[chunk ")[1].split(" ")[0])
        return json.dumps({"answerable": True, "answer": "Now.", "citations": [chunk]}), \
            {"calls": 1, "total_tokens": 120, "reasoning_tokens": 30}

    monkeypatch.setattr(analyze, "call_llm", fake_llm)
    monkeypatch.setattr(sys, "argv", ["run_qa_tests.py", "--scripts", "a", "--effort", flag])
    run_qa_tests.main()
    out = capsys.readouterr().out
    assert seen == [expected, expected]
    assert "tokens 120 (reasoning 30) in 1 call(s)" in out and "TIME" in out
    assert "LLM tokens used this run: 240 (reasoning 60) in 2 calls" in out
