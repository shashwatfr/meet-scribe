"""Q&A check: asks the questions in data/scripts/qa_x.md and prints answers next to the expectations.

Run by hand to judge answer quality; it calls services/qa.py.

Usage: uv run python run_qa_tests.py [--scripts a,d] [--effort low|default]
       [--refresh] [--retrieve-only]

Uses its own database (data/qa_tests.db) with one user "qa" and the four real
transcripts from data/transcripts/, so the app's database is not touched and
Whisper is not called again. All four meetings are always loaded, so the
cross-meeting questions can find their answer. Nothing is scored automatically.

--effort         Q&A reasoning effort for this run: "low", or "default" (Groq's
                 default). Without it, QA_REASONING_EFFORT in services/qa.py is used.
--refresh        re-index the four meetings (OpenAI embeddings)
--retrieve-only  print the top chunks only, no LLM call (no Groq needed)
If Groq's daily limit is hit, the run stops and reports instead of going on.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import services.qa as qa
from services import db, store
from services.analyze import AnalyzeError, DailyLimitError
from services.auth import hash_password
from services.export import format_time
from services.index import EmbeddingError, index_meeting, indexed_ids

ALL_SCRIPTS = "abcd"
QA_USER = "qa"
QA_DB = Path("data/qa_tests.db")  # separate from the app's database
TRANSCRIPT = "data/transcripts/{title}.mp4.json"
QUESTIONS = "data/scripts/qa_{x}.md"


def parse_questions(path: Path) -> list[dict]:
    """Each '## ...' section with 'Key: value' lines becomes one question dict."""
    questions = []
    for section in re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.MULTILINE)[1:]:
        heading, *lines = section.strip().splitlines()
        fields = dict(re.match(r"([\w ]+):\s*(.*)", line).groups() for line in lines if ":" in line)
        questions.append({"heading": heading.strip(), **fields})
    return questions


def load_meetings(refresh: bool) -> tuple[int, dict[str, int]]:
    """Create the qa user and one meeting per real transcript (once); index as needed."""
    if store.get_user_by_username(QA_USER) is None:
        store.create_user(QA_USER, hash_password("qa-only-not-a-login"), "user")
    user_id = store.get_user_by_username(QA_USER)["id"]
    existing = {m["title"]: m["id"] for m in store.list_meetings(user_id)}
    ids = {}
    for x in ALL_SCRIPTS:
        title = f"script_{x}"
        if title not in existing:
            data = json.loads(Path(TRANSCRIPT.format(title=title)).read_text(encoding="utf-8"))
            mid = store.create_meeting(user_id, title, f"{title}.mp4")
            segments = data["segments"]
            store.save_transcript(user_id, mid, segments, data["model"], segments[-1]["end"],
                                  language=data.get("language"))
            store.update_status(user_id, mid, "done")
            existing[title] = mid
        ids[title] = existing[title]
    to_index = [mid for mid in ids.values() if refresh or mid not in indexed_ids(user_id)]
    for mid in to_index:
        print(f"indexing meeting {mid}: {index_meeting(user_id, mid)} chunks")
    return user_id, ids


def citation_line(c: dict) -> str:
    return (f"[{format_time(c['start'])}-{format_time(c['end'])}] {c['title']} "
            f"(chunk {c['chunk_id']}, score {c['score']:.3f})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows pipes default to cp1252
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scripts", default="a,b,c,d")
    parser.add_argument("--effort", choices=["low", "default"],
                        help="Q&A reasoning effort for this run (default: the qa.py constant)")
    parser.add_argument("--refresh", action="store_true", help="re-index the meetings")
    parser.add_argument("--retrieve-only", action="store_true", help="no LLM: top chunks only")
    args = parser.parse_args()

    if args.effort:
        qa.QA_REASONING_EFFORT = None if args.effort == "default" else args.effort
    db.DB_PATH = QA_DB
    try:
        user_id, ids = load_meetings(args.refresh)
    except EmbeddingError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    print(f"Q&A reasoning effort: {qa.QA_REASONING_EFFORT or 'Groq default'}\n")

    totals = {"total_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    for x in [s.strip() for s in args.scripts.split(",") if s.strip()]:
        print("=" * 78 + f"\nscript_{x}\n" + "=" * 78)
        for q in parse_questions(Path(QUESTIONS.format(x=x))):
            scope_id = ids[f"script_{x}"] if q.get("Scope", "").startswith("this") else None
            print(f"--- {q['heading']}")
            print(f"QUESTION   {q['Question']}")
            print(f"SCOPE      {'this meeting (script_' + x + ')' if scope_id else 'all my meetings'}")
            print(f"EXPECTED   {q.get('Expected', '')}")
            print(f"           meeting: {q.get('Expected meeting', '?')}"
                  + (f", time {q['Expected time']}" if q.get("Expected time") else ""))
            t0 = time.perf_counter()
            try:
                if args.retrieve_only:
                    print("TOP CHUNKS (no LLM)")
                    for c in qa.retrieve(user_id, q["Question"], scope_id, k=3):
                        print(f"  {citation_line(c)}\n      {c['text'][:140]}...")
                    print(f"TIME       {time.perf_counter() - t0:.1f} s")
                else:
                    result = qa.answer(user_id, q["Question"], scope_id)
                    seconds = time.perf_counter() - t0
                    usage = result["usage"]
                    for k in totals:
                        totals[k] += usage.get(k, 0)
                    print(f"ANSWERABLE {result['answerable']}")
                    print(f"ANSWER     {result['answer']}")
                    for c in result["citations"] or []:
                        print(f"  cited    {citation_line(c)}")
                    if result["dropped_citations"]:
                        print(f"  dropped citations (not retrieved): {result['dropped_citations']}")
                    print("  retrieved: " + "; ".join(
                        f"{c['title']} {format_time(c['start'])} ({c['score']:.2f})"
                        for c in result["retrieved"]))
                    print(f"TIME       {seconds:.1f} s | tokens {usage.get('total_tokens', 0)} "
                          f"(reasoning {usage.get('reasoning_tokens', 0)}) in "
                          f"{usage.get('calls', 0)} call(s)")
            except DailyLimitError as err:  # every further question would fail too: stop here
                print(f"STOPPED at script_{x} '{q['heading']}': {err}\n"
                      "Answers above this point are complete; nothing after it was run.")
                print(f"LLM tokens used this run: {totals['total_tokens']} "
                      f"(reasoning {totals['reasoning_tokens']}) in {totals['calls']} calls")
                sys.exit(2)
            except (EmbeddingError, AnalyzeError) as err:
                print(f"FAILED     {err}")
            print()
    if not args.retrieve_only:
        print(f"LLM tokens used this run: {totals['total_tokens']} "
              f"(reasoning {totals['reasoning_tokens']}) in {totals['calls']} calls")


if __name__ == "__main__":
    main()
