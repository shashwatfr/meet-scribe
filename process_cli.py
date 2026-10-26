"""Command-line version of the app: log in, then process, list, show, export or ask.

Used for testing without the UI; it calls the same services as the Streamlit pages.

Usage:
  uv run python process_cli.py <file> --user alice [--password ...] [--title "..."]
  uv run python process_cli.py --user alice --list
  uv run python process_cli.py --user alice --show ID
  uv run python process_cli.py --user alice --retry ID
  uv run python process_cli.py --user alice --export ID PATH.docx
  uv run python process_cli.py --user alice --index ID | --index-missing
  uv run python process_cli.py --user alice --search "question" [--meeting ID]   (no LLM)
  uv run python process_cli.py --user alice --ask "question" [--meeting ID]
Without --password you are asked for it (keeps it out of shell history).
"""
import argparse
import getpass
import sys
from pathlib import Path

from analyze_cli import mmss, print_analysis
from services.analyze import AnalyzeError
from services.audio import AudioError
from services.auth import verify_login
from services.export import build_docx
from services.index import EmbeddingError, index_missing, indexed_ids
from services.pipeline import process_recording, retry_analysis, retry_index
from services.qa import answer, retrieve
from services.store import get_meeting, list_meetings


def print_chunks(chunks: list[dict]) -> None:
    if not chunks:
        print("  (no indexed chunks in this scope)")
    for c in chunks:
        print(f"  {c['score']:.3f}  [{mmss(c['start'])}-{mmss(c['end'])}] meeting {c['meeting_id']} "
              f"({c['title']}), chunk {c['chunk_id']}")
        print(f"         {c['text'][:150]}{'...' if len(c['text']) > 150 else ''}")


def print_answer(result: dict) -> None:
    print(f"ANSWERABLE {result['answerable']}\nANSWER     {result['answer']}")
    for c in result["citations"]:
        print(f"  cited [{mmss(c['start'])}] {c['title']} (chunk {c['chunk_id']})")
    if result["dropped_citations"]:
        print(f"  dropped citations (not retrieved): {result['dropped_citations']}")


def show_meeting(m: dict) -> None:
    duration = mmss(m["duration_seconds"]) if m["duration_seconds"] else "-"
    print(f"MEETING {m['id']}: {m['title']}  ({m['source_filename']}, {duration})")
    print(f"STATUS  {m['status']}" + (f"  - {m['error_message']}" if m["error_message"] else ""))
    print("TIMINGS " + ("  ".join(f"{k} {v:.1f}s" for k, v in m["timings"].items()) or "-"))
    if m["transcript"]:
        print(f"TRANSCRIPT ({len(m['transcript']['segments'])} segments)")
        for s in m["transcript"]["segments"]:
            print(f"  [{mmss(s['start'])}] {s['text']}")
    if m["analysis"]:
        print_analysis(m["analysis"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows pipes default to cp1252
    parser = argparse.ArgumentParser(description="Process and view meetings as a user.")
    parser.add_argument("file", nargs="?", help="recording to process")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", help="omit to be asked for it")
    parser.add_argument("--title")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show", type=int, metavar="ID")
    parser.add_argument("--retry", type=int, metavar="ID", help="re-run only the analysis")
    parser.add_argument("--export", nargs=2, metavar=("ID", "PATH"), help="write a .docx")
    parser.add_argument("--index", type=int, metavar="ID", help="(re)build the Q&A index")
    parser.add_argument("--index-missing", action="store_true", help="index all unindexed meetings")
    parser.add_argument("--search", metavar="QUESTION", help="print the top chunks (no LLM)")
    parser.add_argument("--ask", metavar="QUESTION", help="answer with citations")
    parser.add_argument("--meeting", type=int, metavar="ID", help="limit --search/--ask to one meeting")
    args = parser.parse_args()

    password = args.password if args.password is not None else getpass.getpass("Password: ")
    user = verify_login(args.user, password)
    if user is None:
        print("Login failed.", file=sys.stderr)
        sys.exit(1)
    print(f"Logged in as {user['username']} ({user['role']}).", flush=True)

    if args.index is not None or args.index_missing:
        try:
            if args.index_missing:
                results = index_missing(user["id"])
            else:
                count = retry_index(user["id"], args.index)
                if count is None:
                    print("Meeting not found, or it has no transcript.", file=sys.stderr)
                    sys.exit(1)
                results = [(args.index, f"{count} chunks")]
        except EmbeddingError as err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
        for meeting_id, outcome in results or [("-", "nothing to index")]:
            print(f"  meeting {meeting_id}: {outcome}")
        print(f"Indexed meetings: {sorted(indexed_ids(user['id']))}")
    elif args.search or args.ask:
        question = args.search or args.ask
        print(f"QUESTION   {question}  (scope: {'meeting ' + str(args.meeting) if args.meeting else 'all my meetings'})")
        try:
            if args.search:
                print_chunks(retrieve(user["id"], question, args.meeting) or [])
            else:
                result = answer(user["id"], question, args.meeting)
                print_answer(result)
                print("RETRIEVED")
                print_chunks(result["retrieved"])
        except (EmbeddingError, AnalyzeError) as err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
    elif args.export:
        meeting_id, out = int(args.export[0]), Path(args.export[1])
        data = build_docx(user["id"], meeting_id)
        if data is None:
            print("Meeting not found, or it has no transcript.", file=sys.stderr)
            sys.exit(1)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"Wrote {out} ({len(data):,} bytes).")
    elif args.list:
        meetings = list_meetings(user["id"])
        print(f"{len(meetings)} meeting(s)")
        for m in meetings:
            print(f"  {m['id']:>4}  {m['status']:<16} {m['created_at']}  {m['title']}")
    elif args.show is not None or args.retry is not None:
        meeting_id = args.show if args.show is not None else args.retry
        m = get_meeting(user["id"], meeting_id) if args.show is not None \
            else retry_analysis(user["id"], meeting_id)
        if m is None:
            print("Meeting not found." if args.show is not None
                  else "Meeting not found, or it has no transcript to analyse.", file=sys.stderr)
            sys.exit(1)
        show_meeting(m)
    elif args.file:
        try:
            m = process_recording(args.file, user["id"], args.title)
        except AudioError as err:
            print(f"Error: {err}", file=sys.stderr)
            sys.exit(1)
        show_meeting(m)
    else:
        parser.error("give a file, --list, --show ID or --retry ID")


if __name__ == "__main__":
    main()
