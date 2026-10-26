"""Command-line tool: analyse a saved transcript JSON and print the result.

Runs services.analyze.analyze() on a file made by transcribe_cli.py --save. Its
print_analysis() is also used by process_cli.py and run_script_tests.py.

Usage: uv run python analyze_cli.py <transcript.json> [--piece-minutes 1] [--save]
"""
import argparse
import json
import sys
from pathlib import Path

from services.analyze import AnalyzeError, analyze


def mmss(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def print_analysis(result: dict) -> None:
    """Print a result dict as returned by analyze() or loaded from a saved file."""
    summary, ex = result["summary"], result["extraction"]
    print(f"OVERVIEW\n  {summary['overview']}")
    print("KEY TOPICS\n  " + "; ".join(summary["key_topics"]))
    print("DECISIONS")
    for d in ex["decisions"] or [{"timestamp": 0, "topic": "-", "text": "(none)"}]:
        print(f"  [{mmss(d['timestamp'])}] ({d['topic']}) {d['text']}")
    print("ACTION ITEMS")
    for a in ex["action_items"] or []:
        print(f"  [{mmss(a['timestamp'])}] {a['task']}\n"
              f"           owner: {a['owner'] or 'null'} | due: {a['due_date'] or 'null'}")
    if not ex["action_items"]:
        print("  (none)")
    print("OPEN TOPICS")
    for topic in ex["open_topics"] or ["(none)"]:
        print(f"  - {topic}")
    if ex.get("superseded_decisions"):
        print("SUPERSEDED DECISIONS (replaced by a later decision on the same subject)")
        for s in ex["superseded_decisions"]:
            print(f"  [{mmss(s['old_timestamp'])}] {s['old_text']}\n"
                  f"     -> [{mmss(s['new_timestamp'])}] {s['new_text']}")
    log = result.get("merge_log")
    if log:
        print("MERGE OF PARTS")
        if log.get("fallback"):
            print(f"  {log['fallback']}")
        for c in log["clusters"]:
            print(f"  {c['kind']} cluster {c['ids']} keep {c['keep']}"
                  + (f", owner/due/time from {c['details_from']}" if c.get("details_from") else "")
                  + f"\n     -> {c['result']}")
            for member in c["members"]:
                print(f"        was: {member}")
        for d in log["dropped_open_topics"]:
            print(f"  open topic dropped: {d['open_topic']!r}, settled by [{mmss(d['decision_timestamp'])}] "
                  f"{d['decision']}")
    if result.get("owners_cleared"):
        print("OWNERS CLEARED (name not spoken in the task's segment or the two before it)")
        for c in result["owners_cleared"]:
            print(f"  [{mmss(c['timestamp'])}] {c['owner']!r} removed from: {c['task']}")
            for text in c["segments_checked"]:
                print(f"           checked: {text}")
    print(f"STEPS ({result['model']}, {result.get('pieces', '?')} piece(s))")
    for s in result["steps"]:
        print(f"  {s['step']:<16} {s['seconds']:>5.1f}s  calls {s.get('calls', 0)}  "
              f"tokens {s.get('total_tokens', 0)} (reasoning {s.get('reasoning_tokens', 0)})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows pipes default to cp1252
    parser = argparse.ArgumentParser(description="Summary, decisions and action items.")
    parser.add_argument("transcript", help="transcript JSON from transcribe_cli.py --save")
    parser.add_argument("--piece-minutes", type=float,
                        help="force splitting into pieces of this length (tests the merge)")
    parser.add_argument("--save", action="store_true", help="write data/analysis/<name>")
    args = parser.parse_args()

    path = Path(args.transcript)
    segments = json.loads(path.read_text(encoding="utf-8"))["segments"]
    piece_seconds = args.piece_minutes * 60 if args.piece_minutes else None
    try:
        result = analyze(segments, piece_seconds)
    except AnalyzeError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    result["source"] = str(path)
    print_analysis(result)
    if args.save:
        out = Path("data/analysis") / path.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
