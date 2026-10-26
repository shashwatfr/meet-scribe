"""Analysis check: runs the analysis on the test scripts and prints it next to the answer keys.

Run by hand to judge extraction quality; it calls services.analyze.analyze().

Usage: uv run python run_script_tests.py [--scripts a,d] [--inputs clean|real|both] [--refresh]

For each script it prints the answer key, the matching section of
data/transcript_errors.md, then one result per input (clean script text, real
recording). Results are saved in data/analysis/script_tests/ and re-printed
from there unless --refresh is given. Nothing is scored automatically.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from analyze_cli import print_analysis
from services.analyze import AnalyzeError, DailyLimitError, analyze

SCRIPTS_DIR = Path("data/scripts")
OUT_DIR = Path("data/analysis/script_tests")
ERRORS_FILE = Path("data/transcript_errors.md")
INPUTS = {"clean": "data/transcripts_clean/{name}.json", "real": "data/transcripts/{name}.mp4.json"}


def print_totals(totals: dict, reused: int) -> None:
    print("=" * 78)
    print(f"Tokens used this run: {totals['total_tokens']} "
          f"(reasoning {totals['reasoning_tokens']}) in {totals['calls']} calls; "
          f"{reused} result(s) re-printed from saved files.")


def errors_section(name: str) -> str:
    path = ERRORS_FILE
    if not path.exists():
        return "(data/transcript_errors.md not found)"
    match = re.search(rf"^## {name}\b.*?(?=^## |\Z)", path.read_text(encoding="utf-8"),
                      flags=re.MULTILINE | re.DOTALL)
    return match.group(0).strip() if match else "(no section)"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows pipes default to cp1252
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scripts", default="a,b,c,d")
    parser.add_argument("--inputs", choices=["clean", "real", "both"], default="both")
    parser.add_argument("--refresh", action="store_true", help="call the LLM even if saved")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    totals = {"total_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    reused = 0
    sources = ["clean", "real"] if args.inputs == "both" else [args.inputs]

    for letter in [s.strip() for s in args.scripts.split(",") if s.strip()]:
        name = f"script_{letter}"
        key = (SCRIPTS_DIR / f"{name}_key.md").read_text(encoding="utf-8")
        print("=" * 78 + f"\n{name.upper()}\n" + "=" * 78)
        print(f"--- ANSWER KEY ({name}_key.md) ---\n{key.strip()}\n")
        print(f"--- KNOWN RECOGNITION ERRORS (real recording) ---\n{errors_section(name)}\n")

        for source in sources:
            saved = OUT_DIR / f"{name}_{source}.json"
            print("-" * 78 + f"\n{name}  [{source.upper()} transcript]\n" + "-" * 78)

            if saved.exists() and not args.refresh:
                result = json.loads(saved.read_text(encoding="utf-8"))
                reused += 1
                print(f"(re-printed from {saved})")
            else:
                transcript = Path(INPUTS[source].format(name=name))
                if not transcript.exists():
                    print(f"(skipped: {transcript} not found)\n")
                    continue
                segments = json.loads(transcript.read_text(encoding="utf-8"))["segments"]
                try:
                    result = analyze(segments)
                except DailyLimitError as err:  # every further call would fail too: stop here
                    print(f"STOPPED at {name} [{source}]: {err}\n"
                          "Results above this point are complete; nothing after it was run.")
                    print_totals(totals, reused)
                    sys.exit(2)
                except AnalyzeError as err:
                    print(f"(failed: {err})\n")
                    continue
                result.update(source=str(transcript), input=source)
                saved.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
                for step in result["steps"]:
                    for k in totals:
                        totals[k] += step.get(k, 0)
            print_analysis(result)
            print()

    print_totals(totals, reused)


if __name__ == "__main__":
    main()
