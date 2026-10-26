"""Measures transcription accuracy: compares saved transcripts with the scripts read aloud.

Run by hand. Prints the word error rate (WER) and every word difference.

Usage: uv run python eval_transcripts.py
Reads data/transcripts/script_x.mp4.json and data/scripts/script_x.txt for
x = a, b, c, d.
"""
import json
import re
from pathlib import Path

SCRIPTS = ["script_a", "script_b", "script_c", "script_d"]


def normalize(text: str) -> list[str]:
    """Lowercase, remove speaker labels and punctuation, split into words."""
    text = re.sub(r"^[A-Z][A-Za-z]*:", " ", text, flags=re.MULTILINE)  # "PRIYA: ..." lines
    text = re.sub(r"[-–—/]", " ", text.lower())  # dashes and slashes separate words
    return re.sub(r"[^\w\s]", "", text).split()


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str, str]]:
    """Minimum-edit word alignment. Returns (op, ref_word, hyp_word) with op in OK/SUB/DEL/INS."""
    n, m = len(ref), len(hyp)
    # cost[i][j] = fewest edits to turn ref[:i] into hyp[:j]
    cost = [[i + j if i == 0 or j == 0 else 0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                             cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i or j:  # walk back from the end to recover which edit was used
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("OK" if ref[i - 1] == hyp[j - 1] else "SUB", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("DEL", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("INS", "", hyp[j - 1]))
            j -= 1
    return ops[::-1]


def main() -> None:
    print("Note: the recordings may include ad-libs, so not every difference is a "
          "transcription error.\n")
    for name in SCRIPTS:
        script = Path("data/scripts") / f"{name}.txt"
        saved = Path("data/transcripts") / f"{name}.mp4.json"
        if not (script.exists() and saved.exists()):
            print(f"== {name}: skipped (missing {script if not script.exists() else saved})\n")
            continue
        ref = normalize(script.read_text(encoding="utf-8"))
        segments = json.loads(saved.read_text(encoding="utf-8"))["segments"]
        hyp = normalize(" ".join(s["text"] for s in segments))
        ops = align(ref, hyp)
        errors = [op for op in ops if op[0] != "OK"]
        counts = {k: sum(op[0] == k for op in errors) for k in ("SUB", "DEL", "INS")}
        print(f"== {name}: WER {len(errors) / len(ref):.1%} ({len(errors)} errors / "
              f"{len(ref)} script words) | SUB {counts['SUB']}  DEL {counts['DEL']}  "
              f"INS {counts['INS']}")
        for op, r, h in errors:
            print(f"   {op}  {r or '-':<14} -> {h or '-'}")
        print()


if __name__ == "__main__":
    main()
