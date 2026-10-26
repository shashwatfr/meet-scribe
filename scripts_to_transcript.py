"""Makes "perfect" transcripts from the scripts that were read aloud: same words, no speaker labels.

Run by hand; run_script_tests.py uses the output as its "clean" input.

Usage: uv run python scripts_to_transcript.py
Reads data/scripts/script_x.txt, writes data/transcripts_clean/script_x.json in
the same format as transcribe_cli.py --save. Timestamps are made up from word
counts, so they only roughly match the recordings.
"""
import json
import re
from pathlib import Path

WORDS_PER_SECOND = 2.5
LABEL = re.compile(r"^[A-Z][A-Za-z]*:\s*")  # "PRIYA: " at the start of a line


def strip_label(line: str) -> str:
    return LABEL.sub("", line.strip())


def script_to_segments(text: str) -> list[dict]:
    segments, t = [], 0.0
    for line in text.splitlines():
        line = strip_label(line)
        if not line:
            continue
        duration = round(len(line.split()) / WORDS_PER_SECOND, 2)
        segments.append({"id": len(segments), "start": t, "end": t + duration, "text": line})
        t = round(t + duration + 0.5, 2)  # short pause between speakers
    return segments


def main() -> None:
    out_dir = Path("data/transcripts_clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    for script in sorted(Path("data/scripts").glob("script_?.txt")):
        segments = script_to_segments(script.read_text(encoding="utf-8"))
        out = out_dir / f"{script.stem}.json"
        out.write_text(json.dumps({"source": script.name, "model": "none (script text)",
                                   "language": "en", "segments": segments},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{out}: {len(segments)} segments, {segments[-1]['end']:.0f} s")


if __name__ == "__main__":
    main()
