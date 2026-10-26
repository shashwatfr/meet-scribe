"""Command-line tool: transcribe one recording and print the timestamped transcript.

Used to test transcription on its own; --save writes data/transcripts/<file>.json,
which analyze_cli.py can read.

Usage: uv run python transcribe_cli.py <file> [--chunk-minutes 2] [--language en] [--save]
"""
import argparse
import json
import sys
import time
from pathlib import Path

from services.audio import AudioError, prepare_chunks
from services.transcribe import MODEL, TranscribeError, transcribe


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows pipes default to cp1252
    parser = argparse.ArgumentParser(description="Print a timestamped transcript.")
    parser.add_argument("file")
    parser.add_argument("--chunk-minutes", type=float,
                        help="force splitting at this length (default: one chunk if under 20 MB)")
    parser.add_argument("--language", help="ISO-639-1 code such as en (default: auto-detect)")
    parser.add_argument("--save", action="store_true",
                        help="write data/transcripts/<filename>.json, e.g. meeting.mp4.json")
    args = parser.parse_args()

    try:
        t0 = time.perf_counter()
        chunk_seconds = args.chunk_minutes * 60 if args.chunk_minutes else None
        with prepare_chunks(args.file, chunk_seconds=chunk_seconds) as chunks:
            t1 = time.perf_counter()
            segments, detected = transcribe(chunks, language=args.language)
            t2 = time.perf_counter()
    except (AudioError, TranscribeError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    for seg in segments:
        print(f"[{hms(seg['start'])} - {hms(seg['end'])}] {seg['text'].strip()}")
    print(f"\nextract:    {t1 - t0:.1f}s ({len(chunks)} chunk(s))")
    print(f"transcribe: {t2 - t1:.1f}s")

    if args.save:
        # Keep the extension so meeting.mp4 and meeting.wav don't overwrite each other.
        out = Path("data/transcripts") / f"{Path(args.file).name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"source": Path(args.file).name, "model": MODEL,
                                   "language": detected or args.language, "segments": segments},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved:      {out}")


if __name__ == "__main__":
    main()
