"""Score ASR hypotheses on the user's read-aloud recordings against the script they read.

    uv run python asr_user_score.py results/asr_user/*.jsonl

Each .jsonl line: {"engine", "file", "text", "seconds"}. WER is computed after lower-casing and
stripping punctuation; numbers/hyphenation are normalised for a few known cases ("sub tasks").
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import jiwer

REFERENCE = (Path(__file__).parent / "data" / "user" / "reference.txt").read_text()


def normalise(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"\bsubtasks\b", "sub tasks", text)
    text = re.sub(r"\bfrontend\b", "front end", text)
    text = re.sub(r"\bfinetune\b", "fine tune", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def main(paths: list[str]) -> None:
    ref = normalise(REFERENCE)
    by_engine: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                by_engine[row["engine"]].append(row)
    rows = []
    for engine, items in by_engine.items():
        wers = [jiwer.wer(ref, normalise(it["text"])) for it in items]
        secs = sum(it["seconds"] for it in items)
        rows.append((sum(wers) / len(wers), engine, wers, secs))
    for mean, engine, wers, secs in sorted(rows):
        print(f"{engine:28s} WER mean={mean * 100:5.1f}%  per file={[round(w * 100, 1) for w in wers]}  "
              f"time={secs:.1f}s")
    if "--show" in sys.argv:
        for _, engine, _, _ in sorted(rows):
            for it in by_engine[engine]:
                print(f"\n[{engine} {it['file']}]\n{it['text']}")


if __name__ == "__main__":
    main([p for p in sys.argv[1:] if not p.startswith("--")])
