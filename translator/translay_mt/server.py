"""Resident translator for the Translay app: JSON lines over stdin/stdout.

    uv run --project translator python -m translay_mt.server \
        --model models/plamo-3-translate-0.8b-MLX-4bit --targets es,ja [--trace]

One model is loaded once; each target language gets its own StreamingTranslator (its own KV tape,
history and draft) on top of it.

Protocol (one request in flight; every request gets exactly one reply line):

    → {"type": "revise", "text": "<unfinished English sentence>"}
    ← {"type": "translation", "final": false, "texts": {"es": "...", "ja": "..."}, "ms": 71.2}

    → {"type": "finish", "text": "<finished English sentence>"}
    ← {"type": "translation", "final": true,  "texts": {"es": "...", "ja": "..."}, "ms": 66.0}

    → {"type": "reset"}
    ← {"type": "reset"}

On start-up it prints {"type": "ready", ...} once the model is loaded and warmed.

WHY one thread and stdio: MLX streams are bound to the thread that created them (a threaded HTTP
server fails with "There is no Stream(gpu, 1) in current thread"), and the app is the only client.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from translay_mt.plamo_fast import load_plamo
from translay_mt.translating import Interpreters, log

WARMUP = [
    "Okay, so let's get started.",
    "The meeting has been moved to next Monday, which honestly works better for me.",
]


def reply(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def warm(interpreters: Interpreters) -> None:
    """Pay MLX's first-use kernel/graph costs before the first caption, not during it."""
    trace, interpreters.trace = interpreters.trace, False
    for sentence in WARMUP:
        words = sentence.split()
        for i in range(1, len(words)):
            interpreters.revise(" ".join(words[:i]), None)
        interpreters.finish(sentence, None)
        interpreters.settle()
    interpreters.reset()
    interpreters.trace = trace


def serve(interpreters: Interpreters) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        kind = request["type"]
        t0 = time.perf_counter()
        if kind == "revise":
            texts = interpreters.revise(request["text"], None)
            reply({"type": "translation", "final": False, "texts": texts, "ms": (time.perf_counter() - t0) * 1000})
        elif kind == "finish":
            texts = interpreters.finish(request["text"], None)
            reply({"type": "translation", "final": True, "texts": texts, "ms": (time.perf_counter() - t0) * 1000})
            # Idle until the next request arrives anyway: rebuild the context tape now.
            interpreters.settle()
        elif kind == "reset":
            interpreters.reset()
            reply({"type": "reset"})
        else:
            raise ValueError(f"unknown request type: {kind!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="PLaMo 3 Translate MLX checkpoint directory")
    ap.add_argument("--targets", required=True, help="comma-separated target languages, e.g. es,ja")
    ap.add_argument("--trace", action="store_true", help="log per-request timing to stderr")
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        raise SystemExit("--targets must name at least one language")

    t0 = time.perf_counter()
    model, tokenizer = load_plamo(args.model)
    interpreters = Interpreters(model, tokenizer, targets, args.trace)
    warm(interpreters)
    reply({"type": "ready", "model": args.model, "targets": targets, "seconds": time.perf_counter() - t0})
    serve(interpreters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
