"""Does idle time between requests slow MLX down (GPU clock ramp-down)?

    uv run python mt_idle_gap.py --model ../models/plamo-3-translate-0.8b-MLX-4bit --gap-ms 0
    uv run python mt_idle_gap.py --model ../models/plamo-3-translate-0.8b-MLX-4bit --gap-ms 300

Replays word-by-word revisions (es + ja, as the app does) with a fixed idle gap between requests,
without the recogniser running, and reports time per forward pass.
"""
from __future__ import annotations

import argparse
import statistics
import time

from speech_corpus import speech_talks
from translay_mt.plamo_fast import load_plamo
from translay_mt.streaming_translator import StreamingTranslator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gap-ms", type=float, required=True)
    ap.add_argument("--talks", type=int, default=3)
    args = ap.parse_args()
    model, tok = load_plamo(args.model)
    translators = [StreamingTranslator(model, tok, target_lang=lang) for lang in ("es", "ja")]
    per_pass, per_request = [], []
    for warm in (True, False):
        for talk in speech_talks()[: 1 if warm else args.talks]:
            for tr in translators:
                tr.reset()
            for sentence in talk.sentences:
                words = sentence.split()
                for i in range(1, len(words) + 1):
                    if not warm:
                        time.sleep(args.gap_ms / 1000)
                    t0 = time.perf_counter()
                    passes = 0
                    for tr in translators:
                        rev = tr.revise(" ".join(words[:i]))
                        passes += rev.stats.forward_passes
                    dt = (time.perf_counter() - t0) * 1000
                    if not warm:
                        per_request.append(dt)
                        per_pass.append(dt / passes)
                for tr in translators:
                    tr.commit()
                    tr.settle()
    q = lambda xs, p: sorted(xs)[int(len(xs) * p)]
    print(f"gap={args.gap_ms:.0f}ms requests={len(per_request)} "
          f"request median={statistics.median(per_request):.1f}ms p90={q(per_request, .9):.1f}ms | "
          f"per forward pass median={statistics.median(per_pass):.2f}ms p90={q(per_pass, .9):.2f}ms")


if __name__ == "__main__":
    main()
