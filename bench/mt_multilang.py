"""Several target languages: revise them one after another, or packed into shared forward passes?

    uv run python mt_multilang.py --model ../models/plamo-3-translate-0.8b-MLX-4bit --targets es,ja

Replays word-by-word revisions of a few talks (as the app does) in both modes and reports the
per-request time and how often the two modes produce the same translation.
"""
from __future__ import annotations

import argparse
import statistics
import time

from speech_corpus import speech_talks
from translay_mt.plamo_fast import load_plamo
from translay_mt.streaming_translator import StreamingTranslator, revise_together, settle_together


def run(translators: list[StreamingTranslator], talks, together: bool) -> tuple[list[float], list[list[str]]]:
    times, finals = [], []
    for talk in talks:
        for tr in translators:
            tr.reset()
        for sentence in talk.sentences:
            words = sentence.split()
            texts = []
            for i in range(1, len(words) + 1):
                source = " ".join(words[:i])
                t0 = time.perf_counter()
                if together:
                    texts = [r.text for r in revise_together(translators, source)]
                else:
                    texts = [tr.revise(source).text for tr in translators]
                times.append((time.perf_counter() - t0) * 1000)
            finals.append(texts)
            for tr in translators:
                tr.commit()
            if together:
                settle_together(translators)
            else:
                for tr in translators:
                    tr.settle()
    return times, finals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--talks", type=int, default=4)
    args = ap.parse_args()
    model, tok = load_plamo(args.model)
    targets = args.targets.split(",")
    translators = [StreamingTranslator(model, tok, target_lang=t) for t in targets]
    talks = speech_talks()[: args.talks]
    run(translators, talks[:1], together=True)     # warm-up
    run(translators, talks[:1], together=False)

    q = lambda xs, p: sorted(xs)[int(len(xs) * p)]
    results = {}
    for mode, together in (("sequential", False), ("together", True)):
        times, finals = run(translators, talks, together)
        results[mode] = finals
        print(f"{mode:10s} requests={len(times)} median={statistics.median(times):6.1f}ms "
              f"p90={q(times, .9):6.1f}ms max={max(times):6.1f}ms")
    a, b = results["sequential"], results["together"]
    for i, lang in enumerate(targets):
        same = sum(x[i] == y[i] for x, y in zip(a, b))
        print(f"final translations identical ({lang}): {same}/{len(a)}")


if __name__ == "__main__":
    main()
