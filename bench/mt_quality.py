"""Translate every speech talk sentence by sentence (as the live app commits them) and score it.

    uv run python mt_quality.py --model models/plamo-3-translate-2b-MLX-4bit

Writes results/mt_quality_<model>.json with per-talk outputs for both history settings, so that a
heavier metric (COMET) can score the same outputs later without re-translating.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sacrebleu.metrics.chrf import CHRF

from translay_mt.plamo_fast import load_plamo
from speech_corpus import speech_talks
from translay_mt.streaming_translator import StreamingTranslator


def translate_talks(tr: StreamingTranslator, talks) -> tuple[list[str], float]:
    outputs, t0 = [], time.perf_counter()
    for talk in talks:
        tr.reset()
        parts = []
        for sentence in talk.sentences:
            parts.append(tr.revise(sentence).text)
            tr.commit()
        outputs.append("".join(parts))
    return outputs, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    model, tok = load_plamo(args.model)
    talks = speech_talks()
    refs = [t.reference for t in talks]

    result = {"model": Path(args.model).name, "talks": [t.doc_id for t in talks],
              "sources": [t.source for t in talks], "references": refs}
    for name, budget in [("history", 900), ("no_history", 0)]:
        tr = StreamingTranslator(model, tok, use_draft=False, history_budget=budget)
        outs, secs = translate_talks(tr, talks)
        chrf = CHRF().corpus_score(outs, [refs]).score
        result[name] = {"outputs": outs, "chrF": chrf, "seconds": secs}
        print(f"{result['model']} {name}: chrF={chrf:.2f} time={secs:.0f}s", flush=True)
    out = Path("results") / f"mt_quality_{result['model']}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
