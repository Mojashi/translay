"""Simulate live captioning: English arrives word by word, the translation is revised each time.

    uv run python mt_stream_sim.py --model models/plamo-3-translate-2b-MLX-4bit --talks 8

Compares what the naive cascade throws away against the stateful translator.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from sacrebleu.metrics.chrf import CHRF

from translay_mt.plamo_fast import load_plamo
from speech_corpus import speech_talks
from translay_mt.streaming_translator import KVTape, StreamingTranslator

MODES = {
    # name: (keep tape between revisions, keep history, use draft)
    "naive": (False, False, False),
    "naive+history": (False, True, False),
    "tape": (True, True, False),
    "tape+draft": (True, True, True),
}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def run_mode(model, tok, talks, mode: str) -> dict:
    keep_tape, keep_history, use_draft = MODES[mode]
    tr = StreamingTranslator(model, tok, use_draft=use_draft,
                             history_budget=900 if keep_history else 0)
    revisions, finals, talk_outputs = [], [], []
    for talk in talks:
        tr.reset()
        sentences_ja = []
        for sentence in talk.sentences:
            words = sentence.split()
            rev = None
            for i in range(1, len(words) + 1):
                if not keep_tape:
                    tr.tape = KVTape(model)
                rev = tr.revise(" ".join(words[:i]))
                revisions.append(asdict(rev.stats))
            assert rev is not None, "a sentence has at least one word"
            finals.append({"source": sentence, "ja": rev.text, "seconds": rev.stats.seconds})
            sentences_ja.append(rev.text)
            tr.commit()
        talk_outputs.append("".join(sentences_ja))

    secs = [r["seconds"] * 1000 for r in revisions]
    final_secs = [f["seconds"] * 1000 for f in finals]
    offered = sum(r["draft_offered"] for r in revisions)
    return {
        "mode": mode,
        "revisions": len(revisions),
        "revision_ms_median": statistics.median(secs),
        "revision_ms_p90": pct(secs, 0.9),
        "final_ms_median": statistics.median(final_secs),
        "final_ms_p90": pct(final_secs, 0.9),
        "prompt_tokens_fed_mean": statistics.mean(r["fed_prompt"] for r in revisions),
        "sequential_steps_mean": statistics.mean(r["sequential_steps"] for r in revisions),
        "forward_passes_mean": statistics.mean(r["forward_passes"] for r in revisions),
        "draft_acceptance": (sum(r["draft_accepted"] for r in revisions) / offered) if offered else None,
        "chrF_vs_human": CHRF().corpus_score(talk_outputs, [[t.reference for t in talks]]).score,
        "_finals": finals,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--talks", type=int, default=8)
    ap.add_argument("--modes", default=",".join(MODES))
    args = ap.parse_args()

    model, tok = load_plamo(args.model)
    talks = speech_talks()[: args.talks]
    run_mode(model, tok, talks[:1], "tape+draft")          # warm kernels for every shape class

    results = {}
    for mode in args.modes.split(","):
        r = run_mode(model, tok, talks, mode)
        results[mode] = r
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False))

    if "tape" in results and "tape+draft" in results:
        a, b = results["tape"]["_finals"], results["tape+draft"]["_finals"]
        same = sum(x["ja"] == y["ja"] for x, y in zip(a, b))
        print(f"final translations identical tape vs tape+draft: {same}/{len(a)}")
    out = Path("results") / f"mt_stream_{Path(args.model).name}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
