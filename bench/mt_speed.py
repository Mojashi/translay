"""Measure PLaMo 3 Translate decode/prefill speed and record greedy translations.

    uv run python mt_speed.py --model models/plamo-3-translate-2b-MLX-4bit --variant ref
    uv run python mt_speed.py --model models/plamo-3-translate-2b-MLX-4bit --variant fast

Each variant runs in its own process because `fast` patches the modeling classes.
Outputs go to results/mt_speed_<model>_<variant>.json for agreement comparison.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from translay_mt.plamo_fast import accelerate, load_plamo
from speech_corpus import speech_sentences

TAG, BOS = 16, 1
STOP = {TAG, BOS}


def pfml3_prompt(source: str, target_lang: str = "ja") -> str:
    return (
        "<|plamo:tag|>dataset<|plamo:msg|>translation"
        f"<|plamo:tag|>input<|plamo:msg|>{source}"
        f"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target_lang}<|plamo:msg|>"
    )


def greedy(model, ids: list[int], max_new: int = 256) -> tuple[list[int], float, list[float]]:
    """Prefill then pipelined greedy decode. Returns tokens, prefill seconds, per-token seconds."""
    cache = model.make_cache()
    t0 = time.perf_counter()
    y = mx.argmax(model(mx.array(ids)[None], cache=cache)[:, -1], axis=-1)
    mx.eval(y)
    prefill = time.perf_counter() - t0
    out: list[int] = []
    steps: list[float] = []
    t = time.perf_counter()
    for _ in range(max_new):
        ny = mx.argmax(model(y[None], cache=cache)[:, -1], axis=-1)
        mx.async_eval(ny)
        tok = int(np.asarray(y)[0])
        now = time.perf_counter()
        steps.append(now - t)
        t = now
        if tok in STOP:
            break
        out.append(tok)
        y = ny
    return out, prefill, steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["ref", "fast"], required=True)
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    model, tok = load_plamo(args.model, fast=False)
    applied = accelerate(model) if args.variant == "fast" else {}

    def encode(text: str) -> list[int]:
        return [BOS] + tok.encode(text, add_special_tokens=False)

    sentences = speech_sentences()[: args.n]
    for s in sentences[:3]:                              # warm-up: graph build, kernels, allocator
        greedy(model, encode(pfml3_prompt(s)), 32)

    records = []
    prefill_rate = []
    for s in sentences:
        ids = encode(pfml3_prompt(s))
        out, prefill, steps = greedy(model, ids)
        records.append({"source": s, "ja": tok.decode(out), "n_prompt": len(ids), "n_out": len(out),
                        "prefill_s": prefill, "decode_ms": [x * 1000 for x in steps[1:]]})
        prefill_rate.append(len(ids) / prefill)

    # Long-prompt prefill: what a translation history of a few hundred tokens costs to rebuild.
    long_src = " ".join(sentences[:12])
    long_ids = encode(pfml3_prompt(long_src))
    long_prefill = statistics.median(greedy(model, long_ids, 1)[1] for _ in range(5))

    per_tok = [ms for r in records for ms in r["decode_ms"]]
    lat = [r["prefill_s"] + sum(r["decode_ms"]) / 1000 for r in records]
    summary = {
        "model": Path(args.model).name, "variant": args.variant, "applied": applied,
        "decode_ms_per_token_median": statistics.median(per_tok),
        "decode_tok_per_s": 1000 / statistics.median(per_tok),
        "short_prefill_ms_median": statistics.median(r["prefill_s"] for r in records) * 1000,
        "short_prompt_tokens_median": statistics.median(r["n_prompt"] for r in records),
        "long_prefill_tokens": len(long_ids), "long_prefill_ms": long_prefill * 1000,
        "sentence_latency_ms_median": statistics.median(lat) * 1000,
        "sentence_latency_ms_p90": sorted(lat)[int(len(lat) * 0.9)] * 1000,
        "out_tokens_median": statistics.median(r["n_out"] for r in records),
        "peak_mem_gb": mx.get_peak_memory() / 1e9,
    }
    out_path = Path("results") / f"mt_speed_{Path(args.model).name}_{args.variant}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
