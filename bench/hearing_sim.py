"""Replay a recording through RollingListener as if it were live, and report what each pass cost.

    (cd ../translator && uv run python ../bench/hearing_sim.py ../models/Qwen3-ASR-1.7B-4bit \
        ../bench/data/user/read_20261107-201046.wav --terms "API gateway,orchestrator,Redis")

The loop is the real one: each pass takes all the audio that "arrived" (wall clock) while the
previous pass ran. Prints per-pass stats and the final transcript's WER against the script.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_qwen3_asr.audio import load_audio

from translay_mt.hearing import SAMPLE_RATE, RollingListener
from translay_mt.hearing_whisper import WhisperListener


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("audio")
    ap.add_argument("--terms", default="")
    ap.add_argument("--engine", choices=["qwen", "whisper"], required=True)
    ap.add_argument("--ane-encoder", help="WhisperKit AudioEncoder.mlmodelc: encode on the Neural Engine")
    args = ap.parse_args()
    mx.set_cache_limit(512 * 1024 * 1024)       # as the engine does; unbounded, it grows to GBs
    terms = [t for t in args.terms.split(",") if t]
    listener = (WhisperListener.load(args.model, terms=terms, ane_encoder=args.ane_encoder) if args.engine == "whisper"
                else RollingListener.load(args.model, terms=terms))
    audio = np.array(load_audio(args.audio), dtype=np.float32)

    warm = (WhisperListener(model=listener.model, ane_encoder=listener.ane_encoder) if isinstance(listener, WhisperListener)
            else RollingListener(model=listener.model, tokenizer=listener.tokenizer))
    warm.hear(audio[: 3 * SAMPLE_RATE])
    warm.listen()

    t0 = time.perf_counter()
    fed = 0
    passes = []
    while fed < len(audio):
        arrived = min(len(audio), int((time.perf_counter() - t0) * SAMPLE_RATE))
        if arrived - fed < SAMPLE_RATE // 10 and arrived < len(audio):   # the last scrap is fed as is
            time.sleep(0.02)
            continue
        listener.hear(audio[fed:arrived])
        fed = arrived
        listener.listen()
        s = listener.last
        passes.append(s)
        print(f"[{fed / SAMPLE_RATE:5.1f}s] pass {s.seconds * 1000:5.0f}ms window={s.audio_sec:4.1f}s "
              f"encoded={s.encoded_sec:4.1f}s prefilled={s.prefilled:3d} draft={s.accepted}/{s.drafted} "
              f"stepped={s.stepped:2d} | …{listener.text[-60:]}", flush=True)
    secs = [p.seconds * 1000 for p in passes]
    print(f"\npasses={len(passes)} median={statistics.median(secs):.0f}ms p90={sorted(secs)[int(len(secs) * .9)]:.0f}ms "
          f"encoded/pass={statistics.mean(p.encoded_sec for p in passes):.1f}s "
          f"prefilled/pass={statistics.mean(p.prefilled for p in passes):.0f} "
          f"accept={sum(p.accepted for p in passes) / max(1, sum(p.drafted for p in passes)):.0%}")
    print("\nTEXT:", listener.text, flush=True)
    try:
        import jiwer
        sys.path.insert(0, "../bench")
        from asr_user_score import REFERENCE, normalise
        print(f"WER vs script: {jiwer.wer(normalise(REFERENCE), normalise(listener.text)) * 100:.1f}%")
    except ImportError:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
