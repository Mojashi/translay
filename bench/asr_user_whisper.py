"""Whisper large-v3-turbo (MLX, GPU) on the user's recordings, with and without a term list.

    uv run python asr_user_whisper.py data/user/read_*.wav > results/asr_user/whisper.jsonl

The term list stands in for the user's registered words: only terms, never sentences of the script.
"""
from __future__ import annotations

import json
import sys
import time

import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"
TERMS = ["search agent", "API gateway", "orchestrator", "sub tasks", "parallel", "Redis", "latency",
         "fine tune", "deployment"]


def main(paths: list[str]) -> None:
    mlx_whisper.transcribe(paths[0], path_or_hf_repo=MODEL, language="en")      # load + warm
    for engine, prompt in (("whisper-turbo", None), ("whisper-turbo+terms", "Glossary: " + ", ".join(TERMS) + ".")):
        for path in paths:
            t0 = time.perf_counter()
            result = mlx_whisper.transcribe(path, path_or_hf_repo=MODEL, language="en", initial_prompt=prompt,
                                            condition_on_previous_text=True)
            text = str(result["text"]).strip()
            print(json.dumps({"engine": engine, "file": path.split("/")[-1], "text": text,
                              "seconds": time.perf_counter() - t0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
