"""Qwen3-ASR (MLX) on the user's recordings, offline, with and without a term list as context.

    ~/repos/.venv-qwen-asr/bin/python asr_user_qwen.py ../models/Qwen3-ASR-1.7B-4bit data/user/read_*.wav

Runs in the mlx-qwen3-asr virtualenv (not the bench env). Same term list as the Whisper run: only
terms the user might register, never sentences of the script.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx_qwen3_asr as qa

TERMS = ["search agent", "API gateway", "orchestrator", "sub tasks", "parallel", "Redis", "latency",
         "fine tune", "deployment"]


def main(model_path: str, paths: list[str]) -> None:
    session = qa.Session(model_path)
    name = Path(model_path).name
    session.transcribe(paths[0], language="English")               # load + compile
    for engine, context in ((f"{name}", ""), (f"{name}+terms", "Terms: " + ", ".join(TERMS))):
        for path in paths:
            t0 = time.perf_counter()
            result = session.transcribe(path, language="English", context=context)
            print(json.dumps({"engine": engine, "file": Path(path).name, "text": result.text.strip(),
                              "seconds": time.perf_counter() - t0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
