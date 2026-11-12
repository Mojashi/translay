"""How cheap can one Whisper-turbo recognition pass get on this Mac?

    uv run python whisper_probe.py data/user/read_*.wav

1. Speed: encoder time for 5/10/15/30 s windows, encoding the full 30 s (as trained) vs only the
   audio actually present ("trimmed", like whisper.cpp's audio_ctx); plus one decoder pass that
   verifies a 40-token draft.
2. Accuracy: the recordings cut into fixed 10 s chunks, each decoded greedily with the full and the
   trimmed encoder; WER of the concatenation against the script (chunk-edge damage is identical).
"""
from __future__ import annotations

import statistics
import sys
import time

import jiwer
import mlx.core as mx
import numpy as np
from mlx_whisper import audio as wa
from mlx_whisper.load_models import load_model
from mlx_whisper.tokenizer import get_tokenizer

from asr_user_score import REFERENCE, normalise

MODEL = "../models/whisper-large-v3-turbo"
SR = 16_000

model = load_model(MODEL)
tok = get_tokenizer(multilingual=True, num_languages=model.dims.n_vocab - 51765 - 1, language="en", task="transcribe")
PROMPT = list(tok.sot_sequence_including_notimestamps)


def encode(samples: np.ndarray, trimmed: bool) -> mx.array:
    """Mel → encoder. Full: pad to 30 s as trained. Trimmed: only the frames that hold audio."""
    n = len(samples)
    padded = np.zeros(wa.N_SAMPLES, dtype=np.float32)
    padded[: min(n, wa.N_SAMPLES)] = samples[: wa.N_SAMPLES]
    mel = wa.log_mel_spectrogram(mx.array(padded), n_mels=model.dims.n_mels)[None]
    enc = model.encoder
    if not trimmed:
        return enc(mel)
    frames = min(wa.N_FRAMES, int(np.ceil(n / wa.HOP_LENGTH / 2)) * 2)
    x = nn_gelu(enc.conv1(mel[:, :frames]))
    x = nn_gelu(enc.conv2(x))
    x = x + enc._positional_embedding[: x.shape[1]]
    for block in enc.blocks:
        x, _, _ = block(x)
    return enc.ln_post(x)


def nn_gelu(x):
    import mlx.nn as nn
    return nn.gelu(x)


def greedy(xa: mx.array, max_tokens: int = 120) -> list[int]:
    kv = None
    tokens = mx.array([PROMPT])
    out: list[int] = []
    for _ in range(max_tokens):
        logits, kv, _ = model.decoder(tokens, xa, kv_cache=kv)
        nxt = int(mx.argmax(logits[0, -1]).item())
        if nxt == tok.eot:
            break
        out.append(nxt)
        tokens = mx.array([[nxt]])
    return out


def timed(f, reps=5):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(f())
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def main(paths: list[str]) -> None:
    audios = [wa.load_audio(p) for p in paths]
    a = np.array(audios[0])
    print("== speed (median ms)")
    for secs in (5, 10, 15, 30):
        chunk = a[: secs * SR]
        full = timed(lambda: encode(chunk, trimmed=False))
        trim = timed(lambda: encode(chunk, trimmed=True))
        print(f"window {secs:2d}s: encoder full={full:6.1f}  trimmed={trim:6.1f}")
    xa = encode(a[: 10 * SR], trimmed=True)
    draft = mx.array([PROMPT + list(range(400, 440))])
    print(f"decoder verify pass (4 prompt + 40 draft tokens): {timed(lambda: model.decoder(draft, xa)[0]):.1f} ms")
    one = mx.array([PROMPT])
    print(f"decoder single pass (prompt only): {timed(lambda: model.decoder(one, xa)[0]):.1f} ms")

    print("== accuracy on 10 s chunks")
    ref = normalise(REFERENCE)
    for trimmed in (False, True):
        wers = []
        for audio in audios:
            arr = np.array(audio)
            text = " ".join(tok.decode(greedy(encode(arr[i:i + 10 * SR], trimmed)))
                            for i in range(0, len(arr), 10 * SR))
            wers.append(jiwer.wer(ref, normalise(text)))
        print(f"{'trimmed' if trimmed else 'full 30s'}: WER per file={[round(w * 100, 1) for w in wers]} "
              f"mean={statistics.mean(wers) * 100:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
