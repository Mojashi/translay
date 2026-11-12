"""Whisper のエンコーダを Neural Engine で動かす係。別プロセスに隔離し、固まったら作り直す。

The ANE runs Whisper's encoder in ~600 ms per 30 s window (vs ~840 ms for MLX on the GPU) and
leaves the GPU to decoding and translation. But in live use an ANE request was seen to never
return — and Core ML holds the GIL while it waits, so nothing in the same process can time it out,
and the stuck request blocks the ANE for every other process too.

So the Core ML model lives in a child process. Mel features go in and encoder states come out
through shared memory; the parent waits at most `timeout` seconds. A child that does not answer is
killed (which releases the ANE) and respawned, and the caller encodes that pass on the GPU instead.
Captions never stop; at worst a few passes are slower.

Spec in one line: 「エンコードは Neural Engine に頼むが、返事がなければその係を作り直し、その間は GPU で代わる」.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory
from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np

N_MELS, N_FRAMES = 128, 3000
N_STATE, N_AUDIO_CTX = 1280, 1500


def _serve(path: str, conn: Any, mel_name: str, out_name: str) -> None:
    """Child process: load the Core ML encoder, then encode on request until told to stop."""
    import coremltools as ct
    mel_shm = shared_memory.SharedMemory(name=mel_name)
    out_shm = shared_memory.SharedMemory(name=out_name)
    mel = np.ndarray((N_MELS, N_FRAMES), dtype=np.float16, buffer=mel_shm.buf)
    out = np.ndarray((N_STATE, N_AUDIO_CTX), dtype=np.float16, buffer=out_shm.buf)
    parent = os.getppid()
    model = ct.models.CompiledMLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    conn.send("ready")
    while True:
        # Leave with the engine: if it is killed it cannot say "stop", and an orphan would keep the
        # model (1.2 GB) and the ANE busy.
        if not conn.poll(1.0):
            if os.getppid() != parent:
                break
            continue
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg != "encode":
            break
        y = model.predict({"melspectrogram_features": mel[None, :, None, :]})["encoder_output_embeds"]
        out[...] = y[0, :, 0, :]
        conn.send("done")


class IsolatedNeuralEngineEncoder:
    """A Core ML encoder in a child process, with a deadline per request.

    `__call__` returns the encoder output, or None when the ANE is unavailable for this pass (still
    starting, or just restarted after a hang) — the caller then encodes on the GPU.
    """

    def __init__(self, path: str, timeout: float = 4.0, startup_timeout: float = 240.0,
                 notify: Callable[[str], None] = lambda _: None):
        self.path = path
        self.timeout = timeout
        self.notify = notify
        self.ctx = mp.get_context("spawn")
        self.mel_shm = shared_memory.SharedMemory(create=True, size=N_MELS * N_FRAMES * 2)
        self.out_shm = shared_memory.SharedMemory(create=True, size=N_STATE * N_AUDIO_CTX * 2)
        self.mel = np.ndarray((N_MELS, N_FRAMES), dtype=np.float16, buffer=self.mel_shm.buf)
        self.out = np.ndarray((N_STATE, N_AUDIO_CTX), dtype=np.float16, buffer=self.out_shm.buf)
        self.restarts = 0
        self._spawn()
        self._wait_ready(startup_timeout)

    # ---- vocabulary of the spec -------------------------------------------------------------

    def __call__(self, mel: mx.array) -> Optional[mx.array]:
        """mel: (1, 3000, n_mels) → (1, 1500, d), or None if the GPU must do this pass."""
        return self.take() if self.ask(mel) else None

    def ask(self, mel: mx.array) -> bool:
        """Start encoding and return at once. False: the ANE is unavailable for this pass."""
        if not self._is_ready():
            return False
        self.mel[...] = np.array(mel[0]).astype(np.float16).T
        self.conn.send("encode")
        self.deadline = time.monotonic() + self.timeout
        return True

    def has_answered(self) -> bool:
        """The encoding asked for is there — or will never come (`take` then says so)."""
        return self.conn.poll() or time.monotonic() >= self.deadline

    def take(self) -> Optional[mx.array]:
        """The encoding asked for, waiting for it up to the deadline; None if the ANE hung."""
        if self.conn.poll(max(0.0, self.deadline - time.monotonic())):
            self.conn.recv()
            return mx.array(self.out.T[None].copy())
        self.restarts += 1
        self.notify(f"Neural Engine が {self.timeout:.0f} 秒応答しないため作り直します（その間は GPU で聞き取り）")
        self._kill()
        self._spawn()
        return None

    def close(self) -> None:
        try:
            self.conn.send("stop")
        except (BrokenPipeError, OSError):
            pass
        self._kill()
        for shm in (self.mel_shm, self.out_shm):
            shm.close()
            shm.unlink()

    # ---- mechanics --------------------------------------------------------------------------

    def _spawn(self) -> None:
        self.conn, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_serve, args=(self.path, child, self.mel_shm.name, self.out_shm.name),
                                     daemon=True)
        self.proc.start()
        self.ready = False

    def _kill(self) -> None:
        if self.proc.is_alive():
            self.proc.kill()
        self.proc.join(timeout=2)

    def _is_ready(self) -> bool:
        if not self.ready:
            if not self.proc.is_alive():                     # died while loading: try again
                self.notify("Neural Engine の係が落ちたので作り直します")
                self._spawn()
                return False
            if self.conn.poll():
                self.ready = self.conn.recv() == "ready"
                if self.ready and self.restarts:
                    self.notify("Neural Engine に戻りました")
        return self.ready

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._is_ready():
                return
            time.sleep(0.05)
        raise RuntimeError(f"Neural Engine encoder did not start within {timeout:.0f} s: {self.path}")
