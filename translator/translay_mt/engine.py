"""Translay エンジン: 音を受け取り、英語を聞き取り、各言語へ訳し、出来事を流す（JSON 行・stdio）。

    uv run --project translator python -m translay_mt.engine \
        --recogniser whisper --asr-model models/whisper-large-v3-turbo \
        --mt-model models/plamo-3-translate-0.8b-MLX-4bit \
        --targets es,ja --language English|auto [--terms-file terms.txt] [--trace]

All GPU work — recognition (Qwen3-ASR) and translation (PLaMo 3) — happens on this process's main
thread, one after the other, so the two never compete for the GPU. A reader thread only collects
audio.

In  (stdin):   {"type": "audio", "pcm16": "<base64 little-endian int16, 16 kHz mono>"}
               {"type": "terms", "terms": ["Redis", "orchestrator", …]}
               {"type": "language", "language": "English" | "auto"}
               {"type": "targets", "targets": ["ja"]}  — translate into these instead ([] = none)
               {"type": "end"}                       — the source stopped: finish what is pending
Out (stdout):  {"type": "status", "text": "…"}         — start-up progress, for the user to see
               {"type": "ready", …}
               {"type": "speaking", "line": 3, "en": "…", "lang": "en"}   — "en" is the spoken text
               {"type": "finished", "line": 3, "en": "…", "lang": "en"}     (in whatever language)
               {"type": "translation", "line": 3, "final": false, "texts": {"es": "…", "ja": "…"}}
               {"type": "pass", "ms": …, …}           — with --trace only

The loop, in the words of the spec:
  届いた音を取り込み → 聞き直して全文を更新 → 文を切り出す →
  話し途中の文は訳し直し、言い終えた文は確定訳する → 音が来る限り繰り返す
"""
from __future__ import annotations

import argparse
import base64
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from translay_mt.hearing import SAMPLE_RATE, RollingListener
from translay_mt.hearing_whisper import WhisperListener
from translay_mt.plamo_fast import load_plamo
from translay_mt.sentences import Finished, SentenceCutter, Speaking, Utterance
from translay_mt.translating import Interpreters, log

_END = object()
MLX_CACHE_LIMIT = 512 * 1024 * 1024


@dataclass
class Terms:
    words: list[str]


@dataclass
class Targets:
    codes: list[str]


@dataclass
class Language:
    name: str | None          # None = detect


# Every language Qwen3-ASR recognises (its config's support_languages) → the code PLaMo targets use.
LANGUAGE_CODES = {
    "Chinese": "zh", "English": "en", "Cantonese": "yue", "Arabic": "ar", "German": "de", "French": "fr",
    "Spanish": "es", "Portuguese": "pt", "Indonesian": "id", "Italian": "it", "Korean": "ko",
    "Russian": "ru", "Thai": "th", "Vietnamese": "vi", "Japanese": "ja", "Turkish": "tr", "Hindi": "hi",
    "Malay": "ms", "Dutch": "nl", "Swedish": "sv", "Danish": "da", "Finnish": "fi", "Polish": "pl",
    "Czech": "cs", "Filipino": "tl", "Persian": "fa", "Greek": "el", "Romanian": "ro", "Hungarian": "hu",
    "Macedonian": "mk",
}


def parse_language(value: str) -> str | None:
    """"auto" → None (detect); otherwise one of the recogniser's languages, by its English name."""
    name = value.strip()
    if name.lower() == "auto":
        return None
    if name not in LANGUAGE_CODES:
        raise ValueError(f"unsupported recognition language {name!r}; expected auto or one of {sorted(LANGUAGE_CODES)}")
    return name


class Inbox:
    """受付係: stdin から音と指示を受け取って貯める（MLX には触らない）。"""

    def __init__(self) -> None:
        self.items: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        try:
            self._read_lines()
        except Exception as e:          # surface it on the main thread instead of dying silently
            self.items.put(e)

    def _read_lines(self) -> None:
        for line in sys.stdin:
            if not line.strip():
                continue
            msg = json.loads(line)
            kind = msg["type"]
            if kind == "audio":
                pcm = np.frombuffer(base64.b64decode(msg["pcm16"]), dtype="<i2").astype(np.float32) / 32768.0
                self.items.put(pcm)
            elif kind == "terms":
                self.items.put(Terms([str(t) for t in msg["terms"]]))
            elif kind == "targets":
                self.items.put(Targets([str(t) for t in msg["targets"]]))
            elif kind == "language":
                self.items.put(Language(parse_language(str(msg["language"]))))
            elif kind == "end":
                self.items.put(_END)
            else:
                raise ValueError(f"unknown message type {kind!r}")
        self.items.put(_END)

    def take(self) -> tuple[np.ndarray, list, bool]:
        """Everything that has arrived (waiting for at least one item): audio, controls, ended?"""
        items = [self.items.get()]
        while True:
            try:
                items.append(self.items.get_nowait())
            except queue.Empty:
                break
        for i in items:
            if isinstance(i, Exception):
                raise i
        audio = [i for i in items if isinstance(i, np.ndarray)]
        controls = [i for i in items if isinstance(i, (Terms, Language, Targets))]
        pcm = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
        return pcm, controls, any(i is _END for i in items)


def emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


Listener = RollingListener | WhisperListener


class Engine:
    def __init__(self, listener: Listener, interpreters: Interpreters, trace: bool):
        self.listener = listener
        self.interpreters = interpreters
        self.cutter = SentenceCutter()
        self.line = 1
        self.trace = trace

    def run(self, inbox: Inbox) -> None:
        """音が来る限り: 取り込み → 聞き直し → 切り出し → 訳す"""
        while True:
            pcm, controls, ended = inbox.take()
            for c in controls:
                if isinstance(c, Terms):
                    self.listener.terms = c.words
                elif isinstance(c, Targets):
                    self.interpreters.translate_into(c.codes)
                    log(f"translating into: {','.join(c.codes) or '(nothing)'}")
                else:                          # a new language: what was pending is finished first
                    self._caption(self.cutter.flush())
                    self.listener.set_language(c.name)
                    self.cutter = SentenceCutter()
                    log(f"recognition language: {c.name or 'auto'}")
            if len(pcm):
                self.listener.hear(pcm)
                text = self.listener.listen()
                self._log_pass(pcm)
                self._report_pass()
                heard = self.listener.heard_seconds
                if isinstance(self.listener, WhisperListener):    # it commits text itself
                    self._caption(self.cutter.cut_stable(self.listener.committed, self.listener.window_text, heard))
                else:
                    self._caption(self.cutter.cut(text, heard))
            if ended:
                self._caption(self.cutter.flush())
                return

    def _caption(self, utterances: list[Utterance]) -> None:
        # Only the newest unfinished sentence is worth translating; finished ones always are.
        last_speaking = max((i for i, u in enumerate(utterances) if isinstance(u, Speaking)), default=-1)
        lang = LANGUAGE_CODES.get(self.listener.detected_language or "")
        for i, u in enumerate(utterances):
            if isinstance(u, Finished):
                emit({"type": "finished", "line": self.line, "en": u.english, "lang": lang})
                texts = self.interpreters.finish(u.english, lang)
                emit({"type": "translation", "line": self.line, "final": True, "texts": texts})
                self.interpreters.settle()
                self.line += 1
            else:
                emit({"type": "speaking", "line": self.line, "en": u.english, "lang": lang})
                if i == last_speaking:
                    texts = self.interpreters.revise(u.english, lang)
                    emit({"type": "translation", "line": self.line, "final": False, "texts": texts})

    def _log_pass(self, pcm: np.ndarray) -> None:
        """One line per pass on stderr (the app keeps it in ~/Library/Logs/Translay/engine.log)."""
        s = self.listener.last
        where = "skip(silent)" if s.skipped_silent else ("gpu" if s.encoded_on_gpu else "ok")
        if s.dropped_overflow:
            where += "+dropped-unagreed-audio"
        tail = self.listener.text[-50:].replace("\n", " ")
        log(f"pass heard={self.listener.heard_seconds:7.2f}s got={len(pcm) / SAMPLE_RATE:5.2f}s "
            f"peak={float(np.abs(pcm).max()):.3f} rms={float(np.sqrt((pcm ** 2).mean())):.4f} "
            f"{where} {s.seconds * 1000:5.0f}ms window={s.audio_sec:5.1f}s | …{tail}")

    def _report_pass(self) -> None:
        if not self.trace:
            return
        s = self.listener.last
        emit({"type": "pass", "ms": round(s.seconds * 1000, 1), "window": s.audio_sec,
              "encoded": s.encoded_sec, "prefilled": s.prefilled, "drafted": s.drafted,
              "accepted": s.accepted, "stepped": s.stepped, "heard": round(self.listener.heard_seconds, 2)})


def read_terms(path: str | None) -> list[str]:
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def warm(listener: Listener, interpreters: Interpreters) -> None:
    """Pay first-use kernel compilation before the first caption, then forget everything."""
    rng = np.random.default_rng(0)
    listener.hear((rng.standard_normal(2 * SAMPLE_RATE) * 0.01).astype(np.float32))
    listener.listen()
    listener.reset()
    listener.heard = 0          # the warm-up noise is not part of the session's audio clock
    for partial in ("Okay so", "Okay so let's get started."):
        interpreters.revise(partial, "en")
    interpreters.finish("Okay so let's get started.", "en")
    interpreters.settle()
    interpreters.reset()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recogniser", required=True, choices=["qwen", "whisper"],
                    help="qwen: Qwen3-ASR rolling 8 s blocks; whisper: Whisper with committed-segment trimming")
    ap.add_argument("--asr-model", required=True)
    ap.add_argument("--whisper-ane-encoder",
                    help="with --recogniser whisper: WhisperKit AudioEncoder.mlmodelc, to encode on the Neural Engine")
    ap.add_argument("--mt-model", required=True)
    ap.add_argument("--targets", required=True, help="comma-separated target languages, e.g. es,ja")
    ap.add_argument("--language", required=True, help='recognition language, e.g. English, or "auto" to detect')
    ap.add_argument("--terms-file", help="one registered word per line")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        raise SystemExit("--targets must name at least one language")

    # MLX keeps freed GPU buffers for reuse with no practical bound. Every pass allocates buffers of
    # new shapes, so without a cap the cache grew to 10 GB in 20 s and pushed the Mac into swap.
    mx.set_cache_limit(MLX_CACHE_LIMIT)
    t0 = time.perf_counter()
    inbox = Inbox()        # start collecting audio now; it is all processed once the models are up
    terms, language = read_terms(args.terms_file), parse_language(args.language)

    def status(text: str) -> None:
        emit({"type": "status", "text": text})

    if args.recogniser == "whisper" and args.whisper_ane_encoder:
        status("聞き取りモデルを Neural Engine 用に準備中（キャッシュが無いと約1分）")
    else:
        status("聞き取りモデルを読み込み中")
    listener: Listener = (WhisperListener.load(args.asr_model, terms=terms, language=language,
                                               ane_encoder=args.whisper_ane_encoder, notify=status)
                          if args.recogniser == "whisper"
                          else RollingListener.load(args.asr_model, terms=terms, language=language))
    status("翻訳モデルを読み込み中")
    mt_model, mt_tokenizer = load_plamo(args.mt_model)
    interpreters = Interpreters(mt_model, mt_tokenizer, targets, trace=args.trace)
    status("準備運転中")
    warm(listener, interpreters)
    emit({"type": "ready", "targets": targets, "seconds": round(time.perf_counter() - t0, 2)})
    log(f"ready in {time.perf_counter() - t0:.1f}s (recogniser={args.recogniser}, asr={args.asr_model}, mt={args.mt_model})")
    Engine(listener, interpreters, args.trace).run(inbox)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
