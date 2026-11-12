"""聞き取り: Qwen3-ASR を回し続け、話された英語の全文を常に最新に保つ。

A continuous loop of recognition passes over a bounded audio window. Nothing already computed is
thrown away between passes:

  * ENCODER. Qwen3-ASR's audio encoder attends within 8-second blocks only (block-diagonal), so a
    block's encoder output never changes once the block is full. Full blocks are encoded once and
    cached; each pass encodes only the partial tail block. (Verified: per-block encoding of the
    same mel gives word-identical transcripts to whole-audio encoding.)
  * MEL. Whisper-style log-mel clamps at (max − 8) over the WHOLE input, which would make a
    block's features depend on future audio. Here the reference max is the running max of audio
    heard so far, and each block's STFT is computed with half a window of real context on both
    sides, so a cached block stays valid.
  * DECODER KV. The prompt is [context][audio of cached blocks][audio of tail][answer prefix]. The
    KV cache is kept between passes and rewound only to where the prompt changed — normally the
    start of the tail block — so the cached blocks' audio is never prefilled again.
  * PREVIOUS TRANSCRIPT AS DRAFT. The last pass's output is verified in one forward pass; only the
    tokens after the first disagreement are decoded one by one.
  * WINDOW. At most `window_blocks` blocks. When it fills, the oldest block leaves; its text is
    frozen ("committed") one block later, while it was still interior (see `_slide`), the scheme
    measured on the Even G2 work (qwen_rolling.py) to avoid inventing words at the cut.

Spec in one line: 「直近の音だけを聞き直し続け、確かめ済みの部分は計算し直さない」.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import numpy as np
from mlx_qwen3_asr import audio as qa_audio
from mlx_qwen3_asr.generate import _detect_repetition
from mlx_qwen3_asr.load_models import load_model
from mlx_qwen3_asr.tokenizer import Tokenizer, parse_asr_output

SAMPLE_RATE = 16_000
BLOCK_SEC = 8                       # = the encoder's attention window (n_window_infer = 800 frames)
BLOCK = BLOCK_SEC * SAMPLE_RATE
EDGE = qa_audio.N_FFT // 2          # STFT context taken from neighbouring audio
HOP = qa_audio.HOP_LENGTH
EOS = (151643, 151645)
SEAM_TRIM = 8
MAX_NEW = 160
CACHE_LEN = 1536                    # context + 3 blocks of audio (312) + answer + draft, with room


def _ints(x: mx.array) -> list[int]:
    """Evaluate an integer array to a flat Python list."""
    return [int(v) for v in np.array(x).reshape(-1)]


# Scripts written without spaces between words: each character is its own unit.
_UNSPACED = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff01-\uff60\u3000-\u303f"
_UNIT = re.compile(f"[{_UNSPACED}]|[^\\s{_UNSPACED}]+")
_IS_UNSPACED = re.compile(f"[{_UNSPACED}]")


def _units(text: str) -> list[str]:
    """Words for spaced scripts, single characters for Japanese/Chinese."""
    return _UNIT.findall(text)


def _unspaced(ch: str) -> bool:
    return bool(_IS_UNSPACED.match(ch))


def _rejoin(units: list[str]) -> str:
    out = ""
    for u in units:
        if out and not (_unspaced(out[-1]) or _unspaced(u[0])):
            out += " "
        out += u
    return out


def _spaced(a: str, b: str) -> str:
    """Join two pieces of text: one space between words, none next to Japanese/Chinese."""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return a or b
    return a + b if _unspaced(a[-1]) or _unspaced(b[0]) else f"{a} {b}"


def _norm(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalnum())


def _continue_after(prefix: str, text: str) -> str | None:
    """The part of `text` that follows where `prefix`'s last units appear in it, if they do.

    Block seams are matched on WORDS (characters for Japanese/Chinese): the prefix's last few were
    recognised while their block had audio on both sides, so they reappear verbatim next window.
    """
    pu = [_norm(u) for u in _units(prefix)]
    tu = _units(text)
    tn = [_norm(u) for u in tu]
    if not pu:
        return None
    sizes = (8, 6, 4) if _unspaced(prefix.strip()[-1]) else (5, 4, 3)
    for k in sizes:
        if len(pu) < k:
            continue
        key = pu[-k:]
        for i in range(len(tn) - k, -1, -1):
            if tn[i:i + k] == key:
                return _rejoin(tu[i + k:])
    return None


def _join(before: str, after: str) -> str:
    """`after` minus a short repeat of `before`'s ending (block seams are token-index estimates)."""
    for n in range(min(SEAM_TRIM, len(before), len(after)), 0, -1):
        if before.endswith(after[:n]):
            return after[n:]
    return after


@dataclass
class PassStats:
    audio_sec: float = 0.0
    encoded_sec: float = 0.0        # audio actually run through the encoder this pass
    prefilled: int = 0              # prompt tokens prefilled this pass
    drafted: int = 0
    accepted: int = 0
    stepped: int = 0                # tokens decoded one at a time
    seconds: float = 0.0
    encode_seconds: float = 0.0
    decode_seconds: float = 0.0
    encoded_on_gpu: bool = False    # Whisper: the ANE encoder was unavailable for this pass
    skipped_silent: bool = False    # Whisper: no speech energy in the buffer, pass skipped
    reheard_without_prompt: bool = False   # Whisper: the output copied the previous text; decoded again without it
    dropped_overflow: bool = False  # Whisper: nothing committable as the window filled; old audio dropped


class MelStream:
    """Log-mel features whose values never depend on audio that has not been heard yet."""

    def __init__(self) -> None:
        self.reference = -np.inf     # running max of log-mel over everything heard

    def block(self, audio: np.ndarray, start: int, end: int) -> mx.array:
        """Normalised log-mel (n_mels, frames) for audio[start:end], with real STFT context."""
        lo, hi = max(0, start - EDGE), min(len(audio), end + EDGE)
        freqs = qa_audio.stft(mx.array(audio[lo:hi]), qa_audio._hann_window(qa_audio.N_FFT),
                              nperseg=qa_audio.N_FFT, noverlap=qa_audio.N_FFT - HOP)
        mel = qa_audio.mel_filters(128) @ (mx.abs(freqs) ** 2).T
        log_spec = mx.log10(mx.maximum(mel, 1e-10))
        first = (start - lo) // HOP
        log_spec = log_spec[:, first:first + (end - start) // HOP]
        self.reference = max(self.reference, float(np.array(mx.max(log_spec))))
        return (mx.maximum(log_spec, self.reference - 8.0) + 4.0) / 4.0


@dataclass
class RollingListener:
    commits_text_itself = False     # the engine finishes sentences by agreement over the whole text
    model: Any
    tokenizer: Tokenizer
    language: Optional[str] = "English"     # None = let the model detect the language
    terms: list[str] = field(default_factory=list)
    window_blocks: int = 3
    dtype: mx.Dtype = mx.float16

    def __post_init__(self) -> None:
        self.window = np.zeros(0, dtype=np.float32)
        self.mel = MelStream()
        self.block_features: list[mx.array] = []    # encoder output of each FULL block in window
        self.committed = ""                          # text whose audio has left the window
        self.window_text = ""
        self.draft: list[int] = []
        self.splits: list[int] = []                  # token index of each block boundary in draft
        self.prev_tokens = 0
        self.prev_audio = 0
        self.pending: Optional[str] = None           # text of the window's first block
        self.header: list[int] = []                  # "language English<asr_text>" tokens
        self.detected_language: Optional[str] = self.language
        self.cache: Any = None
        self.cache_prompt: list[int] = []            # tokens resident in self.cache
        self.prefilled_blocks = 0                    # full blocks whose audio is in the cache
        self.heard = 0                               # samples heard in this session
        self.last = PassStats()

    @classmethod
    def load(cls, path: str, **kwargs) -> RollingListener:
        model, _config = load_model(path)
        tokenizer = Tokenizer(path)
        return cls(model=model, tokenizer=tokenizer, **kwargs)

    # ---- vocabulary of the spec -------------------------------------------------------------

    @property
    def text(self) -> str:
        """話された英語の全文（確定済み＋窓の中の最新の聞き取り）"""
        return _spaced(self.committed, self.window_text)

    @property
    def heard_seconds(self) -> float:
        return self.heard / SAMPLE_RATE

    def hear(self, pcm: np.ndarray) -> None:
        """音を取り込む（窓が溢れたら古いブロックを手放す）"""
        self.heard += len(pcm)
        self.window = np.concatenate([self.window, pcm.astype(np.float32)])
        while len(self.window) > self.window_blocks * BLOCK:
            self._slide()

    def listen(self) -> str:
        """窓の音を聞き直して全文を更新する（1パス）"""
        if len(self.window) < SAMPLE_RATE // 4:
            return self.text
        t0 = time.perf_counter()
        stats = PassStats(audio_sec=len(self.window) / SAMPLE_RATE)
        features = self._features(stats)
        tokens = self._decode(features, stats)
        self.header = tokens[: self._header_len(tokens)]
        lang, _ = parse_asr_output(self.tokenizer.decode(tokens), user_language=self.language)
        self.detected_language = lang or self.language
        self.window_text = self._visible(tokens)
        self.draft = tokens
        self._record_splits(tokens)
        stats.seconds = time.perf_counter() - t0
        self.last = stats
        return self.text

    def reset(self) -> None:
        heard = self.heard
        self.__post_init__()
        self.heard = heard

    def set_language(self, language: Optional[str]) -> None:
        """認識言語を変える（None は自動判定）。窓は聞き直しになる。"""
        self.language = language
        self.reset()

    # ---- encoder: cached full blocks + fresh tail --------------------------------------------

    def _features(self, stats: PassStats) -> mx.array:
        full = len(self.window) // BLOCK
        while len(self.block_features) < full:
            b = len(self.block_features)
            self.block_features.append(self._encode(b * BLOCK, (b + 1) * BLOCK))
            stats.encoded_sec += BLOCK_SEC
        parts = list(self.block_features)
        tail_start = full * BLOCK
        if len(self.window) - tail_start >= HOP * 2:
            parts.append(self._encode(tail_start, len(self.window)))
            stats.encoded_sec += (len(self.window) - tail_start) / SAMPLE_RATE
        return mx.concatenate(parts, axis=1)

    def _encode(self, start: int, end: int) -> mx.array:
        mel = self.mel.block(self.window, start, end)
        features, _ = self.model.audio_tower(mel[None].astype(self.dtype), mx.array([mel.shape[1]]))
        return features

    # ---- decoder: KV reuse + speculative verification ----------------------------------------

    def _context(self) -> str:
        parts = []
        if self.terms:
            parts.append("Terms: " + ", ".join(self.terms))
        if self.committed:
            parts.append("Previously said: " + self.committed[-400:])
        return "\n".join(parts)

    def _decode(self, features: mx.array, stats: PassStats) -> list[int]:
        m = self.model
        prompt = self.tokenizer.build_prompt_tokens(
            n_audio_tokens=features.shape[1], language=self.language, context=self._context())
        n = len(prompt)
        if self.cache is None or n + MAX_NEW + 256 > self.cache.max_seq_len:
            self.cache = m.create_cache(max_seq_len=max(CACHE_LEN, n + MAX_NEW + 512))
            self.cache_prompt, self.prefilled_blocks = [], 0
        # Reusable prefix: identical tokens, and — because every audio placeholder has the same id —
        # only up to the audio of blocks that were already FULL when they were prefilled.
        keep = 0
        for a, b in zip(self.cache_prompt, prompt):
            if a != b:
                break
            keep += 1
        audio_pos = [i for i, t in enumerate(prompt) if t == m.audio_token_id]
        stable = sum(f.shape[1] for f in self.block_features[: self.prefilled_blocks])
        if stable < len(audio_pos):
            keep = min(keep, audio_pos[stable])
        keep = min(keep, n - 1)
        self.cache.trim(self.cache.offset - keep)

        pos = mx.arange(keep, n)[None]
        skipped_audio = sum(1 for i in audio_pos if i < keep)
        logits = m.prefill(input_ids=mx.array([prompt[keep:]]), audio_features=features[:, skipped_audio:],
                           position_ids=mx.stack([pos, pos, pos], axis=1), cache=self.cache)
        stats.prefilled = n - keep
        self.prefilled_blocks = len(self.block_features)

        token = _ints(mx.argmax(logits, axis=-1))[-1]
        most = self._most_new_tokens()
        generated = [token]
        draft = self.draft
        aligned = bool(draft) and draft[0] == token
        while len(generated) < most:
            if token in EOS or _detect_repetition(generated):
                break
            proposal = (draft[len(generated):len(generated) + 256] if aligned
                        else self._realigned(generated, draft))
            proposal = proposal[:most - len(generated) - 1]
            at = n + len(generated) - 1
            if not proposal:
                p = mx.array([[[at]], [[at]], [[at]]]).reshape(1, 3, 1)
                token = _ints(mx.argmax(m.step(input_ids=mx.array([[token]]), position_ids=p,
                                               cache=self.cache), axis=-1))[-1]
                generated.append(token)
                stats.stepped += 1
                continue
            span = mx.arange(at, at + 1 + len(proposal))[None]
            verify = m.step_many(input_ids=mx.array([[token, *proposal]]),
                                 position_ids=mx.stack([span, span, span], axis=1), cache=self.cache)
            pred = _ints(mx.argmax(verify, axis=-1)[0])
            ok = 0
            while ok < len(proposal) and pred[ok] == proposal[ok]:
                ok += 1
            self.cache.trim(len(proposal) - ok)
            stats.drafted += len(proposal)
            stats.accepted += ok
            stop = False
            for t in proposal[:ok]:
                generated.append(t)
                if t in EOS or _detect_repetition(generated) or len(generated) >= most:
                    stop = True
                    break
            if stop:
                break
            token = pred[ok]
            generated.append(token)
            aligned = False
        # The cache now holds the prompt and every generated token except the last one emitted.
        self.cache_prompt = list(prompt) + generated[:-1]
        if generated and generated[-1] in EOS:
            generated.pop()
        return generated

    def _most_new_tokens(self) -> int:
        return MAX_NEW

    @staticmethod
    def _realigned(generated: list[int], draft: list[int], ngram: int = 2, span: int = 16) -> list[int]:
        """After a divergence: find the last generated n-gram in the old draft, propose what followed."""
        if len(generated) < ngram or not draft:
            return []
        key = generated[-ngram:]
        for i in range(len(draft) - ngram, -1, -1):
            if draft[i:i + ngram] == key:
                return draft[i + ngram:i + ngram + span]
        return []

    def _header_len(self, tokens: list[int]) -> int:
        """Number of leading tokens up to and including the "<asr_text>" marker."""
        for i in range(1, min(len(tokens), 12) + 1):
            if "<asr_text>" in self.tokenizer.decode(tokens[:i]):
                return i
        return 0

    # ---- window sliding (from qwen_rolling.py, seams matched on words) ----------------------

    def _record_splits(self, tokens: list[int]) -> None:
        filled = len(self.window) // BLOCK
        while len(self.splits) < filled:
            boundary = (len(self.splits) + 1) * BLOCK
            span = len(self.window) - self.prev_audio
            frac = (boundary - self.prev_audio) / span if span > 0 else 1.0
            frac = min(1.0, max(0.0, frac))
            self.splits.append(round(self.prev_tokens + (len(tokens) - self.prev_tokens) * frac))
        self.prev_tokens = len(tokens)
        self.prev_audio = len(self.window)

    def _slide(self) -> None:
        """Drop the oldest block. Its text is committed; the next block's text becomes `pending`.

        Both are cut from the text currently shown, by WORDS (characters for Japanese/Chinese), so nothing between them can be lost
        or doubled; the token-index split only estimates how many words each block holds.
        """
        h = len(self.header)
        cut = max(h, min(self.splits[0] if self.splits else len(self.draft), len(self.draft)))
        nxt = max(cut, self.splits[1] if len(self.splits) > 1 else len(self.draft))
        shown = _units(self.window_text)
        first = (_units(self.pending) if self.pending is not None
                 else _units(self._detokenize(self.draft[:cut])))
        first_n = min(len(first), len(shown))
        second_n = len(_units(self._detokenize(self.draft[cut:nxt])))
        self.committed = _spaced(self.committed, _rejoin(shown[:first_n]))
        self.pending = _rejoin(shown[first_n:first_n + second_n])
        self.draft = self.header + self.draft[cut:]
        self.window = self.window[BLOCK:]
        self.block_features = self.block_features[1:]
        self.splits = [s - cut + h for s in self.splits[1:]]
        self.prev_tokens = max(h, self.prev_tokens - cut + h)
        self.prev_audio = max(0, self.prev_audio - BLOCK)
        self.window_text = _rejoin(shown[first_n:])
        # The prompt's audio now starts one block later: nothing past the context is reusable.
        self.cache_prompt, self.prefilled_blocks = [], 0

    def _visible(self, tokens: list[int]) -> str:
        """The window's text as shown: the first block's settled words (`pending`), then whatever
        the new transcript says after them. The window's opening — cut mid-word — is never shown."""
        if self.pending is None:
            return self._detokenize(tokens)
        if not self.pending:
            cut = min(self.splits[0] if self.splits else len(self.header), len(tokens))
            return self._detokenize(tokens[cut:])
        rest = _continue_after(self.pending, self._detokenize(tokens))
        if rest is None:      # anchor not found: fall back to the token-index estimate
            cut = min(self.splits[0] if self.splits else 0, len(tokens))
            rest = _join(self.pending, self._detokenize(tokens[cut:]))
        return _spaced(self.pending, rest)

    def _detokenize(self, tokens: list[int]) -> str:
        _lang, text = parse_asr_output(self.tokenizer.decode(tokens), user_language=self.language)
        return self._strip_echo(text)

    def _strip_echo(self, text: str) -> str:
        """Qwen sometimes reads its own context back ("Terms: …", "Previously said: …"). Drop it."""
        for marker in ("Terms:", "Previously said:"):
            at = text.find(marker)
            if at >= 0:
                text = text[:at]
        return text.strip()
