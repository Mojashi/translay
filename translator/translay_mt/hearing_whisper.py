"""聞き取り（Whisper 版）: 確定した位置からの音だけを聞き直し続け、確かめ済みの区間は捨てる。

Whisper large-v3-turbo on MLX, in the whisper_streaming manner but tighter:

  * BUFFER. Only audio since the last committed segment is recognised (≤ 30 s, Whisper's window).
  * SEGMENTS WITH TIMES. Decoding keeps Whisper's timestamp tokens, so each pass yields segments
    with start/end times inside the buffer. A closed segment that came out identically in two
    consecutive passes and ends clear of the buffer's edge is committed — and its audio is cut
    away. Nothing committed is ever recognised again, and there are no block seams to stitch.
  * ONE SENTENCE, ONE SEGMENT. Whisper is made to close the segment at every sentence end
    (SentenceEndsCloseSegment), and between words once a segment has run long without one
    (LongSegmentsCloseBetweenWords); otherwise it may write the whole window as a single segment
    that can never be committed.
  * DECODING RULES. Whisper's own logit rules are applied (timestamp grammar, no non-speech
    symbols), and a decode that falls into repeating itself keeps only the first repeat.
  * PREVIOUS PASS AS DRAFT. The last pass's tokens (re-timed after a cut) are verified in the same
    decoder forward that prefills the prompt; only tokens after a disagreement are decoded singly.
    The decoder attends to the audio at every layer, so its KV cannot survive a new pass — but it is
    4 layers, so prompt + draft in one forward costs tens of ms. The encoder is the real cost.
  * PROMPT. Committed text (and registered terms) go in as Whisper's previous-text prompt.
  * SILENCE. Passes over audio with no speech energy are skipped, and segments whose audio is
    silent are dropped — Whisper invents "Thank you." from silence otherwise.

Spec in one line: 「確定していない音だけを聞き直し、2回続けて同じだった区間は確定して音ごと手放す」.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_whisper import audio as wa
from mlx_whisper.load_models import load_model
from mlx_whisper.tokenizer import LANGUAGES, get_tokenizer

from translay_mt.hearing import PassStats, _ints, _rejoin, _spaced, _units, _unspaced
from translay_mt.neural_engine import IsolatedNeuralEngineEncoder

SAMPLE_RATE = 16_000
SECONDS_PER_TIMESTAMP = 0.02
COMMIT_MARGIN = 1.0          # a segment must end this far before the buffer's edge to be committed
FORCE_COMMIT_AT = 24.0       # buffer length at which closed segments are committed regardless
KEEP_AFTER_OVERFLOW = 8.0    # when nothing can be committed at that length: keep only this much audio
MAX_TOKENS = 200
PROMPT_TOKENS = 160
SILENCE_CLOSES = 2.0         # this long without a voice: open segments count as closed
SPEECH_RMS = 0.008           # frame RMS above which a 30 ms frame counts as speech


def _norm(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


@dataclass
class Segment:
    start: float
    end: Optional[float]      # None: still open at the end of the decode
    tokens: list[int]


class DecoderPass:
    """Whisper's decoder with a trimmable self-attention cache, for one audio encoding."""

    def __init__(self, decoder: Any, audio_features: mx.array):
        self.d = decoder
        self.k: list[Optional[mx.array]] = [None] * len(decoder.blocks)
        self.v: list[Optional[mx.array]] = [None] * len(decoder.blocks)
        self.cross = [(b.cross_attn.key(audio_features), b.cross_attn.value(audio_features))
                      for b in decoder.blocks]
        self.length = 0
        self.heads = decoder.blocks[0].attn.n_head

    def _heads(self, x: mx.array) -> mx.array:
        b, n, s = x.shape
        return x.reshape(b, n, self.heads, s // self.heads).transpose(0, 2, 1, 3)

    def _attend(self, q: mx.array, k: mx.array, v: mx.array, causal: bool) -> mx.array:
        q, k, v = self._heads(q), self._heads(k), self._heads(v)
        scale = q.shape[-1] ** -0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal" if causal else None)
        b, h, n, d = out.shape
        return out.transpose(0, 2, 1, 3).reshape(b, n, h * d)

    def feed(self, tokens: list[int], logits_from: int) -> list[int]:
        """Append tokens; plain greedy predictions for positions >= logits_from (no timestamp rules)."""
        return _ints(mx.argmax(self.logits(tokens, logits_from), axis=-1)[0])

    def logits(self, tokens: list[int], logits_from: int) -> mx.array:
        """Append tokens; logits (1, n, vocab) for positions >= logits_from."""
        d = self.d
        n = len(tokens)
        x = d.token_embedding(mx.array([tokens])) + d.positional_embedding[self.length:self.length + n]
        for i, blk in enumerate(d.blocks):
            h = blk.attn_ln(x)
            k, v = blk.attn.key(h), blk.attn.value(h)
            past_k, past_v = self.k[i], self.v[i]
            if past_k is not None and past_v is not None:
                k = mx.concatenate([past_k, k], axis=1)
                v = mx.concatenate([past_v, v], axis=1)
            self.k[i], self.v[i] = k, v
            x = x + blk.attn.out(self._attend(blk.attn.query(h), k, v, causal=n > 1))
            ck, cv = self.cross[i]
            x = x + blk.cross_attn.out(self._attend(blk.cross_attn.query(blk.cross_attn_ln(x)), ck, cv, False))
            x = x + blk.mlp2(nn.gelu(blk.mlp1(blk.mlp_ln(x))))
        self.length += n
        return d.token_embedding.as_linear(d.ln(x[:, logits_from:]))

    def trim(self, n: int) -> None:
        if n <= 0:
            return
        keep = self.length - n
        for i in range(len(self.k)):
            k, v = self.k[i], self.v[i]
            if k is not None and v is not None:
                self.k[i], self.v[i] = k[:, :keep], v[:, :keep]
        self.length = keep


class TimestampRules:
    """Whisper の時刻トークンの決まり（本家の ApplyTimestampRules と同じ）。

    Without them greedy decoding happily picks <|notimestamps|> first and never emits a time —
    and with no times there are no closed segments, so nothing is ever committed.
      * the output starts with a timestamp (≤ 1 s); <|notimestamps|> and other specials never appear
      * timestamps come in pairs (end, start); after a pair comes text, after a lone one a timestamp or the end
      * timestamps never go back
      * a timestamp wins when all timestamps together are likelier than the best text token
    """

    def __init__(self, tok: Any, max_initial: float = 1.0):
        self.tb, self.eot = tok.timestamp_begin, tok.eot
        self.max_initial = int(round(max_initial / SECONDS_PER_TIMESTAMP))

    def apply(self, row: np.ndarray, prefix: list[int]) -> None:
        tb, eot = self.tb, self.eot
        row[eot + 1:tb] = -np.inf                         # sot, languages, tasks, <|notimestamps|>, …
        if not prefix:
            row[:tb] = -np.inf
            row[tb + self.max_initial + 1:] = -np.inf
            return
        last_ts = prefix[-1] >= tb
        pen_ts = len(prefix) < 2 or prefix[-2] >= tb
        if last_ts:
            if pen_ts:
                row[tb:] = -np.inf
            else:
                row[:eot] = -np.inf
        stamps = [t for t in prefix if t >= tb]
        if stamps:
            floor = stamps[-1] if (last_ts and not pen_ts) else stamps[-1] + 1
            row[tb:floor] = -np.inf
        finite = row[np.isfinite(row)]
        if finite.size:
            logp = row - (finite.max() + np.log(np.exp(finite - finite.max()).sum()))
            ts = logp[tb:][np.isfinite(logp[tb:])]
            text = logp[:tb][np.isfinite(logp[:tb])]
            if ts.size and text.size:
                ts_mass = ts.max() + np.log(np.exp(ts - ts.max()).sum())
                if ts_mass > text.max():
                    row[:tb] = -np.inf


class NoNonSpeechSymbols:
    """声でない記号を出させない（本家の SuppressTokens と同じ一覧）。

    " -" (a dash opening a word), "♪", brackets, … Without this the decoder slips into subtitle
    dialogue style — "- Okay. - Okay. - Okay." — and, fed back as the prompt, stays in it.
    """

    def __init__(self, tok: Any):
        self.symbols = np.array(sorted(tok.non_speech_tokens))

    def apply(self, row: np.ndarray, prefix: list[int]) -> None:
        row[self.symbols] = -np.inf


class SentenceEndsCloseSegment:
    """文が終わったら、そこで必ず区間を閉じさせる（時刻を打たせる）。

    Left alone, Whisper is free to write a whole 20 s window as ONE segment whose only end is the
    buffer's edge. Such a segment is never clear of the edge, so nothing in it could be committed
    however many sentences it held — the window filled, and the unagreed audio was thrown away with
    its text. With a timestamp after every sentence, each sentence is a segment that can be
    committed on its own.

    A sentence has ended when the last token ends in . ? ! and Whisper's next text choice starts a
    new word or ends the output (so "3.5" and "example.com" are not split), or ends in 。？！.
    """

    def __init__(self, words: Words):
        self.words = words

    def apply(self, row: np.ndarray, prefix: list[int]) -> None:
        w = self.words
        if not prefix or not w.is_text(prefix[-1]) or not w.ends_sentence[prefix[-1]]:
            return
        nxt = w.next_text(row)
        if nxt is not None and (nxt == w.eot or w.starts_word[nxt] or w.unspaced[prefix[-1]]):
            w.only_timestamps(row)


class LongSegmentsCloseBetweenWords:
    """文が終わらないまま長くなったら、語の切れ目で区間を閉じさせる。

    Whisper sometimes writes without any punctuation ("the goal is simple help japan stabilize its
    currency but …"), and some speakers simply do not stop. With no sentence end nothing would be
    committed until the window overflowed. So a segment that has run for `at_clause` tokens closes
    at the next , ; : 、 and one that has run for `anywhere` tokens closes before the next word
    (speech is about 4 tokens a second).
    """

    def __init__(self, words: Words, at_clause: int = 16, anywhere: int = 24):
        self.words, self.at_clause, self.anywhere = words, at_clause, anywhere

    def apply(self, row: np.ndarray, prefix: list[int]) -> None:
        w = self.words
        if not prefix or not w.is_text(prefix[-1]):
            return
        run = 0
        for t in reversed(prefix):
            if not w.is_text(t):
                break
            run += 1
        if run < self.at_clause or (run < self.anywhere and not w.ends_clause[prefix[-1]]):
            return
        nxt = w.next_text(row)
        if nxt is not None and nxt != w.eot and (w.starts_word[nxt] or w.unspaced[prefix[-1]]):
            w.only_timestamps(row)


class Words:
    """語彙の見取り図: 各トークンが文や節を終えるか、語を始めるか（決まりが共有する）。"""

    def __init__(self, tok: Any):
        self.tb, self.eot = tok.timestamp_begin, tok.eot
        pieces = [tok.decode([t]) for t in range(tok.eot)]
        last = [p.rstrip()[-1:] for p in pieces]
        self.ends_sentence = np.array([c in tuple(".?!。？！") for c in last])
        self.ends_clause = np.array([c in tuple(",;:、，；：") for c in last])
        self.unspaced = np.array([bool(c) and _unspaced(c) for c in last])      # Japanese/Chinese: no spaces
        self.starts_word = np.array([p[:1] == " " for p in pieces])

    def is_text(self, token: int) -> bool:
        return token < self.eot

    def next_text(self, row: np.ndarray) -> Optional[int]:
        """Whisper's choice among text tokens and the end — None if a timestamp is not possible
        here anyway, or no text is."""
        if not np.isfinite(row[:self.eot + 1]).any() or not np.isfinite(row[self.tb:]).any():
            return None
        return int(row[:self.eot + 1].argmax())

    def only_timestamps(self, row: np.ndarray) -> None:
        row[:self.tb] = -np.inf


class DecodingRules:
    """次のトークンの選び方: 決まりを順に当てはめ、残った中で最も確からしいものを選ぶ。"""

    def __init__(self, *rules: Any):
        self.rules = rules

    def choose(self, row: np.ndarray, prefix: list[int]) -> int:
        row = row.astype(np.float32, copy=True)
        for rule in self.rules:
            rule.apply(row, prefix)
        return int(row.argmax())


@dataclass
class WhisperListener:
    model: Any
    language: Optional[str] = "English"      # None = detect
    terms: list[str] = field(default_factory=list)
    ane_encoder: Optional[IsolatedNeuralEngineEncoder] = None   # None: encode with MLX on the GPU

    def __post_init__(self) -> None:
        self.tok = get_tokenizer(multilingual=True, num_languages=self.model.num_languages, task="transcribe")
        self.buffer = np.zeros(0, dtype=np.float32)
        self.committed = ""
        self.draft: list[int] = []               # previous pass's output tokens (buffer-relative times)
        self.previous: list[Segment] = []        # previous pass's segments
        self.window_text = ""
        self.heard = 0
        self.detected_language: Optional[str] = self.language
        self.last = PassStats()
        words = Words(self.tok)
        self.rules = DecodingRules(NoNonSpeechSymbols(self.tok), TimestampRules(self.tok),
                                   SentenceEndsCloseSegment(words), LongSegmentsCloseBetweenWords(words))

    @classmethod
    def load(cls, path: str, ane_encoder: Optional[str] = None,
             notify: Callable[[str], None] = lambda _: None, **kwargs) -> WhisperListener:
        """`ane_encoder`: a WhisperKit AudioEncoder.mlmodelc of the same model, to encode on the ANE
        (in a supervised child process; see neural_engine.py)."""
        encoder = IsolatedNeuralEngineEncoder(ane_encoder, notify=notify) if ane_encoder else None
        return cls(model=load_model(path), ane_encoder=encoder, **kwargs)

    # ---- vocabulary of the spec (same as RollingListener) ------------------------------------

    @property
    def text(self) -> str:
        return _spaced(self.committed, self.window_text)

    @property
    def heard_seconds(self) -> float:
        return self.heard / SAMPLE_RATE

    def hear(self, pcm: np.ndarray) -> None:
        """音を取り込む"""
        self.heard += len(pcm)
        self.buffer = np.concatenate([self.buffer, pcm.astype(np.float32)])

    def listen(self) -> str:
        """確定していない音を聞き直し、2回続けて同じだった区間を確定する（1パス）"""
        t0 = time.perf_counter()
        stats = PassStats(audio_sec=len(self.buffer) / SAMPLE_RATE)
        self.last = stats
        if not self._has_speech(self.buffer):
            self._drop_silence()
            stats.skipped_silent = True
            self.last = stats
            return self.text
        audio_features = self._encode(stats)
        tokens = self._decode(audio_features, stats)
        segments = [s for s in self._segments(tokens) if self._segment_has_speech(s)]
        self._commit(segments)
        stats.seconds = time.perf_counter() - t0
        self.last = stats
        return self.text

    def reset(self) -> None:
        heard = self.heard
        self.__post_init__()
        self.heard = heard

    def set_language(self, language: Optional[str]) -> None:
        self.language = language
        self.reset()

    # ---- encoder ------------------------------------------------------------------------------

    def _encode(self, stats: PassStats) -> mx.array:
        padded = np.zeros(wa.N_SAMPLES, dtype=np.float32)
        n = min(len(self.buffer), wa.N_SAMPLES)
        padded[:n] = self.buffer[:n]
        mel = wa.log_mel_spectrogram(padded, n_mels=self.model.dims.n_mels)[None, : wa.N_FRAMES]
        stats.encoded_sec = n / SAMPLE_RATE
        if self.ane_encoder is not None:
            features = self.ane_encoder(mel)
            if features is not None:
                return features.astype(mx.float16)
            stats.encoded_on_gpu = True          # the ANE is restarting: this pass on the GPU
        return self.model.encoder(mel.astype(mx.float16))

    # ---- decoder: prompt + previous pass as draft, in one forward ----------------------------

    def _prompt(self) -> list[int]:
        text = self.committed[-600:]
        if self.terms:
            text = _spaced(", ".join(self.terms) + ".", text)
        return self.tok.encode(" " + text.strip())[-PROMPT_TOKENS:] if text.strip() else []

    def _decode(self, audio_features: mx.array, stats: PassStats) -> list[int]:
        tok = self.tok
        dec = DecoderPass(self.model.decoder, audio_features)
        prev = self._prompt()
        head = ([tok.sot_prev] + prev) if prev else []
        prompt = head + self._sot_sequence_after(dec, head)
        stats.prefilled = len(prompt)
        draft = self.draft[:MAX_TOKENS]
        out: list[int] = []
        ok, nxt = self._verify(dec.logits(prompt + draft, logits_from=len(prompt) - 1), out, draft)
        dec.trim(len(draft) - ok)
        stats.drafted, stats.accepted = len(draft), ok
        out = draft[:ok]
        while nxt != tok.eot and len(out) < MAX_TOKENS and self._loop_start(out) is None:
            out.append(nxt)
            proposal = self._realigned(out, self.draft)
            n, nxt = self._verify(dec.logits([nxt] + proposal, logits_from=0), out, proposal)
            dec.trim(len(proposal) - n)
            if proposal:
                stats.drafted += len(proposal)
                stats.accepted += n
            else:
                stats.stepped += 1
            out.extend(proposal[:n])
        out = self._without_loop(out)
        self.draft = out
        return out

    def _verify(self, logits: mx.array, before: list[int], proposal: list[int]) -> tuple[int, int]:
        """How many proposed tokens Whisper agrees with (row j follows before + proposal[:j]),
        and its own choice at the first disagreement (or after the last one)."""
        rows = np.array(logits[0].astype(mx.float32))
        for j in range(len(rows)):
            choice = self.rules.choose(rows[j], before + proposal[:j])
            if j == len(proposal) or choice != proposal[j]:
                return j, choice
        raise AssertionError("one logits row per proposed token plus one")

    def _sot_sequence_after(self, dec: DecoderPass, head: list[int]) -> list[int]:
        """<|sot|><|lang|><|transcribe|> — detecting the language first if it is not fixed."""
        tok = self.tok
        if self.language is not None:
            code = next((c for c, name in LANGUAGES.items() if name == self.language.lower()), None)
            if code is None:
                raise ValueError(f"Whisper does not know the language {self.language!r}")
        else:
            if head:
                dec.feed(head, logits_from=len(head) - 1)
            after_sot = np.array(dec.logits([tok.sot], logits_from=0)[0, -1].astype(mx.float32))
            dec.trim(len(head) + 1)
            # Only language tokens are candidates: over the whole vocabulary the likeliest token
            # is sometimes not a language at all, and the speech was then taken for English.
            languages = list(tok.all_language_tokens)
            code = tok.all_language_codes[int(after_sot[languages].argmax())]
        self.detected_language = LANGUAGES[code].title()
        return [tok.sot, tok.to_language_token(code), tok.transcribe]

    @staticmethod
    def _realigned(out: list[int], draft: list[int], ngram: int = 2, span: int = 16) -> list[int]:
        if len(out) < ngram or not draft:
            return []
        key = out[-ngram:]
        for i in range(len(draft) - ngram, -1, -1):
            if draft[i:i + ngram] == key:
                return draft[i + ngram:i + ngram + span]
        return []

    def _loop_start(self, tokens: list[int], period_max: int = 8, times: int = 4) -> Optional[int]:
        """同じ言葉の繰り返しに陥っていたら、2回目が始まる位置（無ければ None）。

        Whisper's failure on unclear audio: "the, the, the, the" / "Okay. Okay. Okay. Okay."
        Only the words are compared — each repeat carries different timestamps.
        """
        at = [i for i, t in enumerate(tokens) if t < self.tok.timestamp_begin]
        words = [tokens[i] for i in at]
        for p in range(1, period_max + 1):
            if len(words) >= p * times and all(words[-p:] == words[-p * (i + 1):len(words) - p * i]
                                                for i in range(1, times)):
                return at[len(words) - p * (times - 1)]
        return None

    def _without_loop(self, tokens: list[int]) -> list[int]:
        """繰り返しは1回ぶんだけ残す（残りは聞こえた言葉ではない）。"""
        start = self._loop_start(tokens)
        if start is None:
            return tokens
        kept = tokens[:start]
        begin = self.tok.timestamp_begin
        if len(kept) >= 2 and kept[-1] >= begin and kept[-2] >= begin:    # the dropped repeat's opening time
            kept.pop()
        return kept

    # ---- segments, agreement, commit ----------------------------------------------------------

    def _segments(self, tokens: list[int]) -> list[Segment]:
        begin = self.tok.timestamp_begin
        segments: list[Segment] = []
        start: Optional[float] = None
        text: list[int] = []
        for t in tokens:
            if t >= begin:
                at = (t - begin) * SECONDS_PER_TIMESTAMP
                if start is not None and text:
                    segments.append(Segment(start, at, text))
                    start, text = None, []
                else:
                    start = at
            else:
                if start is None:
                    start = 0.0
                text.append(t)
        if text:
            segments.append(Segment(start or 0.0, None, text))
        return segments

    def _text(self, tokens: list[int]) -> str:
        return self.tok.decode([t for t in tokens if t < self.tok.eot]).strip()

    def _commit(self, segments: list[Segment]) -> None:
        buffer_sec = len(self.buffer) / SAMPLE_RATE
        force = buffer_sec >= FORCE_COMMIT_AT
        fell_silent = buffer_sec - self._last_voice_at(self.buffer) >= SILENCE_CLOSES
        agreed = 0
        for i, seg in enumerate(segments):
            # Whisper stretches the last segment's end to the buffer's edge even through silence;
            # once the speaker has stopped, whatever was said is as closed as it will get.
            closed = fell_silent or (seg.end is not None and seg.end <= buffer_sec - COMMIT_MARGIN)
            same = i < len(self.previous) and self.previous[i].end is not None and \
                _norm(self._text(self.previous[i].tokens)) == _norm(self._text(seg.tokens))
            if closed and (same or force):
                agreed = i + 1
            else:
                break
        if agreed:
            done = segments[:agreed]
            for seg in done:
                self.committed = _spaced(self.committed, self._text(seg.tokens))
            cut = buffer_sec if done[-1].end is None else min(done[-1].end, buffer_sec)
            self.buffer = self.buffer[int(cut * SAMPLE_RATE):]
            segments = [self._shift(s, cut) for s in segments[agreed:]]
            self.draft = self._retime(self.draft, cut, sum(len(s.tokens) + 2 for s in done))
        if len(self.buffer) / SAMPLE_RATE >= FORCE_COMMIT_AT:
            segments = self._make_room()
        self.previous = segments
        self.window_text = _rejoin(_units(" ".join(self._text(s.tokens) for s in segments)))

    def _make_room(self) -> list[Segment]:
        """聞き直す窓があふれそうなのに確定できる区間が無い: 古い音を手放す。

        Whisper hears at most 30 s. Audio beyond that would never be heard, so the buffer must not
        grow past it — this was the "captions stop after a while" failure: a quiet or noisy input
        on which Whisper never closed a segment (it kept inventing "I'll talk about the food…")
        let the buffer grow to 100 s, and everything after the first 30 s went unheard forever.
        Text that never agreed is not committed: it is dropped with its audio.
        """
        self.buffer = self.buffer[-int(KEEP_AFTER_OVERFLOW * SAMPLE_RATE):]
        self.draft = []
        self.last.dropped_overflow = True
        return []

    @staticmethod
    def _shift(seg: Segment, by: float) -> Segment:
        return Segment(max(0.0, seg.start - by), None if seg.end is None else max(0.0, seg.end - by), seg.tokens)

    def _retime(self, tokens: list[int], by: float, drop: int) -> list[int]:
        """The draft after a cut: committed segments removed, timestamps moved back by `by`."""
        begin = self.tok.timestamp_begin
        shift = int(round(by / SECONDS_PER_TIMESTAMP))
        rest = tokens[drop:] if drop < len(tokens) else []
        return [max(begin, t - shift) if t >= begin else t for t in rest]

    # ---- silence ------------------------------------------------------------------------------

    @staticmethod
    def _frame_rms(audio: np.ndarray, frame: int = 480) -> np.ndarray:
        n = len(audio) // frame
        if n == 0:
            return np.zeros(0)
        return np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))

    def _has_speech(self, audio: np.ndarray) -> bool:
        rms = self._frame_rms(audio)
        return bool(len(rms)) and int((rms > SPEECH_RMS).sum()) >= 5      # ≥ 150 ms of voice

    def _last_voice_at(self, audio: np.ndarray) -> float:
        """Seconds into `audio` at which the last voiced frame ends (0 if there is none)."""
        voiced = np.nonzero(self._frame_rms(audio) > SPEECH_RMS)[0]
        return float((voiced[-1] + 1) * 480 / SAMPLE_RATE) if len(voiced) else 0.0

    def _segment_has_speech(self, seg: Segment) -> bool:
        end = seg.end if seg.end is not None else len(self.buffer) / SAMPLE_RATE
        a, b = int(seg.start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        return self._has_speech(self.buffer[a:b]) if b - a > 4800 else True

    def _drop_silence(self) -> None:
        """Nothing is being said: keep only the last second, so silence never piles up."""
        if not self.window_text and len(self.buffer) > SAMPLE_RATE:
            self.buffer = self.buffer[-SAMPLE_RATE:]
            self.draft, self.previous = [], []
