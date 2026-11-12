"""聞き取り（Qwen3-ASR）: 確定した文の音は手放し、まだ確定していない音だけを聞き直し続ける。

RollingListener keeps the last 16–24 s whatever was said, so every pass verifies ~18 s of text.
Here the window is what is still open, plus the last settled words as a foothold:

    window = [足場: 確定済みの文の終わり（文字はもう変わらない）][まだ確定していない音]

  * 確定 (settle). The first sentence of the open text is settled when it came out the same in two
    consecutive passes and the next word has begun — the rule SentenceCutter.cut applies, so
    captions finish exactly when they did before. Settled text never changes again.
  * 足場 (foothold). The window is cut a little BEFORE the settled sentence's last words, and those
    words stay in it. Qwen3-ASR gives no timestamps, so the cut is only an estimate; the foothold
    makes that harmless: each pass reads the window from its opening (possibly mid-word), and only
    what follows the foothold's last words is taken as new text.
  * When did a word begin? Each pass notes how much audio had been heard and how many words the
    whole text had. A word first seen in the pass at `heard` began before it; `CUT_MARGIN` earlier
    is taken as the cut.
  * 静けさ. If the open text has not changed for `QUIET_SETTLES` seconds of audio, it is settled as
    it stands and all but the last second of audio is dropped. Decided from the text alone, never
    from loudness: a distant voice is quiet but still produces words.
  * 繰り返し. The same short phrase four or more times in a row ("Oh, oh, oh, …", written on loud
    noise) is not speech and is dropped from what was heard.
  * あふれ. If nothing settles for `MAX_WINDOW` seconds (no punctuation comes), all but the last few
    words are settled at a clause break if there is one.

Encoder blocks are cached while the window's start stays put, as in RollingListener; a cut moves
the start, so the remaining few seconds are encoded again (cheap next to the decoder).

Spec in one line: 「確定した文の音は手放し、足場の続きだけを聞き直す」.
"""
from __future__ import annotations

import difflib
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .hearing import HOP, MAX_NEW, SAMPLE_RATE, PassStats, RollingListener, _norm, _rejoin, _spaced, _units, _unspaced
from .sentences import TERMINALS, _first_sentence_end

FOOTHOLD_WORDS = 6          # settled words kept in the window (12 characters for Japanese/Chinese)
CUT_MARGIN = 1.0            # seconds before a word was first seen at which it is assumed to begin
QUIET_SETTLES = 1.2         # seconds of audio with the open text unchanged → settled as it stands
KEPT_AFTER_QUIET = 1.0      # seconds of audio kept then (the last word ended before that)
KEPT_WHILE_IDLE = 2.0       # seconds kept while nothing is being said, so a slow start is not cut off
MAX_WINDOW = 20.0           # seconds; beyond it text is settled without waiting for a sentence end
OPEN_WORDS_KEPT = 8         # … except this many last words, which may still be revised
CLAUSE_BREAKS = ",;:、，；：)”\""
TOKENS_PER_SECOND = 12      # nobody speaks faster: more new tokens than this is the model talking to itself


def _without_loops(text: str, longest_phrase: int = 4, times: int = 4) -> str:
    """`text` without phrases said `times` or more times in a row ("Oh, oh, oh, oh, oh, …").

    On loud noise with no speech in it Qwen writes one short word over and over. Nobody says the
    same thing four times running, so such a run is not speech: all of it is dropped.
    """
    words = _units(text)
    said = [_norm(w) for w in words]
    keep = [True] * len(words)
    for period in range(1, longest_phrase + 1):
        run = 0                                     # words in a row equal to the one `period` before
        for i in range(period, len(said) + 1):
            if i < len(said) and said[i] == said[i - period]:
                run += 1
                continue
            if run >= period * (times - 1):
                keep[i - run - period:i] = [False] * (run + period)
            run = 0
    return text if all(keep) else _rejoin([w for w, k in zip(words, keep) if k])


@dataclass
class SentenceWindowListener(RollingListener):
    commits_text_itself = True      # the engine cuts sentences inside `committed`, like Whisper's
    block_seconds: float = 8        # audio encoded as one piece; a full piece is encoded once and kept.
                                    # The encoder was trained on pieces of 1–8 s. Smaller: less of the
                                    # window changes per pass (less to prefill), but less context each.

    def __post_init__(self) -> None:
        super().__post_init__()
        self.foothold = ""               # settled text whose audio is still in the window
        self.seen: list[tuple[int, int]] = []   # per pass: (samples heard, words in the whole text)
        self.previous_open = ""          # the open text after the previous pass
        self.open_unchanged_since = 0    # samples heard when the open text last changed

    @property
    def window_starts_at(self) -> int:
        """Sample (session clock) of the window's first sample."""
        return self.heard - len(self.window)

    # ---- vocabulary of the spec -------------------------------------------------------------

    def hear(self, pcm: np.ndarray) -> None:
        """音を取り込む（手放すのは聞き直したあと、確定に応じて）"""
        self.heard += len(pcm)
        self.window = np.concatenate([self.window, pcm.astype(np.float32)])

    def listen(self) -> str:
        """窓の音を聞き直し、足場の続きを最新にして、確定できる文を確定する（1パス）"""
        if len(self.window) < SAMPLE_RATE // 4:
            return self.text
        t0 = time.perf_counter()
        stats = PassStats(audio_sec=len(self.window) / SAMPLE_RATE)
        features = self._features(stats)
        mx.eval(features)
        stats.encode_seconds = time.perf_counter() - t0
        tokens = self._decode(features, stats)
        stats.decode_seconds = time.perf_counter() - t0 - stats.encode_seconds
        self.header = tokens[: self._header_len(tokens)]
        self.detected_language = self._language_of(tokens)
        self.draft = tokens
        self.window_text = self._after_foothold(self._detokenize(tokens))
        if not self.foothold and self._says_again_what_was_settled(self.window_text):
            self.window_text = ""
        self._note_what_was_seen()
        self._settle()
        stats.seconds = time.perf_counter() - t0
        self.last = stats
        return self.text

    def reset(self) -> None:
        heard = self.heard
        self.__post_init__()
        self.heard = heard
        self.open_unchanged_since = heard

    # ---- 足場の続きを読む -------------------------------------------------------------------

    def _after_foothold(self, transcript: str) -> str:
        """What the window says after the foothold's last words (the whole transcript if there is none)."""
        if not self.foothold:
            return transcript
        hold = [_norm(u) for u in _units(self.foothold)]
        said = _units(transcript)
        said_n = [_norm(u) for u in said]
        sizes = (8, 6, 4) if _unspaced(self.foothold.strip()[-1]) else (5, 4, 3, 2)
        for k in sizes:                              # the foothold opens the window: first match wins
            if len(hold) < k:
                continue
            for i in range(len(said_n) - k + 1):
                if said_n[i:i + k] == hold[-k:]:
                    return _rejoin(said[i + k:])
        # Its last words came out differently this time: line the two up as well as they go. The
        # unmatched end of the foothold is taken to run to the next sentence end if one is that
        # near ("…a web search." for "…a web search tool."), else to be as many words as it has.
        matches = [m for m in difflib.SequenceMatcher(None, hold, said_n, autojunk=False).get_matching_blocks()
                   if m.size]
        if not matches:
            return _rejoin(said[len(hold):])
        last = matches[-1]
        matched_to, unmatched = last.b + last.size, len(hold) - (last.a + last.size)
        if self.foothold.strip()[-1] in TERMINALS:
            for i in range(matched_to - 1, min(len(said), matched_to + unmatched)):
                if said[i][-1] in TERMINALS:
                    return _rejoin(said[i + 1:])
        return _rejoin(said[matched_to + unmatched:])

    def _says_again_what_was_settled(self, text: str) -> bool:
        """A fresh window whose whole text is, word for word, the sentences settled just before it.

        Seen on loud noise with no speech in it: every new window came out as the last caption
        again ("So, maybe start centering." twenty times), whether read back from "previously
        said" or heard in the noise a second time. Somebody who does repeat a whole sentence after
        a pause loses the repeat.
        """
        again, settled = [_norm(u) for u in _units(text)], _units(self.committed)
        if not again or len(again) > len(settled):
            return False
        starts_a_sentence = len(again) == len(settled) or settled[-len(again) - 1][-1] in TERMINALS
        return starts_a_sentence and [_norm(u) for u in settled[-len(again):]] == again

    def _context(self) -> str:
        """As RollingListener's, but "previously said" stops where the window's audio begins."""
        before_window = self.committed[: len(self.committed) - len(self.foothold)] if self.foothold \
            else self.committed
        parts = []
        if self.terms:
            parts.append("Terms: " + ", ".join(self.terms))
        if before_window.strip():
            parts.append("Previously said: " + before_window.strip()[-400:])
        return "\n".join(parts)

    def _most_new_tokens(self) -> int:
        """With no speech in the window Qwen was seen to read its own context back, 160 tokens of it."""
        return min(MAX_NEW, 16 + int(TOKENS_PER_SECOND * len(self.window) / SAMPLE_RATE))

    def _features(self, stats: PassStats) -> mx.array:
        """Encoder output for the window: full pieces from the cache, the unfinished last piece afresh."""
        piece = int(self.block_seconds * SAMPLE_RATE)
        full = len(self.window) // piece
        while len(self.block_features) < full:
            b = len(self.block_features)
            self.block_features.append(self._encode(b * piece, (b + 1) * piece))
            stats.encoded_sec += self.block_seconds
        parts = list(self.block_features)
        if len(self.window) - full * piece >= HOP * 2:
            parts.append(self._encode(full * piece, len(self.window)))
            stats.encoded_sec += (len(self.window) - full * piece) / SAMPLE_RATE
        return mx.concatenate(parts, axis=1)

    def _strip_echo(self, text: str) -> str:
        """As RollingListener's, and also when the echo was cut short ("… schedule. Previously said.")."""
        text = super()._strip_echo(text)
        for line in self._context().split("\n"):
            marker = line.split(":")[0]
            at = text.rfind(marker) if marker else -1
            if at >= 0 and _norm(line).startswith(_norm(text[at:])):
                text = text[:at].strip()
        return text

    def _detokenize(self, tokens: list[int]) -> str:
        return _without_loops(super()._detokenize(tokens))

    def _language_of(self, tokens: list[int]):
        from mlx_qwen3_asr.tokenizer import parse_asr_output
        lang, _ = parse_asr_output(self.tokenizer.decode(tokens), user_language=self.language)
        return lang or self.language

    # ---- 確定 -------------------------------------------------------------------------------

    def _settle(self) -> None:
        """確定できるところまで確定し、その音を手放す"""
        if self.window_text != self.previous_open:
            self.open_unchanged_since = self.heard
        settled = self._agreed_sentences()
        quiet = (self.heard - self.open_unchanged_since) / SAMPLE_RATE >= QUIET_SETTLES
        if quiet and (self.window_text or self.foothold):
            self._settle_everything_and_start_afresh()
        elif quiet:
            if len(self.window) > int((KEPT_WHILE_IDLE + 1.0) * SAMPLE_RATE):
                self._cut_window_at(self.heard - int(KEPT_WHILE_IDLE * SAMPLE_RATE), words_kept=0)
        elif settled:
            self._settle_words(settled)
        elif len(self.window) / SAMPLE_RATE >= MAX_WINDOW:
            self._settle_words(self._words_to_settle_without_a_sentence_end())
        self.previous_open = self.window_text

    def _agreed_sentences(self) -> int:
        """How many words at the start of the open text form sentences that ended (with the next
        word begun) at the same place, with the same words, in the previous pass too."""
        now, before = _units(self.window_text), _units(self.previous_open)
        ended_before = set(self._sentence_ends(self.previous_open))
        agreed = 0
        for n in self._sentence_ends(self.window_text):
            if n not in ended_before or [_norm(u) for u in now[:n]] != [_norm(u) for u in before[:n]]:
                break
            agreed = n
        return agreed

    @staticmethod
    def _sentence_ends(text: str) -> list[int]:
        """Word counts at which a complete sentence ends in `text` (the next word has begun)."""
        ends, words = [], 0
        while (end := _first_sentence_end(text)) is not None:
            words += len(_units(text[:end]))
            ends.append(words)
            text = text[end:]
        return ends

    def _words_to_settle_without_a_sentence_end(self) -> int:
        open_words = _units(self.window_text)
        n = max(0, len(open_words) - OPEN_WORDS_KEPT)
        for i in range(n, 0, -1):                       # the last clause break before that, if any
            if open_words[i - 1][-1] in CLAUSE_BREAKS:
                return i
        return n

    def _settle_words(self, n: int) -> None:
        """開いている文の先頭 n 語を確定し、足場より前の音を手放す"""
        if n <= 0:
            return
        open_words = _units(self.window_text)
        self.committed = _spaced(self.committed, _rejoin(open_words[:n]))
        self.window_text = _rejoin(open_words[n:])
        settled = _units(self.committed)
        wanted = (2 if _unspaced(settled[-1][-1]) else 1) * FOOTHOLD_WORDS
        held = len(_units(self.foothold)) + n            # settled words whose audio is in the window
        keep = min(held, wanted)
        first_kept = len(settled) - keep
        cut = self._began_before(first_kept) - int(CUT_MARGIN * SAMPLE_RATE)
        if cut <= self.window_starts_at:                 # nothing to let go of yet
            self.foothold = _rejoin(settled[len(settled) - held:])
            return
        self.foothold = _rejoin(settled[first_kept:])
        self._cut_window_at(cut, words_kept=keep + len(open_words) - n)

    def _settle_everything_and_start_afresh(self) -> None:
        self.committed = _spaced(self.committed, self.window_text)
        self.window_text, self.foothold = "", ""
        self._cut_window_at(self.heard - int(KEPT_AFTER_QUIET * SAMPLE_RATE), words_kept=0)

    # ---- 音を手放す -------------------------------------------------------------------------

    def _note_what_was_seen(self) -> None:
        self.seen.append((self.heard, len(_units(self.committed)) + len(_units(self.window_text))))

    def _began_before(self, word: int) -> int:
        """Samples heard by the first pass whose text reached past word number `word` (0-based)."""
        for heard, words in self.seen:
            if words > word:
                return heard
        return self.heard

    def _cut_window_at(self, sample: int, words_kept: int) -> None:
        """窓の頭を sample（セッションの時計）まで進める。下書きは残す語のぶんだけにする。"""
        drop = sample - self.window_starts_at
        if drop <= 0:                                    # the window is already shorter than that
            return
        self.window = self.window[drop:]
        self.seen = [(h, w) for h, w in self.seen if h >= sample]
        self.block_features = []                         # blocks are counted from the window's start
        self.cache_prompt, self.prefilled_blocks = [], 0
        self.draft = self.header + self._last_words_of_draft(words_kept)

    def _last_words_of_draft(self, words: int) -> list[int]:
        """The draft's tokens that spell its last `words` words (a proposal only: it is verified)."""
        body = self.draft[len(self.header):]
        if words <= 0:
            return []
        start = 0
        for i in range(len(body) - 1, -1, -1):
            if len(_units(self._detokenize(body[i:]))) > words:
                start = i + 1
                break
        return body[start:]
