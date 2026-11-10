"""文の切り出し: 聞き取りの全文から「言い終えた文」と「話し途中の文」を切り出す。

Unlike an append-only streaming recogniser, the rolling listener may still revise recent words, so
a sentence is finished only when it is both complete and agreed on:

  * complete — terminal punctuation followed by the start of the next word (so it cannot be the
    tail of "Mr." or "3." still being decoded), and not a known abbreviation;
  * agreed   — the same sentence came out of two consecutive passes (LocalAgreement-2).

Or, when nothing new has been recognised for `silence_to_finish` seconds of audio, whatever is
pending is finished as it stands. Fragments that are only punctuation never become lines.

Once finished, a sentence is never re-emitted. If a later pass revises words inside it, the cutter
finds where the finished text ends in the new transcript by its last characters and continues
from there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TERMINALS = ".?!。？！"
FULLWIDTH_TERMINALS = "。？！"
CLOSERS = "\"')”’"
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "jr.", "sr."}
ANCHORS = (24, 16, 10)


@dataclass(frozen=True)
class Speaking:
    english: str


@dataclass(frozen=True)
class Finished:
    english: str


Utterance = Speaking | Finished


def _first_sentence_end(text: str) -> int | None:
    """Index just past the first complete sentence in `text`, if any."""
    i = 0
    while i < len(text):
        if text[i] in TERMINALS:
            end = i + 1
            while end < len(text) and (text[end] in TERMINALS or text[end] in CLOSERS):
                end += 1
            nxt = end
            while nxt < len(text) and text[nxt].isspace():
                nxt += 1
            words = text[:end].split()
            # "。" needs no space after it; "." does (else it may be "3." or "Mr." mid-decode).
            spaced_ok = nxt > end or text[i] in FULLWIDTH_TERMINALS
            if spaced_ok and nxt < len(text) and not (words and words[-1].lower() in ABBREVIATIONS):
                return end
            i = end
            continue
        i += 1
    return None


def _is_only_punctuation(s: str) -> bool:
    return not re.search(r"\w", s)


@dataclass
class SentenceCutter:
    silence_to_finish: float = 1.2
    letters_to_finish: int = 40         # cut_stable: committed text this long is finished without waiting for . ? !
    finished_text: str = ""
    _last_text: str = ""
    _last_change: float = 0.0
    _last_candidate: str | None = None
    _last_spoken: str = ""
    _stable_done: int = field(default=0)     # characters of the stable text already finished

    def cut(self, transcript: str, heard_seconds: float) -> list[Utterance]:
        """聞き取りの全文（その時点で heard_seconds 秒ぶん聞いた）から出来事を切り出す"""
        if transcript != self._last_text:
            self._last_text = transcript
            self._last_change = heard_seconds
        events: list[Utterance] = []
        pending = self._pending(transcript)

        end = _first_sentence_end(pending)
        while end is not None:
            candidate = pending[:end].strip()
            if candidate != self._last_candidate:      # not agreed yet: wait for the next pass
                self._last_candidate = candidate
                break
            self._finish(candidate, events)
            self._last_candidate = None
            pending = pending[end:]
            end = _first_sentence_end(pending)
            if end is not None:                        # the next one needs its own agreement
                self._last_candidate = pending[:end].strip()
                break

        speaking = pending.strip()
        if speaking and heard_seconds - self._last_change >= self.silence_to_finish:
            self._finish(speaking, events)
            self._last_candidate = None
        elif speaking and speaking != self._last_spoken and not _is_only_punctuation(speaking):
            events.append(Speaking(speaking))
            self._last_spoken = speaking
        return events

    def cut_stable(self, stable: str, tentative: str, heard_seconds: float) -> list[Utterance]:
        """聞き取り側が確定済み（二度と変わらない）の文と、未確定の続きから出来事を切り出す。

        For recognisers that commit text themselves (Whisper commits agreed segments and drops
        their audio): sentences are finished only inside the stable text — at once, no second
        agreement — and the tentative rest is only ever "speaking". Two agreement layers that could
        disagree about the same words are never stacked.
        """
        transcript = f"{stable} {tentative}".strip()
        if transcript != self._last_text:
            self._last_text = transcript
            self._last_change = heard_seconds
        events: list[Utterance] = []
        pending = stable[self._stable_done:]
        while True:
            end = _first_sentence_end(f"{pending} {tentative}".rstrip() if tentative else pending)
            if end is None or end > len(pending):
                break
            self._finish(pending[:end].strip(), events)
            self._stable_done += end
            pending = pending[end:]
        rest = f"{pending.strip()} {tentative.strip()}".strip()
        unended = len(re.findall(r"[^\W\d_]", pending))       # letters committed with no sentence end in sight
        if unended >= self.letters_to_finish:
            # No punctuation is coming (Whisper sometimes writes none): what is committed ends at a
            # break the recogniser chose, and waiting longer only delays the caption.
            self._finish(pending.strip(), events)
            self._stable_done = len(stable)
            rest = tentative.strip()
        if rest and not tentative.strip() and heard_seconds - self._last_change >= self.silence_to_finish:
            self._finish(pending.strip(), events)
            self._stable_done = len(stable)
        elif rest and rest != self._last_spoken and not _is_only_punctuation(rest):
            events.append(Speaking(rest))
            self._last_spoken = rest
        return events

    def flush(self) -> list[Utterance]:
        """音が途切れた: 話し途中の文も言い終えたものとして扱う"""
        events: list[Utterance] = []
        pending = self._pending(self._last_text).strip()
        if pending:
            self._finish(pending, events)
        return events

    def _finish(self, sentence: str, events: list[Utterance]) -> None:
        self.finished_text = f"{self.finished_text} {sentence}".strip()
        self._last_spoken = ""
        if not _is_only_punctuation(sentence):
            events.append(Finished(sentence))

    def _pending(self, transcript: str) -> str:
        """The part of the transcript after everything already finished."""
        done = self.finished_text
        if not done:
            return transcript
        if transcript.startswith(done):
            return transcript[len(done):]
        for size in ANCHORS:                       # shorter anchors survive nearer revisions
            anchor = done[-size:]
            at = transcript.rfind(anchor)
            if at >= 0:
                return transcript[at + len(anchor):]
        # The finished words were revised beyond recognition: assume the same number of WORDS
        # (never cut inside a word).
        words, n = transcript.split(), len(done.split())
        return " " + " ".join(words[n:]) if len(words) > n else ""
