"""Stateful streaming translation on PLaMo 3 Translate (MLX).

The source sentence grows word by word while the speaker talks. Nothing already computed is thrown
away between revisions:

  * KVTape keeps the KV cache and the token list it encodes in lock-step. A new prompt is applied
    by rewinding to the longest common prefix and feeding only the difference.
  * The history (finished source/translation pairs) is a prefix of every prompt, so it is never
    re-prefilled; committing a sentence just leaves its tokens on the tape.
  * The previous translation of the current sentence is a draft. It is verified in the same
    forward pass that prefills the changed source tail; accepted tokens cost no sequential steps.
    After a divergence, the draft is re-aligned on the last generated n-gram and retried.
  * Several target languages are revised together: each forward pass packs every language's
    tokens into one sequence (see plamo_fast.packed_forward), so the weights are read once.

Spec in one line: 「話された英文が伸びるたびに、全言語の訳をまとめて訳し直す。過去の文脈と前回の訳は捨てずに使う」.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from mlx_lm.tokenizer_utils import TokenizerWrapper

from translay_mt.plamo_fast import Segment, packed_forward

TAG, BOS = 16, 1
STOP = frozenset({TAG, BOS})


class KVTape:
    """A KV cache plus the exact token sequence it holds. The two never diverge."""

    def __init__(self, model: Any):
        self.model = model
        self.cache = model.make_cache()
        self.tokens: list[int] = []

    def rewind(self, n: int) -> None:
        if n <= 0:
            return
        for c in self.cache:
            if not c.is_trimmable():
                raise RuntimeError("cache no longer trimmable; context exceeded the sliding window")
            c.trim(n)
        del self.tokens[-n:]

    def rewind_to(self, length: int) -> None:
        self.rewind(len(self.tokens) - length)

    def common_prefix(self, target: list[int]) -> int:
        n = 0
        for a, b in zip(self.tokens, target):
            if a != b:
                break
            n += 1
        return n

    @staticmethod
    def feed_together(requests: list[tuple[KVTape, list[int], int]]) -> list[list[int]]:
        """Append tokens to several tapes in one packed forward pass; return, per tape, the greedy
        predictions for its positions >= logits_from."""
        model = requests[0][0].model
        preds = packed_forward(model, [Segment(tokens, tape.cache, lf) for tape, tokens, lf in requests])
        for tape, tokens, _ in requests:
            tape.tokens.extend(tokens)
        return preds

    def feed(self, tokens: list[int], logits_from: int) -> list[int]:
        return KVTape.feed_together([(self, tokens, logits_from)])[0]


@dataclass
class RevisionStats:
    fed_prompt: int = 0          # source/prompt tokens that had to be prefilled
    draft_offered: int = 0
    draft_accepted: int = 0
    sequential_steps: int = 0    # forward passes that produced exactly one new token
    forward_passes: int = 0
    seconds: float = 0.0


@dataclass
class Revision:
    text: str
    tokens: list[int]
    stats: RevisionStats


class Decoding:
    """One revision in progress, advanced one forward pass at a time.

    `feed`/`logits_from` say what this revision needs from the next forward pass; `absorb` takes
    that pass's predictions, accepts what matches the draft, rewinds what does not, and decides
    the next feed. Splitting the loop this way lets several languages share each forward pass.
    """

    def __init__(self, translator: StreamingTranslator, prompt: list[int], draft: list[int]):
        self.translator = translator
        self.tape = translator.tape
        self.draft = draft
        self.stats = RevisionStats()
        self.out: list[int] = []
        self.done = False

        keep = self.tape.common_prefix(prompt)
        if keep == len(prompt):          # need at least one fed token to get fresh logits
            keep -= 1
        self.tape.rewind_to(keep)
        new = prompt[keep:]
        self.stats.fed_prompt = len(new)
        # First pass: prefill the changed tail and verify the whole draft against it.
        self.proposal = draft[: translator.max_new]
        self.feed = new + self.proposal
        self.logits_from = len(new) - 1

    def absorb(self, preds: list[int]) -> None:
        stats = self.stats
        stats.forward_passes += 1
        if self.proposal:
            ok = 0
            while ok < len(self.proposal) and preds[ok] == self.proposal[ok]:
                ok += 1
            stats.draft_offered += len(self.proposal)
            stats.draft_accepted += ok
            self.tape.rewind(len(self.proposal) - ok)
            self.out.extend(self.proposal[:ok])
            nxt = preds[ok]
        else:
            if self.out:
                stats.sequential_steps += 1
            nxt = preds[0]

        if nxt in STOP or len(self.out) >= self.translator.max_new:
            self.done = True
            return
        self.out.append(nxt)
        self.proposal = self.translator.realigned_draft(self.out, self.draft) if self.draft else []
        self.feed = [nxt] + self.proposal
        self.logits_from = 0


@dataclass
class StreamingTranslator:
    model: Any
    tokenizer: TokenizerWrapper
    target_lang: str = "ja"
    use_draft: bool = True
    realign_ngram: int = 2
    realign_span: int = 12
    max_new: int = 200
    # Tokens of past pairs kept on the tape. Must stay well under the 2048-token sliding window:
    # a RotatingKVCache past its window can no longer be rewound.
    history_budget: int = 1500
    history_keep_after_trim: int = 2   # pairs kept when the budget is exceeded
    tape: KVTape = field(init=False)
    history: list[list[int]] = field(default_factory=list, init=False)
    draft: list[int] = field(default_factory=list, init=False)
    _pending: tuple[list[int], list[int]] | None = field(default=None, init=False)

    def __post_init__(self):
        self.tape = KVTape(self.model)
        self._header = [BOS] + self._enc("<|plamo:tag|>dataset<|plamo:msg|>translation")
        self._out_head = self._enc(
            f"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{self.target_lang}<|plamo:msg|>")

    # ---- vocabulary of the spec -------------------------------------------------------------

    def revise(self, source_so_far: str) -> Revision:
        """Translate the unfinished sentence, reusing the tape and the previous translation."""
        return revise_together([self], source_so_far)[0]

    def commit(self) -> None:
        """The sentence is finished: its pair becomes history, already resident on the tape."""
        if self._pending is None:
            return
        src, out = self._pending
        self.history.append(src + self._out_head + out)
        self._pending = None
        self.draft = []
        self._trim_history()

    def discard(self) -> None:
        """This sentence will not be translated by this translator: forget its unfinished draft."""
        self._pending = None
        self.draft = []

    def reset(self) -> None:
        self.tape = KVTape(self.model)
        self.history.clear()
        self.draft = []
        self._pending = None

    def settle(self) -> int:
        """Make the tape hold exactly the current context, so the next revision feeds only the new
        sentence. Call it in idle time right after commit(): when history was trimmed this is the
        one re-prefill, paid here instead of inside the next caption. Returns tokens prefilled."""
        return settle_together([self])[0]

    # ---- mechanics --------------------------------------------------------------------------

    def begin(self, source_so_far: str) -> Decoding:
        src = self._enc(f"<|plamo:tag|>input<|plamo:msg|>{source_so_far}")
        self._pending = (src, [])
        prompt = self._context() + src + self._out_head
        return Decoding(self, prompt, self.draft if self.use_draft else [])

    def conclude(self, decoding: Decoding, seconds: float) -> Revision:
        out = decoding.out
        self.draft = out
        assert self._pending is not None
        self._pending = (self._pending[0], out)
        decoding.stats.seconds = seconds
        return Revision(self.tokenizer.decode(out), out, decoding.stats)

    def settle_feed(self) -> tuple[list[int], int]:
        """Rewind to the shared prefix with the context; return the context tokens still to feed."""
        context = self._context()
        keep = min(self.tape.common_prefix(context), len(context) - 1)
        self.tape.rewind_to(keep)
        new = context[keep:]
        return new, len(new) - 1

    def realigned_draft(self, out: list[int], draft: list[int]) -> list[int]:
        """Find the last generated n-gram in the old draft; propose what followed it there."""
        k = self.realign_ngram
        if len(out) < k:
            return []
        key = out[-k:]
        for i in range(len(draft) - k, -1, -1):
            if draft[i:i + k] == key:
                return draft[i + k:i + k + self.realign_span]
        return []

    def _enc(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _context(self) -> list[int]:
        return self._header + [t for pair in self.history for t in pair]

    def _trim_history(self) -> None:
        # Dropping the oldest pair invalidates the whole tape (it changes the prefix), so drop in
        # bulk and rarely: let history grow to the budget, then keep only the latest few pairs,
        # which makes the one re-prefill (see settle) short.
        if self.history_budget <= 0:
            self.history.clear()
            return
        if sum(map(len, self.history)) > self.history_budget:
            del self.history[: -self.history_keep_after_trim]


def revise_together(translators: list[StreamingTranslator], source_so_far: str) -> list[Revision]:
    """同じ英文を全言語まとめて訳し直す: 各 forward pass に、まだ訳し終えていない言語を詰め合わせる。"""
    t0 = time.perf_counter()
    decodings = [tr.begin(source_so_far) for tr in translators]
    active = list(decodings)
    while active:
        preds = KVTape.feed_together([(d.tape, d.feed, d.logits_from) for d in active])
        for d, p in zip(active, preds):
            d.absorb(p)
        active = [d for d in active if not d.done]
    seconds = time.perf_counter() - t0
    return [tr.conclude(d, seconds) for tr, d in zip(translators, decodings)]


def settle_together(translators: list[StreamingTranslator]) -> list[int]:
    """全言語のテープを現在の文脈に揃える（1回の forward pass で）。"""
    feeds = [tr.settle_feed() for tr in translators]
    KVTape.feed_together([(tr.tape, new, lf) for tr, (new, lf) in zip(translators, feeds)])
    return [len(new) for new, _ in feeds]
