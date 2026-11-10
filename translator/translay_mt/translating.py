"""通訳: 同じ英文を全言語まとめて訳す（モデルは共有、テープは言語ごと、forward pass は詰め合わせ）。"""
from __future__ import annotations

import sys
import time
from typing import Any

from translay_mt.streaming_translator import StreamingTranslator, revise_together, settle_together


# Every language PLaMo 3 Translate translates into (its README's 対応言語).
TARGET_LANGUAGES = frozenset(
    "ja en zh-hans zh-hant ko ar az be bg bn ca ceb de es et eu fa fr gu he hi hr hu hy id it jv ka kk km lt lv "
    "ml mr ms my ne nl pa pl ru si sk sl sr sw ta te th tl ur uz vi".split())


# A word of the previous draft is kept unless another is likelier by more than this logit margin.
# Measured with bench/mt_stability.py (0.8B, word-by-word speech): characters rewritten per revision
# 10.5 → 5.6, and the final translation's jump away from the last draft 14.4 → 11.8, with chrF++ of
# the finals unchanged (26.5 → 26.9) when the final is sticky too.
DRAFT_STICKINESS = 2.0


def log(s: str) -> None:
    sys.stderr.write(f"translay: {s}\n")
    sys.stderr.flush()


class Interpreters:
    """`source_lang` is the language code the sentence was spoken in (None if unknown): a target in
    the same language shows the sentence itself instead of a translation."""

    def __init__(self, model: Any, tokenizer: Any, targets: list[str], trace: bool = False):
        self.model, self.tokenizer = model, tokenizer
        self.translators: dict[str, StreamingTranslator] = {}     # the languages being translated into right now
        self.trace = trace
        self.translate_into(targets)

    def translate_into(self, targets: list[str]) -> None:
        """訳す先の言語を選び直す（PLaMo 3 が訳せる言語の中から。空なら訳さない）。

        A language switched off forgets its history: switched on again, it starts afresh.
        """
        unknown = [lang for lang in targets if lang not in TARGET_LANGUAGES]
        if unknown:
            raise ValueError(f"PLaMo 3 does not translate into {unknown}; it knows {sorted(TARGET_LANGUAGES)}")
        self.translators = {lang: self.translators.get(lang) or
                            StreamingTranslator(self.model, self.tokenizer, target_lang=lang,
                                                draft_stickiness=DRAFT_STICKINESS) for lang in targets}

    def revise(self, text: str, source_lang: str | None) -> dict[str, str]:
        """話し途中の文を訳し直す（前回の訳の言い回しは、僅差なら変えない）"""
        same = {lang for lang in self.translators if lang.split("-")[0] == source_lang}    # zh-hans is zh
        active = {lang: tr for lang, tr in self.translators.items() if lang not in same}
        texts = {lang: text for lang in same}
        for lang in texts:
            self.translators[lang].discard()
        if active:
            revisions = revise_together(list(active.values()), text)
            texts.update({lang: r.text for lang, r in zip(active, revisions)})
            if self.trace:
                parts = " ".join(f"{lang}:passes={r.stats.forward_passes},steps={r.stats.sequential_steps},"
                                 f"draft={r.stats.draft_accepted}/{r.stats.draft_offered}"
                                 for lang, r in zip(active, revisions))
                log(f"revise {revisions[0].stats.seconds * 1000:6.1f}ms {parts} | {text[-40:]!r}")
        return {lang: texts[lang] for lang in self.translators}

    def finish(self, text: str, source_lang: str | None) -> dict[str, str]:
        """言い終えた文を確定訳し、履歴に積む"""
        texts = self.revise(text, source_lang)      # same stickiness: the final does not jump away from the draft
        for tr in self.translators.values():
            tr.commit()            # a no-op for translators that skipped this sentence
        return texts

    def settle(self) -> None:
        """確定直後の空き時間に、履歴のテープを整えておく"""
        t0 = time.perf_counter()
        prefilled = settle_together(list(self.translators.values())) if self.translators else []
        if self.trace:
            log(f"settle {(time.perf_counter() - t0) * 1000:6.1f}ms prefilled={prefilled}")

    def reset(self) -> None:
        for tr in self.translators.values():
            tr.reset()
