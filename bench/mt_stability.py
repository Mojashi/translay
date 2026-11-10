"""How much does the draft translation rewrite itself while a sentence is being spoken?

    uv run python mt_stability.py ../models/plamo-3-translate-0.8b-MLX-4bit 0 1 2 4

English arrives word by word (WMT24++ speech sentences); the draft is revised each time with the
given stickiness (logit margin within which the previous draft's token is kept), and the finished
sentence is translated with plain greedy (or, with STICKY_FINAL=1, with the same stickiness). Per stickiness:
  rewritten  characters of the previous draft that did not survive as a prefix, per revision
  big        share of revisions that rewrote 8 or more characters
  last jump  characters of the last draft rewritten by the final translation
  ms         time per revision
The final translations are identical across settings only if history is too; so chrF of the finals
against the reference is reported as the quality check.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

from sacrebleu.metrics.chrf import CHRF

from speech_corpus import speech_talks  # noqa: E402
from translay_mt.plamo_fast import load_plamo  # noqa: E402
from translay_mt.streaming_translator import StreamingTranslator  # noqa: E402


STICKY_FINAL = os.environ.get("STICKY_FINAL") == "1"     # also keep the draft's wording in the final translation


def lost(old: str, new: str) -> int:
    n = 0
    for a, b in zip(old, new):
        if a != b:
            break
        n += 1
    return len(old) - n


def main() -> None:
    model, tok = load_plamo(sys.argv[1])
    talks = speech_talks()[:12]
    for stickiness in [float(x) for x in sys.argv[2:]]:
        tr = StreamingTranslator(model, tok)
        rewritten, jumps, ms, outputs = [], [], [], []
        for talk in talks:
            tr.reset()
            finals = []
            for sentence in talk.sentences:
                words, previous = sentence.split(), ""
                tr.draft_stickiness = stickiness
                for i in range(1, len(words)):
                    t0 = time.perf_counter()
                    text = tr.revise(" ".join(words[:i])).text
                    ms.append((time.perf_counter() - t0) * 1000)
                    rewritten.append(lost(previous, text))
                    previous = text
                tr.draft_stickiness = stickiness if STICKY_FINAL else 0.0
                final = tr.revise(sentence).text
                jumps.append(lost(previous, final))
                finals.append(final)
                tr.commit()
                tr.settle()
            outputs.append("".join(finals))
        chrf = CHRF(word_order=2).corpus_score(outputs, [[t.reference for t in talks]]).score
        print(f"stickiness={stickiness:3.1f}  rewritten={statistics.mean(rewritten):5.2f} chars/revision  "
              f"big={sum(r >= 8 for r in rewritten) / len(rewritten):5.1%}  last jump={statistics.mean(jumps):5.2f}  "
              f"ms={statistics.median(ms):5.1f}  final chrF++={chrf:.2f}  (revisions={len(rewritten)})", flush=True)


if __name__ == "__main__":
    main()
