"""Checks for the sentence cutter.  uv run python -m translay_mt.sentences_check"""
from __future__ import annotations

from translay_mt.sentences import Finished, SentenceCutter, Speaking

failures = 0


def check(cond: bool, what: str) -> None:
    global failures
    print(("ok   " if cond else "FAIL ") + what)
    failures += 0 if cond else 1


c = SentenceCutter(silence_to_finish=1.0)
check(c.cut("Hello", 0.5) == [Speaking("Hello")], "growing text is speaking")
check(c.cut("Hello there. How", 1.0) == [Speaking("Hello there. How")], "complete but not yet agreed")
check(c.cut("Hello there. How are", 1.4) == [Finished("Hello there."), Speaking("How are")],
      "agreed over two passes → finished")
check(c.cut("Hello there. How are you", 1.8) == [Speaking("How are you")], "finished text not repeated")
check(c.cut("Hello there! How are you", 2.2) == [], "revision inside a finished sentence is tolerated")
check(c.cut("Hello there! How are you", 3.3) == [Finished("How are you")], "silence finishes the rest")
check(c.cut("Hello there! How are you. .", 3.6) == [], "punctuation-only fragment never becomes a line")

c = SentenceCutter(silence_to_finish=1.0)
c.cut("I met Mr. Smith today. Then", 1)
check(c.cut("I met Mr. Smith today. Then it", 2) == [Finished("I met Mr. Smith today."), Speaking("Then it")],
      "abbreviation does not split")

c = SentenceCutter(silence_to_finish=1.0)
c.cut("One. Two. Three", 1)
r = c.cut("One. Two. Three four", 1.5)
check(r == [Finished("One."), Speaking("Two. Three four")], "one agreement per sentence")
check(c.cut("One. Two. Three four five", 2) == [Finished("Two."), Speaking("Three four five")],
      "the next sentence finishes on its own agreement")
check(c.flush() == [Finished("Three four five")], "flush finishes the pending tail")

c = SentenceCutter(silence_to_finish=1.0)
c.cut("こんにちは。今日は", 1)
check(c.cut("こんにちは。今日は晴れ", 1.5) == [Finished("こんにちは。"), Speaking("今日は晴れ")],
      "Japanese sentence ends at 。 without a following space")

c = SentenceCutter(silence_to_finish=1.0)
check(c.cut_stable("", "Hello there", 0.5) == [Speaking("Hello there")], "stable: tentative only → speaking")
check(c.cut_stable("Hello there.", "How", 1.0) == [Finished("Hello there."), Speaking("How")],
      "stable: committed sentence finishes at once when the next word exists")
check(c.cut_stable("Hello there. How are you", "doing", 1.5) == [Speaking("How are you doing")],
      "stable: committed but incomplete sentence stays speaking")
check(c.cut_stable("Hello there. How are you doing", "", 3.0) == [Finished("How are you doing")],
      "stable: silence with nothing tentative finishes the committed rest")
check(c.cut_stable("Hello there. How are you doing in parallel. Each", "worker", 3.5)
      == [Finished("in parallel."), Speaking("Each worker")], "stable: never cuts into a word")

c = SentenceCutter(silence_to_finish=1.0)
c.cut_stable("the goal is simple", "help japan", 1.0)
long = "the goal is simple help japan stabilize its currency but it wants"
check(c.cut_stable(long, "japan to", 2.0) == [Finished(long), Speaking("japan to")],
      "stable: long committed text without punctuation is finished as it stands")
check(c.cut_stable(long + " japan to do it", "without", 3.0) == [Speaking("japan to do it without")],
      "stable: … and the next piece starts a new line")

print("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
raise SystemExit(1 if failures else 0)
