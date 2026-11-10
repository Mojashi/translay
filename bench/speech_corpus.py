"""WMT24++ en→ja 'speech' domain: transcripts of spoken videos, the closest public match to live talk."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).parent / "data" / "wmt24pp" / "en-ja_JP.jsonl"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])")


@dataclass(frozen=True)
class Talk:
    doc_id: str
    source: str          # English paragraph (multi-sentence)
    reference: str       # human Japanese translation of the whole paragraph

    @property
    def sentences(self) -> list[str]:
        return [s.strip() for s in _SENTENCE_END.split(self.source) if s.strip()]


def speech_talks() -> list[Talk]:
    rows = [json.loads(line) for line in DATA.open()]
    return [Talk(r["document_id"], r["source"], r["target"])
            for r in rows if r["domain"] == "speech" and not r["is_bad_source"]]


def speech_sentences() -> list[str]:
    return [s for t in speech_talks() for s in t.sentences]
