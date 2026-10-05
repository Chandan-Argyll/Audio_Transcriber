"""Domain correction dictionaries (backend/vocab/*.json) applied after transcription.

A dictionary has "replacements" and optional "hotwords":
    {"from": ["void dire", "voidir"], "to": "voir dire"}
    {"from": ["EPA"], "to": "AEDPA", "near": ["habeas", "federal"], "window": 3}
Matching is whole-word and case-insensitive on words stripped of surrounding
punctuation; multi-word sources match consecutive words. With "near", a match
is replaced only if one of those words occurs within "window" words before or
after it. Replacements keep the surrounding punctuation and the source's case
(ALL CAPS stays caps, a capitalised first word stays capitalised); a target that
contains capitals (AEDPA, Batson) is used exactly as written. Timing of the
replaced words is spread evenly over the new words.
"""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

VOCAB_DIR = Path(__file__).resolve().parents[1] / 'vocab'
_EDGE = re.compile(r"^([^\w']*)(.*?)([^\w']*)$", re.S)


def normalize(word: str) -> str:
    return _EDGE.match(word).group(2).lower()


@lru_cache
def load(name: str) -> dict[str, Any]:
    """A built-in dictionary name (vocab/<name>.json) or a path to a JSON file."""
    path = Path(name) if name.endswith('.json') else VOCAB_DIR / f'{name}.json'
    if not path.is_file():
        raise ValueError(f'Vocabulary "{name}" not found ({path})')
    data = json.loads(path.read_text(encoding='utf-8'))
    for rule in data.get('replacements', []):
        if not rule.get('from') or not rule.get('to'):
            raise ValueError(f'{path.name}: each replacement needs "from" and "to"')
    return data


def match_case(target: str, source: list[str]) -> str:
    if any(c.isupper() for c in target):
        return target
    letters = ''.join(c for c in ' '.join(source) if c.isalpha())
    if len(letters) > 1 and letters.isupper():
        return target.upper()
    first = next((c for c in source[0] if c.isalpha()), '')
    return target[:1].upper() + target[1:] if first.isupper() else target


class Corrector:
    def __init__(self, dictionaries: list[dict[str, Any]]) -> None:
        self.rules = []
        for data in dictionaries:
            for rule in data.get('replacements', []):
                for source in rule['from']:
                    self.rules.append(([normalize(w) for w in source.split()], rule['to'],
                                       {normalize(w) for w in rule.get('near', [])}, int(rule.get('window', 3))))
        # Longest phrases first, so "void dire" wins over any single-word rule.
        self.rules.sort(key=lambda r: -len(r[0]))
        self.hotwords = list(dict.fromkeys(h for d in dictionaries for h in d.get('hotwords', [])))

    def _context_ok(self, keys: list[str], first: int, last: int, near: set[str], window: int) -> bool:
        return not near or bool(near.intersection(keys[max(0, first - window):first] + keys[last:last + window]))

    def correct_words(self, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keys = [normalize(w.get('word', '')) for w in words]
        output: list[dict[str, Any]] = []
        i = 0
        while i < len(words):
            for source, target, near, window in self.rules:
                last = i + len(source)
                if keys[i:last] == source and self._context_ok(keys, i, last, near, window):
                    output.extend(self._replace(words[i:last], target))
                    i = last
                    break
            else:
                output.append(words[i])
                i += 1
        return output

    @staticmethod
    def _replace(originals: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
        texts = [w.get('word', '') for w in originals]
        lead = _EDGE.match(texts[0]).group(1)
        trail = _EDGE.match(texts[-1]).group(3)
        parts = match_case(target, [_EDGE.match(t).group(2) for t in texts]).split()
        start, end = originals[0].get('start'), originals[-1].get('end')
        scores = [w.get('score') for w in originals if w.get('score') is not None]
        replaced = []
        for n, part in enumerate(parts):
            word = {**originals[0], 'word': (lead if n == 0 else '') + part + (trail if n == len(parts) - 1 else ''),
                    'score': min(scores) if scores else None}
            if start is not None and end is not None:
                step = (end - start) / len(parts)
                word['start'], word['end'] = round(start + n * step, 3), round(start + (n + 1) * step, 3)
            replaced.append(word)
        return replaced

    def apply(self, result: dict[str, Any]) -> dict[str, Any]:
        if not self.rules:
            return result
        for segment in result['segments']:
            if segment.get('words'):
                segment['words'] = self.correct_words(segment['words'])
                segment['text'] = ' '.join(w.get('word', '') for w in segment['words'])
        return result


def corrector_for(names: list[str]) -> Corrector:
    return Corrector([load(name) for name in dict.fromkeys(n for n in names if n)])
