"""Convert a CHAT (.cha) transcript into an evaluation reference (API transcript shape).

    python -m eval.chat SBC019.cha SBC019.json [--duration 1309.6] [--exclude ENV X]

Every timing bullet (start_end in ms) becomes one reference turn, giving
intonation-unit speaker boundaries for DER. Non-speaker tiers (ENV, X by
default), dependent tiers (%mor, %com ...) and headers are ignored. CHAT/CA
markup is stripped: events (&=laugh), scoped markers (&{l=X ... &}l=X),
fragments (&+fr), overlap brackets (⌈ ⌉ ⌊ ⌋ with indices), pauses ((.) (..)),
terminators (+... +/.), [% comments], codes ([/] [>]), unintelligible
xxx/yyy/www, and lengthening colons, cut-off words (r- all-). Fillers (&-uh) keep the spoken word,
replacements (word [: target]) use the target, and omitted sounds are restored
((be)cause -> because). Word times inside a turn are spread evenly; only turn
boundaries are meaningful for DER.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any

BULLET = re.compile('\x15(\\d+)_(\\d+)\x15')
DEFAULT_EXCLUDE = ('ENV', 'X')
_RULES: list[tuple[re.Pattern[str], str]] = [(re.compile(p), r) for p, r in [
    (r'\[%[^\]]*\]', ' '),                    # [% comment]
    (r'(?:<[^>]*>|\S+)\s*\[:\s*([^\]]+)\]', r' \1 '),  # word [: target] -> target
    (r'\[[^\]]*\]', ' '),                     # other codes: [/] [//] [>] [<] [!] [=? x]
    (r'&=\S+', ' '),                          # events: &=laugh &=in &=SNIFF
    (r'&[{}][a-zA-Z]=\S+', ' '),              # scoped markers: &{l=X &}l=X &{n=POUND}
    (r'&-(\S+)', r'\1'),                      # fillers: &-uh -> uh
    (r'&\S+', ' '),                           # fragments: &+fr &cheller
    (r'[⌈⌉⌊⌋]\d*', ''),                       # CA overlap brackets, may sit inside words
    (r'\(\d*\.+\d*\)', ' '),                  # pauses: (.) (..) (...) (1.2)
    (r'(?<!\S)\+\S*', ' '),                   # terminators/linkers: +... +/. +, +"
    (r'(?<!\S)(?:xxx|yyy|www)(?!\S)', ' '),    # unintelligible / untranscribed
    (r'(?<!\S)0\S+', ' '),                    # omitted words: 0is
    (r'\((\w+)\)', r'\1'),                    # omitted sounds: (be)cause -> because
    (r'@\S*', ''),                            # special form markers: hm@i word@l
    (r'_', ' '),                              # compounds: ice_cream
    (r'(?<=\w):+(?=\w)', ''),                  # CA lengthening inside words: Mo:m s:tupid
    (r'[ʔʕ]', ''),                            # CA glottal stop symbols
    (r'(?<!\S)\S*\w-(?!\S)', ' '),            # cut-off words: r- all- Wh-
    (r"[^\w\s'-]", ' '),                      # CA prosody symbols and punctuation (incl. ':')
    (r"(?<!\w)['-]+|['-]+(?!\w)", ' '),        # stray apostrophes/hyphens, truncation dash
    (r'\s+', ' '),
]]


def clean(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


def tiers(text: str) -> list[tuple[str, str]]:
    """(tier name, full text) pairs with tab-continuation lines joined."""
    output: list[tuple[str, str]] = []
    for line in text.splitlines():
        if line.startswith('\t') and output:
            name, body = output[-1]
            output[-1] = (name, body + ' ' + line.strip())
        elif line[:1] in {'*', '%', '@'}:
            name, _, body = line.partition(':')
            output.append((name, body.strip()))
    return output


def parse_chat(text: str, duration: float | None = None, exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
               reference_id: str = 'reference') -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    order: list[str] = []
    for name, body in tiers(text):
        speaker = name[1:]
        if not name.startswith('*') or speaker in exclude:
            continue
        position = 0
        for match in BULLET.finditer(body):
            words = clean(body[position:match.start()]).split()
            position = match.end()
            start, end = int(match.group(1)) / 1000, int(match.group(2)) / 1000
            if not words or end <= start:
                continue
            if speaker not in order:
                order.append(speaker)
            step = (end - start) / len(words)
            turns.append({'speaker_id': speaker, 'start': start, 'end': end, 'text': ' '.join(words),
                          'words': [{'w': w, 'start': round(start + i * step, 3), 'end': round(start + (i + 1) * step, 3),
                                     'score': None} for i, w in enumerate(words)]})
    turns.sort(key=lambda turn: (turn['start'], turn['end']))
    for index, turn in enumerate(turns):
        turn['id'] = index
    return {'job_id': reference_id, 'duration_sec': duration or max((t['end'] for t in turns), default=0),
            'transcription_engine': 'human', 'diarization_engine': 'human',
            'speakers': [{'id': s, 'label': s, 'color_index': i % 8} for i, s in enumerate(order)], 'turns': turns}


def load_chat(path: Path, duration: float | None = None, exclude: tuple[str, ...] = DEFAULT_EXCLUDE) -> dict[str, Any]:
    return parse_chat(path.read_text(encoding='utf-8'), duration, exclude, path.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert CHAT (.cha) to an evaluation reference JSON')
    parser.add_argument('chat', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--duration', type=float, help='Recording length in seconds (default: last bullet end)')
    parser.add_argument('--exclude', nargs='*', default=list(DEFAULT_EXCLUDE), help='Speaker codes to ignore')
    args = parser.parse_args()
    reference = load_chat(args.chat, args.duration, tuple(args.exclude))
    args.output.write_text(json.dumps(reference, indent=2), encoding='utf-8')
    words = sum(len(t['words']) for t in reference['turns'])
    print(f'{len(reference["turns"])} turns, {words} words, speakers: {", ".join(s["id"] for s in reference["speakers"])}')


if __name__ == '__main__':
    main()
