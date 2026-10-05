"""Convert an official Supreme Court oral-argument transcript (PDF or extracted text)
into an evaluation reference.

    python -m eval.scotus 24-7351.pdf 24-7351.json [--duration 6582.4]

Only the argument itself is kept: text between "P R O C E E D I N G S" and the
closing "(Whereupon ... submitted.)". Page headers and footers ("Official -
Subject to Final Review", "Heritage Reporting Corporation" / "Alderson Reporting
Company"), page and line numbers, argument headings ("ORAL ARGUMENT OF ...",
"ON BEHALF OF ..."), and stage directions such as "(Laughter.)" or "(10:27 a.m.)"
are removed. Turns start at speaker labels like "JUSTICE KAGAN:", "CHIEF JUSTICE
ROBERTS:", "MR. STEWART:" or "MS. FERGUSON:". The transcript has no timings, so
turns carry zero-length times: use it for WER, cpWER and speaker attribution, not DER.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any

SPEAKER = re.compile(r"^((?:CHIEF )?JUSTICE [A-Z][A-Z'-]+|M(?:R|S|RS)\. [A-Z][A-Z'-]+|GENERAL [A-Z][A-Z'-]+):\s*")
BOILERPLATE = re.compile(r'^(?:Official - Subject to Final Review|Heritage Reporting Corporation|Alderson Reporting Company'
                         r'|\d{1,3})$')
HEADING = re.compile(r"^[A-Z0-9 .,;'()&-]+$")
STAGE_DIRECTION = re.compile(r'\((?:[A-Z][a-z]+[^()]*\.|\d{1,2}:\d{2} [ap]\.m\.)\)')


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    pages = [page.extract_text() or '' for page in PdfReader(str(path)).pages]
    if not any(page.strip() for page in pages):
        raise ValueError(f'{path.name} has no text layer (scanned images). Download the official transcript '
                         'from supremecourt.gov/oral_arguments/argument_transcript, or OCR it first.')
    return '\n'.join(pages)


def argument_lines(text: str) -> list[str]:
    start = re.search(r'P\s*R\s*O\s*C\s*E\s*E\s*D\s*I\s*N\s*G\s*S', text)
    end = re.search(r'\(Whereupon[^)]*submitted\.\)', text)
    if not start or not end:
        raise ValueError('Not a Supreme Court argument transcript: missing "P R O C E E D I N G S" or "(Whereupon ... submitted.)"')
    lines = []
    for raw in STAGE_DIRECTION.sub(' ', text[start.end():end.start()]).splitlines():
        line = re.sub(r'^\s*\d{1,2}\s{2,}', '', raw).strip()  # Line numbers merged into text.
        if not line or BOILERPLATE.match(line):
            continue
        if HEADING.match(line) and not SPEAKER.match(line) and any(c.isalpha() for c in line):
            continue  # ORAL ARGUMENT OF ..., ON BEHALF OF THE PETITIONER
        lines.append(line)
    return lines


def parse_transcript(text: str, duration: float = 0.0, reference_id: str = 'reference') -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    for line in argument_lines(text):
        match = SPEAKER.match(line)
        if match:
            turns.append({'speaker_id': match.group(1), 'text': line[match.end():]})
        elif turns:
            turns[-1]['text'] += ' ' + line
    order: list[str] = []
    output = []
    for turn in turns:
        words = ' '.join(turn['text'].split()).split()
        if not words:
            continue
        if turn['speaker_id'] not in order:
            order.append(turn['speaker_id'])
        output.append({'id': len(output), 'speaker_id': turn['speaker_id'], 'start': 0.0, 'end': 0.0, 'text': ' '.join(words),
                       'words': [{'w': w, 'start': 0.0, 'end': 0.0, 'score': None} for w in words]})
    if not output:
        raise ValueError('No speaker turns found in the transcript')
    return {'job_id': reference_id, 'duration_sec': duration, 'transcription_engine': 'human', 'diarization_engine': 'human',
            'speakers': [{'id': s, 'label': s.title(), 'color_index': i % 8} for i, s in enumerate(order)], 'turns': output}


def load_scotus(path: Path, duration: float = 0.0) -> dict[str, Any]:
    text = pdf_text(path) if path.suffix.lower() == '.pdf' else path.read_text(encoding='utf-8')
    return parse_transcript(text, duration, path.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert a Supreme Court argument transcript (PDF/TXT) to reference JSON')
    parser.add_argument('transcript', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--duration', type=float, default=0.0, help='Recording length in seconds')
    args = parser.parse_args()
    reference = load_scotus(args.transcript, args.duration)
    args.output.write_text(json.dumps(reference, indent=2), encoding='utf-8')
    words = sum(len(t['words']) for t in reference['turns'])
    print(f'{len(reference["turns"])} turns, {words} words, {len(reference["speakers"])} speakers: '
          f'{", ".join(s["id"] for s in reference["speakers"])}')


if __name__ == '__main__':
    main()
