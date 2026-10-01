import json
from typing import Any


def timestamp(seconds: float, srt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    total, ms = divmod(milliseconds, 1000)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    base = f'{hours:02}:{minutes:02}:{secs:02}'
    return f'{base},{ms:03}' if srt else base


def export(transcript: dict[str, Any], format: str) -> str:
    labels = {s['id']: s['label'] for s in transcript['speakers']}
    if format == 'json':
        return json.dumps(transcript, ensure_ascii=False, indent=2)
    if format == 'txt':
        return '\n'.join(f"[{timestamp(t['start'])}] {labels[t['speaker_id']]}: {t['text']}" for t in transcript['turns'])
    if format != 'srt':
        raise ValueError('Unsupported export format')
    cues = []
    for turn in transcript['turns']:
        prefix = labels[turn['speaker_id']] + ': '
        chunk: list[dict[str, Any]] = []
        def flush() -> None:
            if chunk:
                cues.append(f"{len(cues) + 1}\n{timestamp(chunk[0]['start'], True)} --> {timestamp(chunk[-1]['end'], True)}\n{prefix}{' '.join(w['w'] for w in chunk)}\n")
        for word in turn['words']:
            if chunk and (word['end'] - chunk[0]['start'] > 7 or len(prefix + ' '.join(w['w'] for w in chunk) + ' ' + word['w']) > 84):
                flush()
                chunk = []
            # Split unusually long words and long aligned intervals as well.
            width = max(1, 84 - len(prefix))
            pieces = [word['w'][i:i + width] for i in range(0, len(word['w']), width)] or ['']
            slots = max(len(pieces), int((word['end'] - word['start']) / 7) + 1)
            if slots > 1:
                flush()
                chunk = []
                for i in range(slots):
                    a = word['start'] + (word['end'] - word['start']) * i / slots
                    b = word['start'] + (word['end'] - word['start']) * (i + 1) / slots
                    piece = pieces[min(i, len(pieces) - 1)]
                    cues.append(f"{len(cues) + 1}\n{timestamp(a, True)} --> {timestamp(b, True)}\n{prefix}{piece}\n")
            else:
                chunk.append(word)
        flush()
    return '\n'.join(cues)
