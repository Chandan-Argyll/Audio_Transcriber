import time
from pathlib import Path
from typing import Any
from app.config import get_settings

LINES = [
    'Thanks everyone for joining. Let us walk through the recording and agree on our next steps.',
    'I reviewed the latest feedback. People want a clear transcript and an easy way to find decisions.',
    'That makes sense. We should keep the speaker labels consistent throughout the entire conversation.',
    'The first milestone is the upload experience. It needs to feel simple even with a longer recording.',
    'I can take that action. We should also show progress so people know what is happening.',
    'For accessibility, the timestamps and playback controls need to work with a keyboard too.',
    'How do we handle an unfamiliar voice or a very short interruption in the middle of a sentence?',
    'We can mark uncertain speakers as unknown and let the user review the result afterwards.',
    'Good idea. Let us capture that in the notes and revisit it during the next review.',
    'Before we finish, are there any remaining questions about the timeline or the export formats?',
]


def mock_segments(duration: float) -> list[dict[str, Any]]:
    segments = []
    for i in range(40):
        start = i * duration / 40
        end = (i + 0.88) * duration / 40
        tokens = LINES[i % len(LINES)].split()
        step = (end - start) / len(tokens)
        words = [{'word': token, 'start': round(start + j * step, 4), 'end': round(start + (j + .9) * step, 4), 'score': .96} for j, token in enumerate(tokens)]
        segments.append({'start': start, 'end': end, 'text': ' '.join(tokens), 'words': words})
    return segments


class MockEngine:
    """Deterministic demo data; never represents the uploaded speech."""
    def transcribe(self, audio: Path, duration: float) -> dict[str, Any]:
        time.sleep(get_settings().mock_delay_sec)
        return {'language': 'en', 'segments': mock_segments(duration)}

    def align(self, result: dict[str, Any], audio: Path) -> dict[str, Any]:
        time.sleep(get_settings().mock_delay_sec)
        return result

    def diarize(self, audio: Path, duration: float, options: dict[str, int]) -> list[dict[str, Any]]:
        time.sleep(get_settings().mock_delay_sec)
        count = options.get('num_speakers', max(options.get('min_speakers', 1), min(3, options.get('max_speakers', 8))))
        return [{'start': i * duration / 40, 'end': (i + 1) * duration / 40, 'speaker': f'SPEAKER_{i % count:02d}'} for i in range(40)]
