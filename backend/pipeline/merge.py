from typing import Any


def assign_mock_speakers(segments: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    """Match WhisperX's maximum intersection policy without ML dependencies."""
    for segment in result['segments']:
        for word in segment.get('words', []):
            overlaps: dict[str, float] = {}
            for interval in segments:
                overlap = max(0.0, min(word['end'], interval['end']) - max(word['start'], interval['start']))
                overlaps[interval['speaker']] = overlaps.get(interval['speaker'], 0) + overlap
            if overlaps and max(overlaps.values()) > 0:
                word['speaker'] = max(overlaps, key=overlaps.get)
    return result


def build_transcript(job_id: str, duration: float, result: dict[str, Any]) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    speakers: dict[str, dict[str, Any]] = {}
    for segment in result['segments']:
        for raw in segment.get('words', []):
            # WhisperX may omit timestamps for punctuation/numerals. Retain these
            # words at the previous word's end (or segment start), never drop text.
            start = raw.get('start', turns[-1]['end'] if turns else segment.get('start', 0))
            end = max(start, raw.get('end', start))
            speaker = raw.get('speaker', 'Unknown')
            if speaker not in speakers:
                index = sum(s['id'] != 'Unknown' for s in speakers.values())
                speakers[speaker] = {'id': speaker, 'label': 'Unknown' if speaker == 'Unknown' else f'Person {index + 1}', 'color_index': index % 8}
            word = {'w': raw.get('word', '').strip(), 'start': start, 'end': end, 'score': raw.get('score')}
            if not turns or turns[-1]['speaker_id'] != speaker:
                turns.append({'id': len(turns), 'speaker_id': speaker, 'start': start, 'end': end, 'text': '', 'words': []})
            turn = turns[-1]
            turn['words'].append(word)
            turn['end'] = max(turn['end'], end)
            turn['text'] = ' '.join(w['w'] for w in turn['words'])
    return {'job_id': job_id, 'duration_sec': duration, 'speakers': list(speakers.values()), 'turns': turns}
