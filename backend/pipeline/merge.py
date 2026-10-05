from bisect import bisect_left
from typing import Any
from math import isfinite

UNKNOWN = 'Unknown'
SENTENCE_END = ('.', '?', '!')


def finite_time(value: Any, fallback: float) -> float:
    return float(value) if value is not None and isfinite(float(value)) else fallback


class SpeakerTimeline:
    """Diarization segments indexed by start time for fast overlap queries."""
    def __init__(self, segments: list[dict[str, Any]]) -> None:
        self.segments = sorted(segments, key=lambda s: s['start'])
        self.starts = [s['start'] for s in self.segments]
        self.longest = max((s['end'] - s['start'] for s in self.segments), default=0.0)

    def overlaps(self, start: float, end: float) -> dict[str, float]:
        totals: dict[str, float] = {}
        for segment in self.segments[bisect_left(self.starts, start - self.longest):bisect_left(self.starts, end)]:
            overlap = min(end, segment['end']) - max(start, segment['start'])
            if overlap > 0:
                totals[segment['speaker']] = totals.get(segment['speaker'], 0) + overlap
        return totals


def assign_speakers(segments: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    """Give each word the speaker with the largest total overlap (WhisperX's policy);
    words overlapping no speaker stay unassigned and become Unknown."""
    timeline = SpeakerTimeline(segments)
    for segment in result['segments']:
        for word in segment.get('words', []):
            if word.get('start') is None or word.get('end') is None:
                continue
            overlaps = timeline.overlaps(word['start'], word['end'])
            if overlaps:
                word['speaker'] = max(overlaps, key=overlaps.get)
    return result


def timed_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The result's words in order (same dict objects), skipping words without timestamps."""
    return [w for s in result['segments'] for w in s.get('words', []) if w.get('start') is not None and w.get('end') is not None]


def speaker_runs(words: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    runs: list[tuple[str, int, int]] = []
    for i, word in enumerate(words):
        speaker = word.get('speaker') or UNKNOWN
        if runs and runs[-1][0] == speaker:
            runs[-1] = (speaker, runs[-1][1], i + 1)
        else:
            runs.append((speaker, i, i + 1))
    return runs


def absorb_short_turns(words: list[dict[str, Any]], segments: list[dict[str, Any]], max_words: int = 2) -> list[dict[str, Any]]:
    """Merge turns of at most max_words words into the adjacent turn whose speaker
    overlaps them most in the diarization (the closer neighbour on a tie)."""
    timeline = SpeakerTimeline(segments)
    changed = True
    while changed:
        changed = False
        runs = speaker_runs(words)
        if len(runs) < 2:
            break
        for index, (speaker, first, last) in enumerate(runs):
            if last - first > max_words:
                continue
            start, end = words[first]['start'], words[last - 1]['end']
            overlaps = timeline.overlaps(start, end)
            options = []
            if index > 0:
                options.append((overlaps.get(runs[index - 1][0], 0.0), -(start - words[first - 1]['end']), runs[index - 1][0]))
            if index + 1 < len(runs):
                options.append((overlaps.get(runs[index + 1][0], 0.0), -(words[last]['start'] - end), runs[index + 1][0]))
            target = max(options)[2]
            for word in words[first:last]:
                word['speaker'] = target
            changed = True
            break
    return words


def snap_speaker_changes(words: list[dict[str, Any]], min_pause: float = 0.2, max_shift: float = 1.5) -> list[dict[str, Any]]:
    """Move speaker changes that fall inside a sentence to a sentence boundary.

    A change already sits on a boundary when the pause before it is at least
    min_pause seconds or the previous word ends a sentence. Otherwise it moves to
    the nearest such boundary within max_shift seconds (never past the turn's
    other speaker changes). Without one, a change at a real gap stays put; a
    change between two words with no gap moves to the nearest gap in the turn,
    or else the shorter side joins its neighbour, so a turn is never split
    between touching words.
    """
    def gap(k: int) -> float:
        return words[k]['start'] - words[k - 1]['end']

    def on_boundary(k: int) -> bool:
        return gap(k) >= min_pause or words[k - 1].get('word', '').rstrip('"\')]').endswith(SENTENCE_END)

    def middle(k: int) -> float:
        return (words[k - 1]['end'] + words[k]['start']) / 2

    def speaker(k: int) -> str:
        return words[k].get('speaker') or UNKNOWN

    for _ in range(len(words)):
        changes = [k for k in range(1, len(words)) if speaker(k) != speaker(k - 1)]
        pending = next(((n, k) for n, k in enumerate(changes) if not on_boundary(k) and not words[k].get('_snapped')), None)
        if pending is None:
            break
        n, k = pending
        low = changes[n - 1] if n > 0 else 0
        high = changes[n + 1] if n + 1 < len(changes) else len(words)
        inside = [c for c in range(low + 1, high) if c != k]
        strong = [c for c in inside if on_boundary(c) and abs(middle(c) - middle(k)) <= max_shift]
        if strong:
            target = min(strong, key=lambda c: abs(middle(c) - middle(k)))
        elif gap(k) > 0:
            words[k]['_snapped'] = True  # A real pause, if short; leave the change here.
            continue
        else:
            weak = [c for c in inside if gap(c) > 0]
            target = min(weak, key=lambda c: abs(middle(c) - middle(k))) if weak else (low if k - low <= high - k else high)
        if target < k:
            for word in words[target:k]:
                word['speaker'] = speaker(k)
        else:
            for word in words[k:target]:
                word['speaker'] = speaker(k - 1)
    for word in words:
        word.pop('_snapped', None)
    return words


def build_transcript(job_id: str, duration: float, result: dict[str, Any]) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    speakers: dict[str, dict[str, Any]] = {}
    for segment in result['segments']:
        for raw in segment.get('words', []):
            # WhisperX may omit timestamps for punctuation/numerals. Retain these
            # words at the previous word's end (or segment start), never drop text.
            fallback = turns[-1]['end'] if turns else finite_time(segment.get('start'), 0)
            start = max(0, min(duration, finite_time(raw.get('start'), fallback)))
            end = max(start, min(duration, finite_time(raw.get('end'), start)))
            speaker = raw.get('speaker') or 'Unknown'
            if speaker not in speakers:
                index = sum(s['id'] != 'Unknown' for s in speakers.values())
                speakers[speaker] = {'id': speaker, 'label': 'Unknown' if speaker == 'Unknown' else f'Person {index + 1}', 'color_index': index % 8}
            score = raw.get('score')
            score = float(score) if score is not None and isfinite(float(score)) else None
            word = {'w': raw.get('word', '').strip(), 'start': start, 'end': end, 'score': score}
            if not turns or turns[-1]['speaker_id'] != speaker:
                turns.append({'id': len(turns), 'speaker_id': speaker, 'start': start, 'end': end, 'text': '', 'words': []})
            turn = turns[-1]
            turn['words'].append(word)
            turn['end'] = max(turn['end'], end)
            turn['text'] = ' '.join(w['w'] for w in turn['words'])
    return {'job_id': job_id, 'duration_sec': duration, 'speakers': list(speakers.values()), 'turns': turns}
