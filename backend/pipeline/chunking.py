"""Model-free helpers for chunked CPU transcription: where to cut long audio and
how to turn sub-word token timestamps into word timestamps."""
from collections.abc import Callable, Sequence
from math import exp
from typing import Any

import numpy as np

# Parakeet's encoder emits one frame every 80 ms; a token never lasts less.
FRAME_SEC = 0.08
# VAD gaps shorter than this are treated as a forced split inside speech, not a pause.
MIN_PAUSE_SEC = 0.2


def merge_speech(speech: Sequence[tuple[float, float]], min_pause: float = MIN_PAUSE_SEC) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(speech):
        if merged and start - merged[-1][1] < min_pause:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def plan_chunks(speech: Sequence[tuple[float, float]], duration: float, max_len: float,
                fallback_cut: Callable[[float, float], float] | None = None) -> list[tuple[float, float]]:
    """Split [0, duration] into chunks of at most max_len seconds.

    Cuts go in the middle of pauses between speech regions, so words near a cut
    stay whole and nothing at a VAD edge is lost. A stretch of speech with no
    pause inside the window is cut at fallback_cut(window_start, window_end)
    (the quietest point), or at the window end without one. Chunks containing
    no speech are dropped.
    """
    regions = merge_speech(speech)
    cuts = [(a_end + b_start) / 2 for (_, a_end), (b_start, _) in zip(regions, regions[1:])]
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > max_len:
        limit = start + max_len
        inside = [cut for cut in cuts if start < cut <= limit]
        if inside:
            cut = inside[-1]
        else:
            cut = fallback_cut(start, limit) if fallback_cut else limit
            cut = min(max(cut, start + max_len / 2), limit)
        chunks.append((start, cut))
        start = cut
    chunks.append((start, duration))
    return [(a, b) for a, b in chunks if any(s < b and e > a for s, e in regions)]


def quietest_point(samples: np.ndarray, sample_rate: int, start: float, end: float, frame_sec: float = 0.02) -> float:
    """Time of the lowest-energy frame in the second half of [start, end]."""
    first = int((start + (end - start) / 2) * sample_rate)
    last = min(int(end * sample_rate), len(samples))
    frame = max(1, int(frame_sec * sample_rate))
    count = (last - first) // frame
    if count < 1:
        return end
    window = samples[first:first + count * frame].astype(np.float32).reshape(count, frame)
    quietest = int(np.argmin(np.square(window).mean(axis=1)))
    return (first + quietest * frame + frame / 2) / sample_rate


def merge_tokens(tokens: Sequence[str], starts: Sequence[float], durations: Sequence[float] | None = None,
                 log_probs: Sequence[float] | None = None, end_limit: float | None = None) -> list[dict[str, Any]]:
    """Merge SentencePiece sub-word tokens into words with start/end/score.

    A token starting with a space or '▁' begins a new word; other tokens
    (including punctuation) continue the current word. The vocabulary has no
    '▁'-prefixed piece for some characters (digits, rare letter runs), so the
    model emits a standalone '▁' token instead: it marks that the *next* token
    starts a word ("in", "▁", "2", "0", "0", "6" -> "in 2006"). A word ends at
    its last token's start + duration (at least one frame) and never overlaps
    the next word.
    """
    words: list[dict[str, Any]] = []
    probs: list[list[float]] = []
    word_start_pending = False
    for i, token in enumerate(tokens):
        text = token.replace('▁', ' ')
        start = float(starts[i])
        length = float(durations[i]) if durations is not None and i < len(durations) else 0.0
        end = start + max(length, FRAME_SEC)
        prob = exp(float(log_probs[i])) if log_probs is not None and i < len(log_probs) else None
        if not text.strip():
            # A bare word-start marker: remember it for the next real token.
            word_start_pending = word_start_pending or bool(text)
            continue
        if text.startswith(' ') or word_start_pending or not words:
            word_start_pending = False
            words.append({'word': text.strip(), 'start': start, 'end': end, 'score': None})
            probs.append([])
        else:
            words[-1]['word'] += text.strip()
            # Punctuation is emitted at the word's last frame; it must not stretch the word.
            if any(c.isalnum() for c in text):
                words[-1]['end'] = max(words[-1]['end'], end)
        if prob is not None:
            probs[-1].append(prob)
    for word, values, following in zip(words, probs, words[1:] + [None]):
        if following is not None:
            word['end'] = min(word['end'], following['start'])
        if end_limit is not None:
            word['end'] = min(word['end'], end_limit)
        word['end'] = max(word['end'], word['start'])
        word['score'] = round(sum(values) / len(values), 4) if values else None
    return words


def offset_words(words: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    return [{**word, 'start': round(word['start'] + offset, 3), 'end': round(word['end'] + offset, 3)} for word in words]
