"""Speaker clean-up after diarization: fold tiny clusters into the closest real speaker."""
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger('thread.speakers')
Embed = Callable[[list[dict[str, Any]]], np.ndarray | None]


def speech_seconds(segments: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for segment in segments:
        totals[segment['speaker']] += max(0.0, segment['end'] - segment['start'])
    return dict(totals)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def merge_small_speakers(segments: list[dict[str, Any]], min_seconds: float, embed: Embed) -> list[dict[str, Any]]:
    """Relabel every speaker with less than min_seconds of speech as the speaker
    whose average embedding is most similar. embed(speaker_segments) returns that
    average (or None when there is no usable audio; the speaker then joins the
    speaker talking closest in time). Does nothing when no speaker is large enough.
    """
    totals = speech_seconds(segments)
    small = [s for s, total in totals.items() if total < min_seconds]
    large = [s for s, total in totals.items() if total >= min_seconds]
    if not small or not large or min_seconds <= 0:
        return segments
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        by_speaker[segment['speaker']].append(segment)
    centroids = {s: embed(by_speaker[s]) for s in large}
    mapping: dict[str, str] = {}
    for speaker in small:
        vector = embed(by_speaker[speaker])
        candidates = [(cosine(vector, c), s) for s, c in centroids.items() if vector is not None and c is not None]
        if candidates:
            mapping[speaker] = max(candidates)[1]
        else:
            middle = sum((g['start'] + g['end']) / 2 for g in by_speaker[speaker]) / len(by_speaker[speaker])
            mapping[speaker] = min(large, key=lambda s: min(abs((g['start'] + g['end']) / 2 - middle) for g in by_speaker[s]))
    logger.info('Merged %d small speaker(s) (< %.0fs of speech): %s', len(mapping), min_seconds, mapping)
    return [{**segment, 'speaker': mapping.get(segment['speaker'], segment['speaker'])} for segment in segments]


class SherpaEmbedder:
    """Average WeSpeaker embedding over a speaker's longest segments (bounded cost)."""
    def __init__(self, samples: np.ndarray, sample_rate: int, max_segments: int = 10, max_segment_sec: float = 10,
                 min_segment_sec: float = 0.5) -> None:
        import sherpa_onnx
        from app.config import get_settings
        from pipeline.model_store import ensure_model
        model = ensure_model('embedding') / 'wespeaker_en_voxceleb_resnet34_LM.onnx'
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=get_settings().num_threads))
        self.samples, self.rate = samples, sample_rate
        self.max_segments, self.max_segment_sec, self.min_segment_sec = max_segments, max_segment_sec, min_segment_sec

    def __call__(self, segments: list[dict[str, Any]]) -> np.ndarray | None:
        longest = sorted(segments, key=lambda s: s['start'] - s['end'])[:self.max_segments]
        vectors = []
        for segment in longest:
            if segment['end'] - segment['start'] < self.min_segment_sec:
                continue
            first = int(segment['start'] * self.rate)
            last = int(min(segment['end'], segment['start'] + self.max_segment_sec) * self.rate)
            stream = self.extractor.create_stream()
            stream.accept_waveform(self.rate, self.samples[first:last].astype(np.float32) / 32768)
            stream.input_finished()
            if self.extractor.is_ready(stream):
                vector = np.array(self.extractor.compute(stream))
                vectors.append(vector / (np.linalg.norm(vector) + 1e-9))
        return np.mean(vectors, axis=0) if vectors else None
