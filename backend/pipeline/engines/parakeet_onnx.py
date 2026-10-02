import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from pipeline.chunking import offset_words, merge_tokens, plan_chunks, quietest_point
from pipeline.model_store import ensure_model

logger = logging.getLogger('thread.parakeet')
SAMPLE_RATE = 16000


def read_wav(audio: Path) -> np.ndarray:
    """16 kHz mono int16 samples (half the memory of float32 for multi-hour hearings)."""
    import soundfile
    samples, rate = soundfile.read(str(audio), dtype='int16', always_2d=False)
    if rate != SAMPLE_RATE or samples.ndim != 1:
        raise ValueError(f'Expected 16 kHz mono audio, got {rate} Hz with shape {samples.shape}')
    return samples


def to_float(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float32) / 32768


def detect_speech(samples: np.ndarray) -> list[tuple[float, float]]:
    """Silero VAD speech regions in seconds.

    Uses the per-window model directly: sherpa's VoiceActivityDetector buffers a
    whole speech region, which grows without bound on continuous multi-speaker
    audio. Short minimum silence lets brief conversational pauses become cut points.
    """
    import sherpa_onnx
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(ensure_model('silero_vad') / 'silero_vad.onnx')
    config.silero_vad.min_silence_duration = 0.1
    config.silero_vad.min_speech_duration = 0.1
    config.sample_rate = SAMPLE_RATE
    config.num_threads = 1
    model = sherpa_onnx.VadModel.create(config)
    window = model.window_size()
    regions: list[tuple[float, float]] = []
    start: int | None = None
    for offset in range(0, len(samples) - window + 1, window):
        speech = model.is_speech(to_float(samples[offset:offset + window]))
        if speech and start is None:
            start = offset
        elif not speech and start is not None:
            regions.append((start / SAMPLE_RATE, offset / SAMPLE_RATE))
            start = None
    if start is not None:
        regions.append((start / SAMPLE_RATE, len(samples) / SAMPLE_RATE))
    return regions


class ParakeetOnnxEngine:
    """NVIDIA Parakeet TDT 0.6B v2 (English, int8 ONNX) on CPU via sherpa-onnx.

    Word timestamps come from the model's token timestamps, so align() is a no-op.
    """
    def __init__(self) -> None:
        self.recognizer: Any = None

    def _load(self) -> Any:
        if self.recognizer is None:
            import sherpa_onnx
            folder = ensure_model('parakeet')
            threads = get_settings().num_threads
            logger.info('Loading Parakeet TDT 0.6B v2 int8 with %d threads', threads)
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(folder / 'encoder.int8.onnx'), decoder=str(folder / 'decoder.int8.onnx'),
                joiner=str(folder / 'joiner.int8.onnx'), tokens=str(folder / 'tokens.txt'),
                num_threads=threads, model_type='nemo_transducer', decoding_method='greedy_search')
        return self.recognizer

    def transcribe(self, audio: Path, duration: float) -> dict[str, Any]:
        settings = get_settings()
        recognizer = self._load()
        samples = read_wav(audio)
        total = len(samples) / SAMPLE_RATE
        speech = detect_speech(samples)
        chunks = plan_chunks(speech, total, settings.max_chunk_sec,
                             fallback_cut=lambda a, b: quietest_point(samples, SAMPLE_RATE, a, b))
        logger.info('Transcribing %.0fs of audio in %d chunks (max %.0fs each)', total, len(chunks), settings.max_chunk_sec)
        segments = []
        began = time.perf_counter()
        for index, (start, end) in enumerate(chunks, 1):
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, to_float(samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]))
            recognizer.decode_stream(stream)
            result = stream.result
            words = merge_tokens(result.tokens, result.timestamps, getattr(result, 'durations', None),
                                 getattr(result, 'ys_log_probs', None), end_limit=end - start)
            words = offset_words(words, start)
            if words:
                segments.append({'start': words[0]['start'], 'end': words[-1]['end'],
                                 'text': ' '.join(w['word'] for w in words), 'words': words})
            logger.info('Chunk %d/%d (%.0f-%.0fs) done; %.1fs elapsed', index, len(chunks), start, end, time.perf_counter() - began)
        return {'language': 'en', 'segments': segments}

    def align(self, result: dict[str, Any], audio: Path) -> dict[str, Any]:
        return result
