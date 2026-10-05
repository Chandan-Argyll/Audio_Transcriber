import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from pipeline.audio import SAMPLE_RATE, read_wav, to_float
from pipeline.chunking import offset_words, merge_tokens, plan_chunks, quietest_point
from pipeline.model_store import ensure_model

logger = logging.getLogger('thread.parakeet')


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


def encode_hotwords(phrases: list[str], vocabulary: set[str]) -> str:
    """sherpa-onnx per-stream hotwords: model tokens separated by spaces, phrases by '/'.

    Parakeet's tokens.txt has no bpe.vocab, so each word is split greedily into the
    longest pieces in the vocabulary ('▁' marks a word start). Phrases containing a
    character the vocabulary cannot spell are skipped.
    """
    encoded = []
    for phrase in phrases:
        pieces: list[str] = []
        for word in phrase.split():
            text, position = '▁' + word, 0
            while position < len(text):
                end = next((j for j in range(len(text), position, -1) if text[position:j] in vocabulary), None)
                if end is None:
                    break
                pieces.append(text[position:end])
                position = end
            if position < len(text):
                pieces = []
                break
        if pieces:
            encoded.append(' '.join(pieces))
    return '/'.join(encoded)


class ParakeetOnnxEngine:
    """NVIDIA Parakeet TDT 0.6B v2 (English, int8 ONNX) on CPU via sherpa-onnx.

    Word timestamps come from the model's token timestamps, so align() is a no-op.
    """
    def __init__(self) -> None:
        self.recognizer: Any = None
        self.vocabulary: set[str] = set()

    def _load(self) -> Any:
        if self.recognizer is None:
            import sherpa_onnx
            folder = ensure_model('parakeet')
            settings = get_settings()
            # Hotwords need beam search; with HOTWORDS_SCORE=0 keep (slightly faster) greedy decoding.
            biasing = settings.hotwords_score > 0
            logger.info('Loading Parakeet TDT 0.6B v2 int8 with %d threads (%s)', settings.num_threads,
                        f'hotwords score {settings.hotwords_score}' if biasing else 'greedy')
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(folder / 'encoder.int8.onnx'), decoder=str(folder / 'decoder.int8.onnx'),
                joiner=str(folder / 'joiner.int8.onnx'), tokens=str(folder / 'tokens.txt'),
                num_threads=settings.num_threads, model_type='nemo_transducer',
                decoding_method='modified_beam_search' if biasing else 'greedy_search',
                hotwords_score=settings.hotwords_score if biasing else 1.5)
            self.vocabulary = {line.rsplit(' ', 1)[0] for line in (folder / 'tokens.txt').read_text(encoding='utf-8').splitlines()}
        return self.recognizer

    def transcribe(self, audio: Path, duration: float, hotwords: list[str] | None = None) -> dict[str, Any]:
        settings = get_settings()
        recognizer = self._load()
        biasing = encode_hotwords(hotwords or [], self.vocabulary) if settings.hotwords_score > 0 else ''
        samples = read_wav(audio)
        total = len(samples) / SAMPLE_RATE
        speech = detect_speech(samples)
        chunks = plan_chunks(speech, total, settings.max_chunk_sec,
                             fallback_cut=lambda a, b: quietest_point(samples, SAMPLE_RATE, a, b))
        logger.info('Transcribing %.0fs of audio in %d chunks (max %.0fs each)', total, len(chunks), settings.max_chunk_sec)
        segments = []
        began = time.perf_counter()
        for index, (start, end) in enumerate(chunks, 1):
            stream = recognizer.create_stream(hotwords=biasing) if biasing else recognizer.create_stream()
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
