import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from app.config import get_settings, installed, speaker_options, vocabularies_for
from app.models import Job, SessionLocal
from app.storage import storage
from pipeline.engines.base import DiarizationEngine, TranscriptionEngine
from pipeline.engines.mock import MockEngine
from pipeline.merge import absorb_short_turns, assign_speakers, build_transcript, snap_speaker_changes, timed_words
from pipeline.preprocess import normalize
from pipeline.vocabulary import corrector_for


@lru_cache
def transcription_engine(name: str) -> TranscriptionEngine:
    """One cached instance per engine name, so loaded models persist between jobs."""
    if name == 'mock':
        return MockEngine()
    if name == 'parakeet_onnx':
        from pipeline.engines.parakeet_onnx import ParakeetOnnxEngine
        return ParakeetOnnxEngine()
    if name == 'whisperx':
        from pipeline.engines.whisperx_engine import WhisperXEngine
        return WhisperXEngine()
    raise ValueError(f'Unknown transcription engine: {name}')


@lru_cache
def diarization_engine(name: str) -> DiarizationEngine:
    if name == 'mock':
        return MockEngine()
    if name == 'sherpa_onnx':
        from pipeline.engines.sherpa_diarization import SherpaOnnxDiarizationEngine
        return SherpaOnnxDiarizationEngine()
    if name == 'pyannote':
        from pipeline.engines.pyannote_engine import PyannoteEngine
        return PyannoteEngine()
    raise ValueError(f'Unknown diarization engine: {name}')


@lru_cache
def engines() -> tuple[TranscriptionEngine, DiarizationEngine]:
    stt, diar = get_settings().active_engines()
    return transcription_engine(stt), diarization_engine(diar)


logger = logging.getLogger('thread.run')


def clean_segments(segments: list[dict[str, Any]], audio: Path) -> list[dict[str, Any]]:
    """Fold speakers with < MIN_SPEAKER_SECONDS of speech into the most similar speaker."""
    min_seconds = get_settings().min_speaker_seconds
    if min_seconds <= 0 or not segments:
        return segments
    if not installed('sherpa_onnx'):
        logger.warning('Small-speaker clean-up skipped: sherpa-onnx (embedding model) is not installed')
        return segments
    from pipeline.audio import SAMPLE_RATE, read_wav
    from pipeline.speakers import SherpaEmbedder, merge_small_speakers
    return merge_small_speakers(segments, min_seconds, SherpaEmbedder(read_wav(audio), SAMPLE_RATE))


def attribute_words(result: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Word speakers by maximum overlap, then fold 1-2 word turns into a neighbour
    and move mid-sentence speaker changes to pauses or sentence ends."""
    result = assign_speakers(segments, result)
    words = timed_words(result)
    snap_speaker_changes(absorb_short_turns(words, segments))
    return result


def update(job_id: str, **values: Any) -> bool:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            return False
        for key, value in values.items():
            setattr(job, key, value)
        session.commit()
        return True


def run_job(job_id: str) -> None:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        duration, options = job.duration_sec, job.options
    try:
        settings = get_settings()
        settings.require_safe_worker()
        stt, diar = settings.active_engines()
        # Persist actual executing worker settings even if API settings differed.
        update(job_id, transcription_engine=stt, diarization_engine=diar)
        timings: dict[str, float] = {}
        began = clock = time.perf_counter()

        def lap(name: str) -> dict[str, float]:
            # Persisted after every stage, so partial timings survive a failure.
            nonlocal clock
            now = time.perf_counter()
            timings[name] = round(timings.get(name, 0) + now - clock, 3)
            timings['total'] = round(now - began, 3)
            clock = now
            return dict(timings)

        update(job_id, stage='preprocessing', progress=10, stage_timings=None)
        audio = storage.path(job_id, 'normalized.wav')
        normalize(storage.path(job_id, 'original'), audio, duration)
        transcription, diarization = engines()
        vocabulary = corrector_for(vocabularies_for(options))
        update(job_id, stage='transcribing', progress=25, stage_timings=lap('preprocess'))
        result = transcription.transcribe(audio, duration, hotwords=vocabulary.hotwords)
        update(job_id, stage='aligning', progress=50, stage_timings=lap('transcribe'))
        result = vocabulary.apply(transcription.align(result, audio))
        update(job_id, stage='diarizing', progress=70, stage_timings=lap('transcribe'))
        segments = clean_segments(diarization.diarize(audio, duration, speaker_options(options)), audio)
        update(job_id, stage='merging', progress=90, stage_timings=lap('diarize'))
        result = attribute_words(result, segments)
        transcript = build_transcript(job_id, duration, result)
        transcript.update(transcription_engine=stt, diarization_engine=diar)
        timings = lap('merge')
        logger.info('Job %s: %.0fs audio in %.1fs (%.2fx real time) %s', job_id, duration, timings['total'],
                    duration / timings['total'] if timings['total'] else 0, timings)
        update(job_id, stage='completed', progress=100, transcript=transcript, error=None, stage_timings=timings)
    except Exception as exc:
        # Some upstream exceptions contain auth URLs/tokens. Never persist them.
        token = get_settings().hf_token
        message = str(exc).replace(token, '[redacted]') if token else str(exc)
        update(job_id, stage='failed', error=f'{type(exc).__name__}: {message[:500]}')
