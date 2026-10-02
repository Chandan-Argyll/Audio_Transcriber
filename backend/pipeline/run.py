import logging
import time
from functools import lru_cache
from typing import Any
from app.config import get_settings
from app.models import Job, SessionLocal
from app.storage import storage
from pipeline.engines.base import DiarizationEngine, TranscriptionEngine
from pipeline.engines.mock import MockEngine
from pipeline.merge import assign_speakers, build_transcript
from pipeline.preprocess import normalize


@lru_cache
def engines() -> tuple[TranscriptionEngine, DiarizationEngine]:
    settings = get_settings()
    stt, diar = settings.active_engines()
    if stt == 'mock':
        transcription = MockEngine()
    elif stt == 'parakeet_onnx':
        from pipeline.engines.parakeet_onnx import ParakeetOnnxEngine
        transcription = ParakeetOnnxEngine()
    elif stt == 'whisperx':
        from pipeline.engines.whisperx_engine import WhisperXEngine
        transcription = WhisperXEngine()
    else:
        raise ValueError(f'Unknown transcription engine: {stt}')
    if diar == 'mock':
        diarization = MockEngine()
    elif diar == 'sherpa_onnx':
        from pipeline.engines.sherpa_diarization import SherpaOnnxDiarizationEngine
        diarization = SherpaOnnxDiarizationEngine()
    elif diar == 'pyannote':
        from pipeline.engines.pyannote_engine import PyannoteEngine
        diarization = PyannoteEngine()
    else:
        raise ValueError(f'Unknown diarization engine: {diar}')
    return transcription, diarization


logger = logging.getLogger('thread.run')


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
        update(job_id, stage='transcribing', progress=25, stage_timings=lap('preprocess'))
        result = transcription.transcribe(audio, duration)
        update(job_id, stage='aligning', progress=50, stage_timings=lap('transcribe'))
        result = transcription.align(result, audio)
        update(job_id, stage='diarizing', progress=70, stage_timings=lap('transcribe'))
        segments = diarization.diarize(audio, duration, options)
        update(job_id, stage='merging', progress=90, stage_timings=lap('diarize'))
        result = assign_speakers(segments, result)
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
