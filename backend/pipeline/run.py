from functools import lru_cache
from typing import Any
from app.config import get_settings
from app.models import Job, SessionLocal, init_db
from app.storage import storage
from pipeline.engines.base import DiarizationEngine, TranscriptionEngine
from pipeline.engines.mock import MockEngine
from pipeline.merge import assign_mock_speakers, build_transcript
from pipeline.preprocess import normalize


@lru_cache
def engines() -> tuple[TranscriptionEngine, DiarizationEngine]:
    settings = get_settings()
    stt, diar = settings.active_engines()
    if stt == 'mock':
        transcription = MockEngine()
    elif stt == 'whisperx':
        from pipeline.engines.whisperx_engine import WhisperXEngine
        transcription = WhisperXEngine()
    elif stt == 'parakeet':
        from pipeline.engines.stubs import ParakeetEngine
        transcription = ParakeetEngine()
    else:
        raise ValueError(f'Unknown transcription engine: {stt}')
    if diar == 'mock':
        diarization = MockEngine()
    elif diar == 'pyannote':
        from pipeline.engines.pyannote_engine import PyannoteEngine
        diarization = PyannoteEngine()
    elif diar == 'nemotron':
        from pipeline.engines.stubs import NemotronDiarizationEngine
        diarization = NemotronDiarizationEngine()
    else:
        raise ValueError(f'Unknown diarization engine: {diar}')
    return transcription, diarization


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
    init_db()
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
        update(job_id, stage='preprocessing', progress=10)
        audio = storage.path(job_id, 'normalized.wav')
        normalize(storage.path(job_id, 'original'), audio)
        transcription, diarization = engines()
        update(job_id, stage='transcribing', progress=25)
        result = transcription.transcribe(audio, duration)
        update(job_id, stage='aligning', progress=50)
        result = transcription.align(result, audio)
        update(job_id, stage='diarizing', progress=70)
        segments = diarization.diarize(audio, duration, options)
        update(job_id, stage='merging', progress=90)
        if isinstance(segments, list):
            result = assign_mock_speakers(segments, result)
        else:
            import whisperx
            result = whisperx.assign_word_speakers(segments, result)
        transcript = build_transcript(job_id, duration, result)
        transcript.update(transcription_engine=stt, diarization_engine=diar)
        update(job_id, stage='completed', progress=100, transcript=transcript, error=None)
    except Exception as exc:
        # Some upstream exceptions contain auth URLs/tokens. Never persist them.
        token = get_settings().hf_token
        message = str(exc).replace(token, '[redacted]') if token else str(exc)
        update(job_id, stage='failed', error=f'{type(exc).__name__}: {message[:500]}')
