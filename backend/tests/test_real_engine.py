"""Opt-in integration test: actual models, authorized HF token and speech audio."""
import os
from pathlib import Path
import pytest
from app.config import get_settings
from app.routers import jobs
from pipeline.preprocess import probe
from pipeline.run import engines, run_job


@pytest.mark.skipif(os.environ.get('RUN_REAL_ENGINE_TEST') != '1', reason='Set RUN_REAL_ENGINE_TEST=1 and REAL_ENGINE_AUDIO to run downloaded real models')
def test_real_engine_transcript_duration_matches_input(client, monkeypatch):
    source = Path(os.environ.get('REAL_ENGINE_AUDIO', ''))
    assert source.is_file(), 'REAL_ENGINE_AUDIO must point to a real English speech recording'
    settings = get_settings()
    assert settings.hf_token, 'HF_TOKEN is required for actual community-1 inference'
    monkeypatch.setattr(settings, 'engine', 'whisperx')
    monkeypatch.setattr(settings, 'transcription_engine', 'whisperx')
    monkeypatch.setattr(settings, 'diarization_engine', 'pyannote')
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    engines.cache_clear()
    try:
        input_duration = probe(source)
        with source.open('rb') as audio:
            response = client.post('/api/jobs', files={'file': (source.name, audio)})
        assert response.status_code == 202, response.text
        job_id = response.json()['id']
        run_job(job_id)  # Real WhisperX and pyannote; no inference mocks.
        status = client.get(f'/api/jobs/{job_id}').json()
        assert status['stage'] == 'completed', status['error']
        transcript = client.get(f'/api/jobs/{job_id}/transcript').json()
        assert transcript['transcription_engine'] == 'whisperx'
        assert transcript['diarization_engine'] == 'pyannote'
        assert transcript['turns'], 'Speech recording must yield actual transcript turns'
        assert abs(transcript['duration_sec'] - input_duration) <= 2
        assert all(0 <= turn['start'] <= turn['end'] <= input_duration + 2 for turn in transcript['turns'])
    finally:
        engines.cache_clear()
