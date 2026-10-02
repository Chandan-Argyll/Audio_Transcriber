import time

import pytest

from app import queue as queue_module
from app.config import get_settings
from app.models import Job, SessionLocal
from app.routers import jobs
from worker import main as worker_main


def wait_for_stage(client, job_id: str, timeout: float = 15) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stage = client.get(f'/api/jobs/{job_id}').json()['stage']
        if stage in {'completed', 'failed'}:
            return stage
        time.sleep(0.05)
    return stage


def test_local_backend_runs_jobs_without_redis(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(get_settings(), 'queue_backend', 'local')
    monkeypatch.setattr(queue_module, 'redis_connection', lambda: pytest.fail('Local mode must not touch Redis'))
    health = client.get('/api/health').json()
    assert health['queue_status'] == 'ready' and health['status'] == 'ok'
    assert health['queue_backend'] == 'local' and health['engine_issue'] is None
    job_id = client.post('/api/jobs', files={'file': ('hearing.wav', wav_bytes)}).json()['id']
    assert wait_for_stage(client, job_id) == 'completed'
    assert len(client.get(f'/api/jobs/{job_id}/transcript').json()['turns']) == 40


def test_local_backend_resumes_unfinished_jobs(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    job_id = client.post('/api/jobs', files={'file': ('hearing.wav', wav_bytes)}).json()['id']
    with SessionLocal() as session:
        session.get(Job, job_id).stage = 'transcribing'
        session.commit()
    monkeypatch.setattr(get_settings(), 'queue_backend', 'redis')
    assert queue_module.resume_local_jobs() == 0
    monkeypatch.setattr(get_settings(), 'queue_backend', 'local')
    assert queue_module.resume_local_jobs() == 1
    assert wait_for_stage(client, job_id) == 'completed'


def test_local_backend_refuses_upload_when_engine_cannot_run(client, wav_bytes, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, 'queue_backend', 'local')
    monkeypatch.setattr(settings, 'diarization_engine', 'pyannote')
    monkeypatch.setattr(settings, 'hf_token', None)
    monkeypatch.setattr('app.config.find_spec', lambda name: object())
    health = client.get('/api/health').json()
    assert health['status'] == 'degraded' and 'HF_TOKEN' in health['engine_issue']
    response = client.post('/api/jobs', files={'file': ('hearing.wav', wav_bytes)})
    assert response.status_code == 503 and 'HF_TOKEN' in response.json()['detail']
    assert client.get('/api/jobs').json() == []


def test_engine_issue_detects_missing_models(monkeypatch):
    settings = get_settings()
    assert settings.engine_issue() is None
    monkeypatch.setattr(settings, 'transcription_engine', 'whisperx')
    monkeypatch.setattr('app.config.find_spec', lambda name: None)
    assert 'not installed' in settings.engine_issue()
    monkeypatch.setattr(settings, 'transcription_engine', 'parakeet')
    assert 'not implemented' in settings.engine_issue()


def test_worker_refuses_to_start_when_engine_cannot_run(monkeypatch):
    monkeypatch.setattr(get_settings(), 'transcription_engine', 'whisperx')
    monkeypatch.setattr('app.config.find_spec', lambda name: None)
    monkeypatch.setattr(worker_main, 'init_db', lambda: pytest.fail('Must refuse before DB or queue startup'))
    with pytest.raises(RuntimeError, match='not installed'):
        worker_main.main()


def test_speaker_limit_is_configurable(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    assert client.post('/api/jobs', files={'file': ('a.wav', wav_bytes)}, data={'num_speakers': '14'}).status_code == 202
    monkeypatch.setattr(get_settings(), 'max_speakers', 8)
    assert client.post('/api/jobs', files={'file': ('a.wav', wav_bytes)}, data={'num_speakers': '9'}).status_code == 400


def test_long_recordings_get_a_proportional_timeout(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    job_id = client.post('/api/jobs', files={'file': ('a.wav', wav_bytes)}).json()['id']
    assert queue_module.job_timeout(job_id) == get_settings().job_timeout_sec
    with SessionLocal() as session:
        session.get(Job, job_id).duration_sec = 3 * 3600
        session.commit()
    assert queue_module.job_timeout(job_id) == 3 * 3 * 3600
