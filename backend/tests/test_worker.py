from app.config import get_settings
from app.routers import jobs
from pipeline.run import run_job


def test_worker_failure_is_exposed_and_redacts_token(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    response = client.post('/api/jobs', files={'file': ('meeting.wav', wav_bytes)})
    job_id = response.json()['id']
    monkeypatch.setattr(get_settings(), 'hf_token', 'secret-example-token')
    def fail(*args):
        raise RuntimeError('Upstream rejected secret-example-token')
    monkeypatch.setattr('pipeline.run.normalize', fail)
    run_job(job_id)
    status = client.get(f'/api/jobs/{job_id}').json()
    assert status['stage'] == status['status'] == 'failed'
    assert 'secret-example-token' not in status['error']
    assert '[redacted]' in status['error']
    assert client.get(f'/api/jobs/{job_id}/transcript').status_code == 500


def test_queue_failure_and_processing_delete(client, wav_bytes, monkeypatch):
    def fail(*args):
        raise RuntimeError('Redis offline')
    monkeypatch.setattr(jobs, 'enqueue', fail)
    assert client.post('/api/jobs', files={'file': ('meeting.wav', wav_bytes)}).status_code == 503
    assert client.get('/api/jobs').json()[0]['stage'] == 'failed'
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    job_id = client.post('/api/jobs', files={'file': ('meeting.wav', wav_bytes)}).json()['id']
    assert client.delete(f'/api/jobs/{job_id}').status_code == 409
