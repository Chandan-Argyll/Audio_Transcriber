from app.routers import jobs
from app.config import get_settings
from pipeline.run import run_job


def test_full_mock_flow(client, wav_bytes, monkeypatch):
    queued = []
    monkeypatch.setattr(jobs, 'enqueue', queued.append)
    response = client.post('/api/jobs', files={'file': ('meeting.wav', wav_bytes, 'audio/wav')})
    assert response.status_code == 202, response.text
    job_id = response.json()['id']
    assert queued == [job_id]
    assert client.get(f'/api/jobs/{job_id}/transcript').status_code == 409
    run_job(job_id)
    assert client.get(f'/api/jobs/{job_id}').json()['stage'] == 'completed'
    transcript = client.get(f'/api/jobs/{job_id}/transcript').json()
    assert len(transcript['turns']) == 40
    assert len(transcript['speakers']) == 3
    response = client.patch(f'/api/jobs/{job_id}/speakers', json={'Person 1': 'Alice'})
    assert response.json()['speakers'][0]['label'] == 'Alice'
    assert 'Alice:' in client.get(f'/api/jobs/{job_id}/export?format=txt').text
    assert client.get(f'/api/jobs/{job_id}/export?format=srt').status_code == 200
    assert client.get(f'/api/jobs/{job_id}/export?format=json').json()['job_id'] == job_id
    response = client.get(f'/api/jobs/{job_id}/audio', headers={'Range': 'bytes=0-43'})
    assert response.status_code == 206 and len(response.content) == 44
    assert client.get(f'/api/jobs/{job_id}/audio', headers={'Range': 'bytes=-10'}).status_code == 206
    assert client.get(f'/api/jobs/{job_id}/audio', headers={'Range': 'bytes=999999-'}).status_code == 416
    assert len(client.get('/api/jobs').json()) == 1
    assert client.delete(f'/api/jobs/{job_id}').status_code == 204
    assert client.get(f'/api/jobs/{job_id}').status_code == 404


def test_bad_files_options_limits(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    assert client.post('/api/jobs', files={'file': ('fake.wav', b'not audio')}).status_code == 400
    assert client.post('/api/jobs', files={'file': ('a.wav', wav_bytes)}, data={'num_speakers': '9'}).status_code == 400
    assert client.post('/api/jobs', files={'file': ('a.wav', wav_bytes)}, data={'min_speakers': '5', 'max_speakers': '2'}).status_code == 400
    monkeypatch.setattr(get_settings(), 'max_file_size_mb', 0)
    assert client.post('/api/jobs', files={'file': ('a.wav', wav_bytes)}).status_code == 413


def test_duration_and_content_validation(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    # Content, not extension: a WAV named as MP3 remains valid.
    assert client.post('/api/jobs', files={'file': ('a.mp3', wav_bytes)}).status_code == 202
    monkeypatch.setattr(get_settings(), 'max_duration_sec', .1)
    assert client.post('/api/jobs', files={'file': ('a.wav', wav_bytes)}).status_code == 400
