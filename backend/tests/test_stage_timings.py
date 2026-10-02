from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app import models
from app.routers import jobs
from pipeline.run import run_job


def test_completed_jobs_report_each_stage_time(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    job_id = client.post('/api/jobs', files={'file': ('hearing.wav', wav_bytes)}).json()['id']
    assert client.get(f'/api/jobs/{job_id}').json()['stage_timings'] is None
    run_job(job_id)
    timings = client.get(f'/api/jobs/{job_id}').json()['stage_timings']
    assert set(timings) == {'preprocess', 'transcribe', 'diarize', 'merge', 'total'}
    assert all(value >= 0 for value in timings.values())
    assert timings['total'] >= max(v for k, v in timings.items() if k != 'total')
    assert client.get('/api/jobs').json()[0]['stage_timings'] == timings


def test_failed_jobs_keep_timings_of_finished_stages(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    job_id = client.post('/api/jobs', files={'file': ('hearing.wav', wav_bytes)}).json()['id']

    def broken_diarization(*args):
        raise RuntimeError('diarization crashed')
    monkeypatch.setattr('pipeline.engines.mock.MockEngine.diarize', broken_diarization)
    run_job(job_id)
    job = client.get(f'/api/jobs/{job_id}').json()
    assert job['stage'] == 'failed'
    assert set(job['stage_timings']) == {'preprocess', 'transcribe', 'total'}


def test_legacy_database_gains_stage_timings_column(tmp_path, monkeypatch):
    database = create_engine(f'sqlite:///{tmp_path / "legacy.db"}')
    with database.begin() as connection:
        connection.execute(text("CREATE TABLE jobs (id VARCHAR PRIMARY KEY, filename VARCHAR, transcription_engine VARCHAR(32) NOT NULL DEFAULT 'unknown', diarization_engine VARCHAR(32) NOT NULL DEFAULT 'unknown', stage VARCHAR, progress INTEGER, error VARCHAR, duration_sec FLOAT, created_at DATETIME, options JSON, transcript JSON)"))
        connection.execute(text("INSERT INTO jobs VALUES ('old','a.wav','mock','mock','completed',100,NULL,1,'2026-10-02 00:00:00','{}',NULL)"))
    monkeypatch.setattr(models, 'engine', database)
    monkeypatch.setattr(models, 'SessionLocal', sessionmaker(database))
    models.init_db()
    models.init_db()
    assert 'stage_timings' in {c['name'] for c in inspect(database).get_columns('jobs')}
    with models.SessionLocal() as session:
        assert session.get(models.Job, 'old').stage_timings is None
    database.dispose()
