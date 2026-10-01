import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app import models


def test_existing_jobs_gain_unknown_provenance_without_data_loss(tmp_path, monkeypatch):
    database = create_engine(f'sqlite:///{tmp_path / "legacy.db"}')
    session_factory = sessionmaker(database)
    transcript = {'job_id': 'legacy', 'duration_sec': 1, 'speakers': [], 'turns': []}
    with database.begin() as connection:
        connection.execute(text('CREATE TABLE jobs (id VARCHAR PRIMARY KEY, filename VARCHAR, stage VARCHAR, progress INTEGER, error VARCHAR, duration_sec FLOAT, created_at DATETIME, options JSON, transcript JSON)'))
        connection.execute(text("INSERT INTO jobs VALUES ('legacy','audio.wav','completed',100,NULL,1,'2026-10-02 00:00:00','{}',:transcript)"), {'transcript': json.dumps(transcript)})
    monkeypatch.setattr(models, 'engine', database)
    monkeypatch.setattr(models, 'SessionLocal', session_factory)
    models.init_db()
    models.init_db()  # Idempotent on subsequent startups.
    with session_factory() as session:
        job = session.get(models.Job, 'legacy')
        assert job.filename == 'audio.wav'
        assert job.transcription_engine == job.diarization_engine == 'unknown'
        assert job.transcript['transcription_engine'] == 'unknown'
    database.dispose()
