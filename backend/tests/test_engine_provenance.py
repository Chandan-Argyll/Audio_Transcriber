import logging
import pytest
from pydantic import ValidationError
from app.config import Settings, get_settings
from app.routers import jobs
from pipeline.run import run_job


def test_mock_provenance_survives_environment_changes(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    job = client.post('/api/jobs', files={'file': ('real-upload.wav', wav_bytes)}).json()
    assert job['transcription_engine'] == job['diarization_engine'] == 'mock'
    run_job(job['id'])
    monkeypatch.setattr(get_settings(), 'transcription_engine', 'whisperx')
    monkeypatch.setattr(get_settings(), 'diarization_engine', 'pyannote')
    for route in ('', '/transcript', '/export?format=json'):
        data = client.get(f"/api/jobs/{job['id']}{route}").json()
        assert data['transcription_engine'] == data['diarization_engine'] == 'mock'


def test_worker_records_executing_engines(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    settings = get_settings()
    monkeypatch.setattr(settings, 'transcription_engine', 'whisperx')
    monkeypatch.setattr(settings, 'diarization_engine', 'pyannote')
    job = client.post('/api/jobs', files={'file': ('recording.wav', wav_bytes)}).json()
    assert job['transcription_engine'] == 'whisperx'
    monkeypatch.setattr(settings, 'transcription_engine', 'mock')
    monkeypatch.setattr(settings, 'diarization_engine', 'mock')
    run_job(job['id'])
    assert client.get(f"/api/jobs/{job['id']}").json()['transcription_engine'] == 'mock'
    assert client.get(f"/api/jobs/{job['id']}/transcript").json()['diarization_engine'] == 'mock'


@pytest.mark.parametrize('engine,stt,diar', [('mock', 'whisperx', 'pyannote'), ('whisperx', 'mock', 'pyannote'), ('whisperx', 'whisperx', 'mock')])
def test_production_refuses_mock_before_connecting_to_redis(monkeypatch, engine, stt, diar):
    from worker import main
    settings = Settings(_env_file=None, app_env='production', engine=engine, transcription_engine=stt, diarization_engine=diar)
    monkeypatch.setattr(main, 'get_settings', lambda: settings)
    monkeypatch.setattr(main, 'init_db', lambda: pytest.fail('Must refuse before DB or queue startup'))
    with pytest.raises(RuntimeError, match='Worker refused to start.*MockEngine'):
        main.main()


def test_environment_validation_and_real_production_configuration():
    Settings(_env_file=None, app_env='production', engine='whisperx', transcription_engine='whisperx', diarization_engine='pyannote').require_safe_worker()
    with pytest.raises(ValidationError):
        Settings(app_env='prod')


def test_startup_log_never_contains_token(monkeypatch, caplog):
    from worker import main
    settings = Settings(_env_file=None, engine='whisperx', transcription_engine='whisperx', diarization_engine='pyannote', device='cuda', hf_token='do-not-log-this-token')
    monkeypatch.setattr(main, 'get_settings', lambda: settings)
    monkeypatch.setattr('app.config.find_spec', lambda name: object())
    def stop_after_log():
        raise RuntimeError('test stops startup')
    monkeypatch.setattr(main, 'init_db', stop_after_log)
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match='test stops'):
        main.main()
    assert 'do-not-log-this-token' not in caplog.text
    for expected in ('whisperx', 'pyannote', 'cuda', 'large-v3', 'speaker-diarization-community-1', 'HF_TOKEN present=True'):
        assert expected in caplog.text


def test_health_exposes_engines_device_without_secret(client, monkeypatch):
    monkeypatch.setattr('app.main.queue_health', lambda: {'queue_status': 'ready', 'worker_count': 1, 'queue_name': 'audio-transcriber', 'redis_endpoint': 'redis://redis:6379/0'})
    data = client.get('/api/health').json()
    assert data['transcription_engine'] == data['diarization_engine'] == 'mock'
    assert data['device'] in {'cpu', 'cuda'}
    assert data['status'] == 'ok'
    assert data['queue_status'] == 'ready'
    assert data['worker_count'] == 1
    assert 'hf_token' not in data
    assert client.get('/api/live').json() == {'status': 'ok'}


def test_cpu_engines_are_the_default_and_explicit_engines_win(monkeypatch):
    for name in ('ENGINE', 'TRANSCRIPTION_ENGINE', 'DIARIZATION_ENGINE'):
        monkeypatch.delenv(name, raising=False)
    assert Settings(_env_file=None).active_engines() == ('parakeet_onnx', 'sherpa_onnx')
    assert Settings(_env_file=None, engine='mock').active_engines() == ('mock', 'mock')
    assert Settings(_env_file=None, engine='whisperx').active_engines() == ('whisperx', 'pyannote')
    assert Settings(_env_file=None, engine='mock', diarization_engine='sherpa_onnx').active_engines() == ('mock', 'sherpa_onnx')
    # Empty values from .env / Compose (TRANSCRIPTION_ENGINE=) fall back to the defaults.
    assert Settings(_env_file=None, transcription_engine='', diarization_engine='pyannote').active_engines() == ('parakeet_onnx', 'pyannote')
    assert 'Unknown TRANSCRIPTION_ENGINE "nemo"' in Settings(_env_file=None, engine='nemo').engine_issue()
    Settings(_env_file=None, app_env='production').require_safe_worker()


def test_data_defaults_live_outside_synced_folders(monkeypatch, tmp_path):
    for name in ('DATABASE_URL', 'STORAGE_DIR', 'MODELS_DIR'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    settings = Settings(_env_file=None)
    root = tmp_path / 'transcriber'
    assert settings.database_url == f'sqlite:///{(root / "jobs.db").as_posix()}'
    assert settings.storage_dir == root / 'files'
    assert settings.models_dir == root / 'models'
    monkeypatch.delenv('LOCALAPPDATA')
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'xdg'))
    assert Settings(_env_file=None).storage_dir == tmp_path / 'xdg' / 'transcriber' / 'files'
