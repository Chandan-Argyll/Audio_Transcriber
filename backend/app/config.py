from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    database_url: str = 'sqlite:///./data/jobs.db'
    redis_url: str = 'redis://localhost:6379/0'
    storage_dir: Path = Path('data/files')
    engine: str = 'mock'
    transcription_engine: str | None = None
    diarization_engine: str | None = None
    hf_token: str | None = None
    device: str = 'cpu'
    compute_type: str = 'int8'
    batch_size: int = 2
    free_gpu_memory: bool = False
    max_file_size_mb: int = 500
    max_duration_sec: float = 10800
    mock_delay_sec: float = 0.4
    cors_origins: str = 'http://localhost:5173'
    job_timeout_sec: int = 14400
    ffmpeg_binary: str = 'ffmpeg'
    ffprobe_binary: str = 'ffprobe'


@lru_cache
def get_settings() -> Settings:
    return Settings()
