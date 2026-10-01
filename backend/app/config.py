from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Read repository settings even when local commands run from backend/.
    model_config = SettingsConfigDict(env_file=(Path(__file__).resolve().parents[2] / '.env', '.env'), extra='ignore')
    database_url: str = 'sqlite:///./data/jobs.db'
    redis_url: str = 'redis://localhost:6379/0'
    storage_dir: Path = Path('data/files')
    engine: str = 'mock'
    app_env: Literal['development', 'production'] = 'development'
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

    def active_engines(self) -> tuple[str, str]:
        return (self.transcription_engine or self.engine,
                self.diarization_engine or ('pyannote' if self.engine == 'whisperx' else self.engine))

    def require_safe_worker(self) -> None:
        if self.app_env == 'production' and (self.engine == 'mock' or 'mock' in self.active_engines()):
            raise RuntimeError('Worker refused to start: MockEngine is forbidden when APP_ENV=production. Set ENGINE=whisperx and remove mock engine overrides.')

    def engine_info(self) -> dict[str, str]:
        transcription, diarization = self.active_engines()
        return {'transcription_engine': transcription, 'diarization_engine': diarization,
                'device': self.device, 'app_env': self.app_env,
                'transcription_model': {'mock': 'MockEngine (synthetic dialogue)', 'whisperx': 'large-v3', 'parakeet': 'Parakeet (TODO)'}.get(transcription, 'unknown'),
                'diarization_model': {'mock': 'MockEngine (synthetic speakers)', 'pyannote': 'pyannote/speaker-diarization-community-1', 'nemotron': 'nvidia/Nemotron-3-Diarization (TODO)'}.get(diarization, 'unknown')}


@lru_cache
def get_settings() -> Settings:
    return Settings()
