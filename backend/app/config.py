import os
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_data_root() -> Path:
    """Per-user app data outside synced folders such as OneDrive: %LOCALAPPDATA%/transcriber."""
    base = os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share'
    return Path(base) / 'transcriber'


def physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


class Settings(BaseSettings):
    # Read repository settings even when local commands run from backend/.
    model_config = SettingsConfigDict(env_file=(Path(__file__).resolve().parents[2] / '.env', '.env'), extra='ignore')
    database_url: str = 'sqlite:///./data/jobs.db'
    redis_url: str = 'redis://127.0.0.1:6379/0'
    rq_queue_name: str = 'audio-transcriber'
    # redis: separate RQ worker (Compose/production). local: run jobs on a
    # background thread inside the API, so native dev needs no Redis or worker.
    queue_backend: Literal['redis', 'local'] = 'redis'
    storage_dir: Path = Path('data/files')
    engine: str = 'mock'
    app_env: Literal['development', 'production'] = 'development'
    transcription_engine: str | None = None
    diarization_engine: str | None = None
    hf_token: str | None = None
    # sherpa-onnx CPU engines: model downloads, ONNX Runtime threads, and the
    # longest audio chunk handed to Parakeet (chunks are cut at VAD silences).
    models_dir: Path = Field(default_factory=lambda: default_data_root() / 'models')
    num_threads: int = Field(default_factory=physical_cores)
    max_chunk_sec: float = 60
    # sherpa-onnx clustering distance threshold: larger merges more voices into fewer speakers.
    diarization_threshold: float = 0.5
    # large-v3-turbo is several times faster on CPU with similar English accuracy.
    whisper_model: str = 'large-v3'
    # Court sessions often exceed eight participants; pyannote has no fixed cap.
    max_speakers: int = 20
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
                'transcription_model': {'mock': 'MockEngine (synthetic dialogue)', 'whisperx': self.whisper_model, 'parakeet_onnx': 'nvidia/parakeet-tdt-0.6b-v2 (ONNX int8)', 'parakeet': 'Parakeet (TODO)'}.get(transcription, 'unknown'),
                'diarization_model': {'mock': 'MockEngine (synthetic speakers)', 'pyannote': 'pyannote/speaker-diarization-community-1', 'sherpa_onnx': 'pyannote-segmentation-3.0 + WeSpeaker ResNet34 (VoxCeleb, ONNX)', 'nemotron': 'nvidia/Nemotron-3-Diarization (TODO)'}.get(diarization, 'unknown')}

    def engine_issue(self) -> str | None:
        """Why the configured engines cannot run in this process, or None if they can."""
        transcription, diarization = self.active_engines()
        if transcription not in {'mock', 'whisperx', 'parakeet', 'parakeet_onnx'}:
            return f'Unknown TRANSCRIPTION_ENGINE "{transcription}"'
        if diarization not in {'mock', 'pyannote', 'nemotron', 'sherpa_onnx'}:
            return f'Unknown DIARIZATION_ENGINE "{diarization}"'
        if transcription == 'parakeet' or diarization == 'nemotron':
            return 'Parakeet/Nemotron engines are not implemented yet'
        if transcription == 'parakeet_onnx' or diarization == 'sherpa_onnx':
            if find_spec('sherpa_onnx') is None:
                return 'sherpa-onnx is not installed: pip install -r backend/requirements-cpu.txt'
        if transcription == 'whisperx' or diarization == 'pyannote':
            if find_spec('whisperx') is None:
                return 'WhisperX is not installed: pip install -r backend/requirements-models.txt'
        if diarization == 'pyannote' and not self.hf_token:
            return 'HF_TOKEN is missing; pyannote diarization needs a Hugging Face token'
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
