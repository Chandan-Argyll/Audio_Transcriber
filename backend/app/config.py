import os
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
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


def installed(module: str) -> bool:
    """True if importable, without importing it (heavy packages such as torch stay unloaded)."""
    try:
        return find_spec(module) is not None
    except ModuleNotFoundError:  # Parent package of a dotted name is missing.
        return False


TRANSCRIPTION_ENGINES = ('parakeet_onnx', 'whisperx', 'mock')
DIARIZATION_ENGINES = ('sherpa_onnx', 'pyannote', 'mock')
# Legacy ENGINE shorthand; TRANSCRIPTION_ENGINE / DIARIZATION_ENGINE always win.
# Recording types offered in the upload pane: speaker counts passed to diarization
# and correction dictionaries (backend/vocab/<name>.json) applied after transcription.
RECORDING_PRESETS: dict[str, dict] = {
    'general': {'label': 'General', 'speakers': {}, 'vocabularies': []},
    'supreme_court': {'label': 'Supreme Court argument', 'speakers': {'min_speakers': 10, 'max_speakers': 14}, 'vocabularies': ['legal']},
    'meeting': {'label': 'Meeting', 'speakers': {'min_speakers': 2, 'max_speakers': 8}, 'vocabularies': []},
}
ENGINE_PRESETS = {'cpu': ('parakeet_onnx', 'sherpa_onnx'), 'whisperx': ('whisperx', 'pyannote'), 'mock': ('mock', 'mock')}


class Settings(BaseSettings):
    # Read repository settings even when local commands run from backend/.
    model_config = SettingsConfigDict(env_file=(Path(__file__).resolve().parents[2] / '.env', '.env'), extra='ignore')
    # Defaults live outside synced folders: OneDrive locks and syncing corrupt SQLite.
    database_url: str = Field(default_factory=lambda: f"sqlite:///{(default_data_root() / 'jobs.db').as_posix()}")
    redis_url: str = 'redis://127.0.0.1:6379/0'
    rq_queue_name: str = 'audio-transcriber'
    # redis: separate RQ worker (Compose/production). local: run jobs on a
    # background thread inside the API, so native dev needs no Redis or worker.
    queue_backend: Literal['redis', 'local'] = 'redis'
    storage_dir: Path = Field(default_factory=lambda: default_data_root() / 'files')
    engine: str | None = None
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
    # Speakers with less total speech are merged into the most similar speaker (0 disables).
    min_speaker_seconds: float = 10
    # Parakeet contextual biasing with dictionary hotwords (> 0 enables it). Off by
    # default: sherpa-onnx hotwords need modified beam search, which drops words on
    # Parakeet TDT (measured), and scores of 4+ corrupt neighbouring words. The
    # correction dictionaries do the vocabulary fixing instead.
    hotwords_score: float = 0
    # Correction dictionaries applied to every job (comma separated names or .json paths).
    vocabularies: str = ''
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

    @field_validator('num_threads', mode='before')
    @classmethod
    def default_threads(cls, value: object) -> object:
        # Compose passes NUM_THREADS= when unset; empty means "use physical cores".
        return physical_cores() if value in ('', None) else value

    def active_engines(self) -> tuple[str, str]:
        # An unknown ENGINE value passes through so engine_issue() can name it.
        preset = ENGINE_PRESETS.get(self.engine or 'cpu', (self.engine, self.engine))
        return self.transcription_engine or preset[0], self.diarization_engine or preset[1]

    def require_safe_worker(self) -> None:
        if self.app_env == 'production' and (self.engine == 'mock' or 'mock' in self.active_engines()):
            raise RuntimeError('Worker refused to start: MockEngine is forbidden when APP_ENV=production. Set TRANSCRIPTION_ENGINE=parakeet_onnx and DIARIZATION_ENGINE=sherpa_onnx and remove ENGINE=mock.')

    def engine_info(self) -> dict[str, str]:
        transcription, diarization = self.active_engines()
        return {'transcription_engine': transcription, 'diarization_engine': diarization,
                'device': self.device, 'app_env': self.app_env,
                'transcription_model': {'mock': 'MockEngine (synthetic dialogue)', 'whisperx': self.whisper_model, 'parakeet_onnx': 'nvidia/parakeet-tdt-0.6b-v2 (ONNX int8)'}.get(transcription, 'unknown'),
                'diarization_model': {'mock': 'MockEngine (synthetic speakers)', 'pyannote': 'pyannote/speaker-diarization-community-1', 'sherpa_onnx': 'pyannote-segmentation-3.0 + WeSpeaker ResNet34 (VoxCeleb, ONNX)'}.get(diarization, 'unknown')}

    def engine_issue(self) -> str | None:
        """Why the configured engines cannot run in this process, or None if they can."""
        transcription, diarization = self.active_engines()
        if transcription not in TRANSCRIPTION_ENGINES:
            return f'Unknown TRANSCRIPTION_ENGINE "{transcription}"; use {" | ".join(TRANSCRIPTION_ENGINES)}'
        if diarization not in DIARIZATION_ENGINES:
            return f'Unknown DIARIZATION_ENGINE "{diarization}"; use {" | ".join(DIARIZATION_ENGINES)}'
        if transcription == 'parakeet_onnx' or diarization == 'sherpa_onnx':
            if not installed('sherpa_onnx'):
                return 'sherpa-onnx is not installed: pip install -r backend/requirements-cpu.txt'
        if transcription == 'whisperx' and not installed('whisperx'):
            return 'WhisperX is not installed: pip install -r backend/requirements-models.txt'
        if diarization == 'pyannote' and not (installed('torch') and installed('pyannote.audio')):
            return 'pyannote is optional and not installed: pip install -r backend/requirements-pyannote.txt'
        if diarization == 'pyannote' and not self.hf_token:
            return 'HF_TOKEN is missing; pyannote diarization needs a Hugging Face token'
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def speaker_options(options: dict) -> dict[str, int]:
    """Only the diarization speaker counts from a job's options."""
    return {k: v for k, v in options.items() if k in ('num_speakers', 'min_speakers', 'max_speakers')}


def vocabularies_for(options: dict) -> list[str]:
    configured = [name.strip() for name in get_settings().vocabularies.split(',') if name.strip()]
    preset = RECORDING_PRESETS.get(options.get('recording_type', 'general'), RECORDING_PRESETS['general'])
    return list(dict.fromkeys(configured + preset['vocabularies']))
