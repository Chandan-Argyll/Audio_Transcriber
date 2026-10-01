import gc
from pathlib import Path
from typing import Any
from app.config import get_settings


class WhisperXEngine:
    """Lazy, per-worker cached large-v3 transcription and English alignment."""
    def __init__(self) -> None:
        self.model: Any = None
        self.alignment: Any = None
        self.metadata: Any = None

    def transcribe(self, audio: Path, duration: float) -> dict[str, Any]:
        import whisperx
        settings = get_settings()
        if self.model is None:
            self.model = whisperx.load_model('large-v3', settings.device, compute_type=settings.compute_type, language='en')
        if settings.free_gpu_memory and self.alignment is not None:
            self.alignment.to('cpu')
        result = self.model.transcribe(whisperx.load_audio(str(audio)), batch_size=settings.batch_size, language='en')
        if settings.free_gpu_memory and settings.device == 'cuda':
            # Offload rather than destroy weights; later jobs reuse loaded models.
            self.model.model.model.unload_model(to_cpu=True)
            self._clear_cache()
        return result

    def align(self, result: dict[str, Any], audio: Path) -> dict[str, Any]:
        import whisperx
        settings = get_settings()
        if self.alignment is None:
            self.alignment, self.metadata = whisperx.load_align_model(language_code='en', device=settings.device)
        else:
            self.alignment.to(settings.device)
        result = whisperx.align(result['segments'], self.alignment, self.metadata, whisperx.load_audio(str(audio)), settings.device, return_char_alignments=False)
        if settings.free_gpu_memory and settings.device == 'cuda':
            self.alignment.to('cpu')
            self._clear_cache()
        return result

    @staticmethod
    def _clear_cache() -> None:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
