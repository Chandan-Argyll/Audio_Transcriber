from pathlib import Path
from typing import Any
from app.config import get_settings


class PyannoteEngine:
    def __init__(self) -> None:
        self.model: Any = None

    def diarize(self, audio: Path, duration: float, options: dict[str, int]) -> Any:
        from whisperx.diarize import DiarizationPipeline
        import whisperx
        import torch
        settings = get_settings()
        if not settings.hf_token:
            raise ValueError('Set HF_TOKEN and accept the community-1 model conditions on Hugging Face')
        if self.model is None:
            self.model = DiarizationPipeline(model_name='pyannote/speaker-diarization-community-1', token=settings.hf_token, device=settings.device)
        else:
            self.model.model.to(torch.device(settings.device))
        limits = dict(options)
        if 'num_speakers' not in limits:
            limits = {'min_speakers': 1, 'max_speakers': settings.max_speakers, **limits}
        result = self.model(whisperx.load_audio(str(audio)), **limits)
        if settings.free_gpu_memory and settings.device == 'cuda':
            self.model.model.to(torch.device('cpu'))
            torch.cuda.empty_cache()
        return result
