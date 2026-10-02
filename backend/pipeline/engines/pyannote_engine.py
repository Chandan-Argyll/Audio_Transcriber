"""Optional pyannote Community-1 diarization (PyTorch, CPU or CUDA).

Install with requirements-pyannote.txt; nothing here is imported unless
DIARIZATION_ENGINE=pyannote, so the default ONNX install needs no PyTorch.
"""
from pathlib import Path
from typing import Any

from app.config import get_settings
from pipeline.audio import SAMPLE_RATE, read_wav, to_float

MODEL = 'pyannote/speaker-diarization-community-1'


def to_segments(annotation: Any) -> list[dict[str, Any]]:
    return [{'start': float(turn.start), 'end': float(turn.end), 'speaker': str(label)}
            for turn, _, label in annotation.itertracks(yield_label=True)]


class PyannoteEngine:
    def __init__(self) -> None:
        self.model: Any = None

    def diarize(self, audio: Path, duration: float, options: dict[str, int]) -> list[dict[str, Any]]:
        import torch
        from pyannote.audio import Pipeline
        settings = get_settings()
        if not settings.hf_token:
            raise ValueError('Set HF_TOKEN and accept the community-1 model conditions on Hugging Face')
        if self.model is None:
            torch.set_num_threads(settings.num_threads)
            self.model = Pipeline.from_pretrained(MODEL, token=settings.hf_token)
        self.model.to(torch.device(settings.device))
        limits = dict(options)
        if 'num_speakers' not in limits:
            limits = {'min_speakers': 1, 'max_speakers': settings.max_speakers, **limits}
        # In-memory waveform: avoids pyannote's file decoding (torchcodec/FFmpeg DLLs on Windows).
        waveform = torch.from_numpy(to_float(read_wav(audio))).unsqueeze(0)
        output = self.model({'waveform': waveform, 'sample_rate': SAMPLE_RATE}, **limits)
        # pyannote.audio 4 returns both; the exclusive (one speaker at a time)
        # annotation is the one intended for reconciling with word timestamps.
        annotation = getattr(output, 'exclusive_speaker_diarization', None) or getattr(output, 'speaker_diarization', output)
        if settings.free_gpu_memory and settings.device == 'cuda':
            self.model.to(torch.device('cpu'))
            torch.cuda.empty_cache()
        return to_segments(annotation)
