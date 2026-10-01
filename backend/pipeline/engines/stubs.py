from pathlib import Path
from typing import Any


class ParakeetEngine:
    """TODO: integrate NVIDIA NeMo Parakeet and adapt word alignment output."""
    def transcribe(self, audio: Path, duration: float) -> dict[str, Any]:
        raise NotImplementedError('TODO: implement ParakeetEngine with NVIDIA NeMo')

    def align(self, result: dict[str, Any], audio: Path) -> dict[str, Any]:
        raise NotImplementedError('TODO: implement Parakeet word alignment')


class NemotronDiarizationEngine:
    """TODO: integrate nvidia/Nemotron-3-Diarization (maximum eight speakers)."""
    def diarize(self, audio: Path, duration: float, options: dict[str, int]) -> Any:
        raise NotImplementedError('TODO: implement NemotronDiarizationEngine (max 8 speakers)')
