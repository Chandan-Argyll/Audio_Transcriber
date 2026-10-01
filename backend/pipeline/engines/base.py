from pathlib import Path
from typing import Any, Protocol


class TranscriptionEngine(Protocol):
    def transcribe(self, audio: Path, duration: float) -> dict[str, Any]: ...
    def align(self, result: dict[str, Any], audio: Path) -> dict[str, Any]: ...


class DiarizationEngine(Protocol):
    def diarize(self, audio: Path, duration: float, options: dict[str, int]) -> Any: ...
