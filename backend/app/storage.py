from pathlib import Path
from typing import Protocol
from app.config import get_settings


class Storage(Protocol):
    """Implement this interface to add object storage (e.g. S3)."""
    def path(self, job_id: str, name: str) -> Path: ...
    def delete(self, job_id: str) -> None: ...


class LocalStorage:
    def path(self, job_id: str, name: str) -> Path:
        if not all(c in '0123456789abcdef-' for c in job_id) or name not in {'original', 'normalized.wav'}:
            raise ValueError('Invalid storage key')
        folder = get_settings().storage_dir.resolve() / job_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder / name

    def delete(self, job_id: str) -> None:
        folder = self.path(job_id, 'original').parent
        for child in folder.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
        folder.rmdir()


storage = LocalStorage()
