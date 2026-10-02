import errno
import logging
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Protocol

from app.config import get_settings


logger = logging.getLogger('thread.storage')
_DELETE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)
_RETRYABLE_ERRNOS = {errno.EACCES, errno.EPERM, errno.EBUSY, errno.ENOTEMPTY}
_RETRYABLE_WINERRORS = {5, 32, 33, 145}


class Storage(Protocol):
    """Implement this interface to add object storage (e.g. S3)."""
    def path(self, job_id: str, name: str) -> Path: ...
    def delete(self, job_id: str) -> None: ...


class LocalStorage:
    def _job_folder(self, job_id: str) -> Path:
        if not job_id or not all(c in '0123456789abcdef-' for c in job_id):
            raise ValueError('Invalid storage key')
        root = get_settings().storage_dir.resolve()
        folder = root / job_id
        # Keep the containment check independent of filesystem resolution so a
        # symlink at the job path is never followed outside the storage root.
        try:
            folder.relative_to(root)
        except ValueError as exc:
            raise ValueError('Invalid storage key') from exc
        return folder

    def path(self, job_id: str, name: str) -> Path:
        if name not in {'original', 'normalized.wav'}:
            raise ValueError('Invalid storage key')
        folder = self._job_folder(job_id)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / name

    @staticmethod
    def _is_retryable(error: OSError) -> bool:
        return (
            isinstance(error, PermissionError)
            or error.errno in _RETRYABLE_ERRNOS
            or getattr(error, 'winerror', None) in _RETRYABLE_WINERRORS
        )

    @staticmethod
    def _remove_error(folder: Path, function, path: str, exc_info) -> None:
        error = exc_info[1]
        if isinstance(error, FileNotFoundError):
            return
        candidate = Path(path)
        try:
            candidate.relative_to(folder)
        except ValueError as exc:
            raise RuntimeError('Refusing to delete outside the job folder') from exc
        if not isinstance(error, PermissionError) and getattr(error, 'errno', None) not in {errno.EACCES, errno.EPERM}:
            raise error
        # chmod on a symlink can alter its target, so never use it for links.
        if candidate.is_symlink():
            raise error
        try:
            mode = stat.S_IMODE(os.lstat(candidate).st_mode)
            os.chmod(candidate, mode | stat.S_IWRITE)
        except FileNotFoundError:
            return
        function(path)

    def _delete_once(self, folder: Path) -> None:
        if folder.is_symlink():
            folder.unlink(missing_ok=True)
            return
        if folder.exists():
            shutil.rmtree(
                folder,
                onerror=lambda function, path, exc_info: self._remove_error(folder, function, path, exc_info),
            )

    def delete(self, job_id: str) -> None:
        folder = self._job_folder(job_id)
        for attempt in range(len(_DELETE_RETRY_DELAYS) + 1):
            try:
                self._delete_once(folder)
                return
            except FileNotFoundError:
                return
            except OSError as exc:
                if not self._is_retryable(exc) or attempt == len(_DELETE_RETRY_DELAYS):
                    raise
                delay = _DELETE_RETRY_DELAYS[attempt]
                logger.warning(
                    'Storage for job %s is temporarily locked; retrying deletion in %.2fs',
                    job_id,
                    delay,
                )
                time.sleep(delay)


storage = LocalStorage()
