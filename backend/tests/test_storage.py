import errno
import os
import stat
from types import SimpleNamespace

import pytest

from app import storage as storage_module
from app.storage import LocalStorage


@pytest.fixture
def local_storage(tmp_path, monkeypatch):
    root = tmp_path / 'files'
    monkeypatch.setattr(storage_module, 'get_settings', lambda: SimpleNamespace(storage_dir=root))
    return LocalStorage(), root


def test_delete_removes_read_only_storage(local_storage):
    storage, _ = local_storage
    folder = storage.path('abc-123', 'original').parent
    original = folder / 'original'
    nested = folder / 'artifacts'
    nested.mkdir()
    result = nested / 'transcript.json'
    original.write_bytes(b'audio')
    result.write_text('{}', encoding='utf-8')
    original.chmod(stat.S_IREAD)
    result.chmod(stat.S_IREAD)
    folder.chmod(stat.S_IREAD)

    storage.delete('abc-123')

    assert not folder.exists()


def test_delete_clears_read_only_after_permission_error(local_storage, monkeypatch):
    storage, _ = local_storage
    folder = storage.path('abc-123', 'original').parent
    original = folder / 'original'
    original.write_bytes(b'audio')
    original.chmod(stat.S_IREAD)

    def report_windows_read_only(_folder, *, onerror):
        error = PermissionError(errno.EACCES, 'read-only', str(original))
        onerror(os.unlink, str(original), (PermissionError, error, None))
        os.rmdir(folder)

    monkeypatch.setattr(storage_module.shutil, 'rmtree', report_windows_read_only)

    storage.delete('abc-123')

    assert not folder.exists()


def test_delete_retries_a_transient_windows_lock(local_storage, monkeypatch):
    storage, _ = local_storage
    folder = storage.path('abc-123', 'original').parent
    (folder / 'original').write_bytes(b'audio')
    real_rmtree = storage_module.shutil.rmtree
    calls = 0
    delays = []

    def locked_once(path, *, onerror):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError(errno.EACCES, 'used by another process', str(path))
        real_rmtree(path, onerror=onerror)

    monkeypatch.setattr(storage_module.shutil, 'rmtree', locked_once)
    monkeypatch.setattr(storage_module.time, 'sleep', delays.append)

    storage.delete('abc-123')

    assert calls == 2
    assert delays == [0.05]
    assert not folder.exists()


def test_delete_missing_storage_does_not_recreate_it(local_storage):
    storage, root = local_storage

    storage.delete('abc-123')

    assert not root.exists()


def test_delete_does_not_follow_job_directory_symlink(local_storage, tmp_path):
    storage, root = local_storage
    outside = tmp_path / 'outside'
    outside.mkdir()
    keep = outside / 'keep.txt'
    keep.write_text('keep', encoding='utf-8')
    root.mkdir()
    link = root / 'abc-123'
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f'directory symlinks are unavailable: {exc}')

    storage.delete('abc-123')

    assert keep.read_text(encoding='utf-8') == 'keep'
    assert not link.exists()
