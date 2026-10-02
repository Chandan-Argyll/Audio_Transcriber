import io
import tarfile

import pytest

from app.config import get_settings
from pipeline import model_store
from pipeline.model_store import ModelSpec


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), 'models_dir', tmp_path / 'models')
    monkeypatch.setitem(model_store.MODELS, 'single', ModelSpec('https://example.invalid/m.onnx', 'single-model', ('m.onnx',)))
    monkeypatch.setitem(model_store.MODELS, 'archive', ModelSpec('https://example.invalid/a.tar.bz2', 'pkg', ('a.onnx', 'tokens.txt'), archive=True))
    return tmp_path / 'models'


def archive_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:bz2') as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def fake_download(payload: bytes):
    def download(url, target):
        target.write_bytes(payload)
    return download


def test_existing_models_are_not_downloaded_again(store, monkeypatch):
    (store / 'single-model').mkdir(parents=True)
    (store / 'single-model' / 'm.onnx').write_bytes(b'x')
    monkeypatch.setattr(model_store, '_download', lambda *a: pytest.fail('must not download'))
    assert model_store.ensure_model('single') == store / 'single-model'


def test_single_file_and_archive_downloads_are_verified(store, monkeypatch):
    monkeypatch.setattr(model_store, '_download', fake_download(b'weights'))
    assert (model_store.ensure_model('single') / 'm.onnx').read_bytes() == b'weights'
    monkeypatch.setattr(model_store, '_download', fake_download(archive_bytes({'pkg/a.onnx': b'1', 'pkg/tokens.txt': b'2'})))
    folder = model_store.ensure_model('archive')
    assert model_store.is_downloaded('archive') and (folder / 'tokens.txt').read_bytes() == b'2'
    assert [p.name for p in store.iterdir() if p.name.startswith('.download-')] == []


def test_incomplete_or_failed_downloads_never_look_complete(store, monkeypatch):
    monkeypatch.setattr(model_store, '_download', fake_download(archive_bytes({'pkg/a.onnx': b'1'})))
    with pytest.raises(RuntimeError, match='missing tokens.txt'):
        model_store.ensure_model('archive')
    assert not (store / 'pkg').exists()

    def interrupted(url, target):
        target.write_bytes(b'partial')
        raise ConnectionError('network dropped')
    monkeypatch.setattr(model_store, '_download', interrupted)
    with pytest.raises(ConnectionError):
        model_store.ensure_model('single')
    assert not model_store.is_downloaded('single')
    assert list(store.iterdir()) == []


def test_archives_cannot_write_outside_the_models_folder(store, monkeypatch):
    monkeypatch.setattr(model_store, '_download', fake_download(archive_bytes({'../evil.onnx': b'x', 'pkg/a.onnx': b'1', 'pkg/tokens.txt': b'2'})))
    with pytest.raises(RuntimeError, match='Unsafe entry'):
        model_store.ensure_model('archive')
    assert not (store.parent / 'evil.onnx').exists()
