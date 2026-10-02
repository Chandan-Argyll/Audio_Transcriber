"""First-use download of the sherpa-onnx CPU models into MODELS_DIR."""
import logging
import shutil
import tarfile
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger('thread.models')
RELEASES = 'https://github.com/k2-fsa/sherpa-onnx/releases/download'


@dataclass(frozen=True)
class ModelSpec:
    url: str
    folder: str
    files: tuple[str, ...]
    archive: bool = False


MODELS = {
    'parakeet': ModelSpec(f'{RELEASES}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2',
                          'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8',
                          ('encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt'), archive=True),
    'silero_vad': ModelSpec(f'{RELEASES}/asr-models/silero_vad.onnx', 'silero-vad', ('silero_vad.onnx',)),
    'segmentation': ModelSpec(f'{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2',
                              'sherpa-onnx-pyannote-segmentation-3-0', ('model.onnx',), archive=True),
    # English VoxCeleb-trained WeSpeaker ResNet34 (large-margin fine-tuned), not the zh-cn example models.
    'embedding': ModelSpec(f'{RELEASES}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx',
                           'wespeaker-en-voxceleb-resnet34-LM', ('wespeaker_en_voxceleb_resnet34_LM.onnx',)),
}
_lock = threading.Lock()


def model_dir(name: str) -> Path:
    return get_settings().models_dir / MODELS[name].folder


def is_downloaded(name: str) -> bool:
    folder = model_dir(name)
    return all((folder / file).is_file() for file in MODELS[name].files)


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, target.open('wb') as output:
        total = int(response.headers.get('Content-Length') or 0)
        done, next_report = 0, 10
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            done += len(chunk)
            if total and done * 100 // total >= next_report:
                logger.info('Downloading %s: %d%% (%.0f / %.0f MB)', target.name, done * 100 // total, done / 2**20, total / 2**20)
                next_report += 10


def _extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, 'r:bz2') as tar:
        root = destination.resolve()
        for member in tar.getmembers():
            if not (member.isfile() or member.isdir()) or not (root / member.name).resolve().is_relative_to(root):
                raise RuntimeError(f'Unsafe entry in model archive: {member.name}')
        tar.extractall(destination)


def ensure_model(name: str) -> Path:
    """Return the folder holding the model files, downloading them on first use."""
    spec = MODELS[name]
    folder = model_dir(name)
    with _lock:
        if is_downloaded(name):
            return folder
        folder.parent.mkdir(parents=True, exist_ok=True)
        logger.info('Model %s not found in %s; downloading %s', name, folder.parent, spec.url)
        # Assemble in a sibling temp folder and rename, so an interrupted
        # download never leaves a folder that looks complete.
        with tempfile.TemporaryDirectory(dir=folder.parent, prefix='.download-') as temp:
            staging = Path(temp) / spec.folder
            download = Path(temp) / spec.url.rsplit('/', 1)[1]
            _download(spec.url, download)
            if spec.archive:
                _extract(download, Path(temp))
            else:
                staging.mkdir()
                download.replace(staging / download.name)
            missing = [file for file in spec.files if not (staging / file).is_file()]
            if missing:
                raise RuntimeError(f'Model {name} download is missing {", ".join(missing)}')
            if folder.exists():
                shutil.rmtree(folder)
            staging.replace(folder)
        logger.info('Model %s ready in %s', name, folder)
        return folder
