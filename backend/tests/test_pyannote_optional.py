import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from app import config
from app.config import get_settings
from pipeline.engines import pyannote_engine


def test_installed_handles_missing_parent_packages():
    assert config.installed('json')
    assert not config.installed('surely_missing_package.submodule')


def test_missing_pyannote_is_reported_without_breaking_onnx_engines(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(config, 'installed', lambda name: name == 'sherpa_onnx')
    monkeypatch.setattr(settings, 'transcription_engine', 'parakeet_onnx')
    monkeypatch.setattr(settings, 'diarization_engine', 'sherpa_onnx')
    assert settings.engine_issue() is None
    monkeypatch.setattr(settings, 'diarization_engine', 'pyannote')
    assert 'requirements-pyannote.txt' in settings.engine_issue()


def test_engine_modules_import_without_torch():
    # Importing the pipeline must never pull in torch or pyannote.audio.
    code = ('import sys, pipeline.run, pipeline.engines.pyannote_engine, pipeline.engines.sherpa_diarization; '
            'print("torch" in sys.modules or "pyannote.audio" in sys.modules)')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.stdout.strip() == 'False'


def test_annotation_tracks_become_speaker_segments():
    class Annotation:
        def itertracks(self, yield_label=False):
            yield SimpleNamespace(start=0.0, end=1.5), 'a', 'SPEAKER_00'
            yield SimpleNamespace(start=1.5, end=4.0), 'b', 'SPEAKER_01'
    assert pyannote_engine.to_segments(Annotation()) == [
        {'start': 0.0, 'end': 1.5, 'speaker': 'SPEAKER_00'}, {'start': 1.5, 'end': 4.0, 'speaker': 'SPEAKER_01'}]
