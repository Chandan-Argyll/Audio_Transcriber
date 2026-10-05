"""Real sherpa-onnx inference. Skipped automatically unless the models are in MODELS_DIR
(they download on first app use) or when SKIP_MODEL_TESTS=1. Speech checks also need
the Santa Barbara corpus recording SBC0019.wav (SBC019_WAV or ~/Downloads)."""
import difflib
import os
import wave
from pathlib import Path

import pytest

pytest.importorskip('sherpa_onnx')
pytest.importorskip('soundfile')
from app.config import get_settings
from pipeline.model_store import MODELS, is_downloaded

pytestmark = [
    pytest.mark.skipif(not all(is_downloaded(name) for name in MODELS), reason='sherpa-onnx models not downloaded to MODELS_DIR'),
    # Real inference takes minutes on a laptop CPU; set SKIP_MODEL_TESTS=1 for quick runs.
    pytest.mark.skipif(os.environ.get('SKIP_MODEL_TESTS') == '1', reason='SKIP_MODEL_TESTS=1'),
]
CLIP_SEC = 30


def write_silence(path: Path, seconds: float) -> Path:
    with wave.open(str(path), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b'\0\0' * int(16000 * seconds))
    return path


def speech_source() -> Path | None:
    candidates = [os.environ.get('SBC019_WAV'), Path.home() / 'Downloads' / 'SBC0019.wav']
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


@pytest.fixture(scope='module')
def speech_clip(tmp_path_factory) -> Path:
    import soundfile
    from pipeline.preprocess import normalize
    source = speech_source()
    if source is None:
        pytest.skip('Set SBC019_WAV to the Santa Barbara corpus recording SBC0019.wav')
    folder = tmp_path_factory.mktemp('speech')
    normalize(source, folder / 'full.wav')
    samples, rate = soundfile.read(str(folder / 'full.wav'), dtype='int16')
    soundfile.write(str(folder / 'clip.wav'), samples[:rate * CLIP_SEC], rate, subtype='PCM_16')
    return folder / 'clip.wav'


@pytest.fixture(scope='module')
def parakeet():
    from pipeline.engines.parakeet_onnx import ParakeetOnnxEngine
    return ParakeetOnnxEngine()


def words_of(result: dict) -> list[dict]:
    return [word for segment in result['segments'] for word in segment['words']]


def test_silence_produces_no_speech_words_or_speakers(tmp_path, parakeet):
    from pipeline.audio import read_wav
    from pipeline.engines.parakeet_onnx import detect_speech
    from pipeline.engines.sherpa_diarization import SherpaOnnxDiarizationEngine
    audio = write_silence(tmp_path / 'silence.wav', 5)
    assert detect_speech(read_wav(audio)) == []
    assert parakeet.transcribe(audio, 5) == {'language': 'en', 'segments': []}
    assert SherpaOnnxDiarizationEngine().diarize(audio, 5, {}) == []


def test_parakeet_word_timestamps_on_speech(speech_clip, parakeet):
    words = words_of(parakeet.transcribe(speech_clip, CLIP_SEC))
    text = ' '.join(w['word'].lower() for w in words)
    assert 'graduate' in text and 'ballot' in text
    assert all(0 <= w['start'] <= w['end'] <= CLIP_SEC for w in words)
    assert all(a['end'] <= b['start'] + 1e-6 for a, b in zip(words, words[1:]))
    assert all(w['score'] is not None and 0 < w['score'] <= 1 for w in words)


def test_short_chunks_cut_in_pauses_keep_the_same_words(speech_clip, parakeet, monkeypatch):
    whole = [w['word'] for w in words_of(parakeet.transcribe(speech_clip, CLIP_SEC))]
    monkeypatch.setattr(get_settings(), 'max_chunk_sec', 10)
    chunked = words_of(parakeet.transcribe(speech_clip, CLIP_SEC))
    assert len({round(w['start']) // 10 for w in chunked}) >= 3
    # Cutting at pauses must not split, drop or duplicate words at chunk edges.
    assert difflib.SequenceMatcher(a=whole, b=[w['word'] for w in chunked]).ratio() > 0.9


def test_sherpa_diarization_respects_exact_speaker_count(speech_clip):
    from pipeline.engines.sherpa_diarization import SherpaOnnxDiarizationEngine
    engine = SherpaOnnxDiarizationEngine()
    segments = engine.diarize(speech_clip, CLIP_SEC, {'num_speakers': 2})
    assert len({s['speaker'] for s in segments}) == 2
    # The loaded pipeline is reused; a new exact count must still be honoured.
    assert len({s['speaker'] for s in engine.diarize(speech_clip, CLIP_SEC, {'num_speakers': 1})}) == 1
    assert all(0 <= s['start'] < s['end'] <= CLIP_SEC + 0.1 for s in segments)
    assert [s['start'] for s in segments] == sorted(s['start'] for s in segments)


def test_full_job_with_cpu_engines(client, speech_clip, monkeypatch):
    from app.routers import jobs
    from pipeline.run import engines, run_job
    settings = get_settings()
    monkeypatch.setattr(settings, 'transcription_engine', 'parakeet_onnx')
    monkeypatch.setattr(settings, 'diarization_engine', 'sherpa_onnx')
    # A 30 s clip gives most speakers under the 10 s default, which would merge them.
    monkeypatch.setattr(settings, 'min_speaker_seconds', 2)
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    engines.cache_clear()
    try:
        with speech_clip.open('rb') as audio:
            job_id = client.post('/api/jobs', files={'file': ('clip.wav', audio)}).json()['id']
        run_job(job_id)
    finally:
        engines.cache_clear()
    job = client.get(f'/api/jobs/{job_id}').json()
    assert job['stage'] == 'completed', job['error']
    assert set(job['stage_timings']) == {'preprocess', 'transcribe', 'diarize', 'merge', 'total'}
    transcript = client.get(f'/api/jobs/{job_id}/transcript').json()
    assert (transcript['transcription_engine'], transcript['diarization_engine']) == ('parakeet_onnx', 'sherpa_onnx')
    assert len(transcript['speakers']) >= 2 and transcript['turns']
