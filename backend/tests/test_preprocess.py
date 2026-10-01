import subprocess
from pathlib import Path
from app.config import get_settings
from pipeline.preprocess import normalize, probe


def test_fixture_normalizes_to_16khz_mono(tmp_path):
    source = Path(__file__).parent / 'fixtures' / 'tiny.wav'
    target = tmp_path / 'out.wav'
    assert probe(source) == 1
    normalize(source, target)
    import json
    result = subprocess.run([get_settings().ffprobe_binary, '-v', 'error', '-show_streams', '-of', 'json', str(target)], capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)['streams'][0]
    assert stream['channels'] == 1
    assert stream['sample_rate'] == '16000'
    assert stream['codec_name'] == 'pcm_s16le'
