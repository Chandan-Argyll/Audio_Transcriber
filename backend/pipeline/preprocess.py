import json
import subprocess
from pathlib import Path
from app.config import get_settings

ALLOWED_FORMATS = {'mp3', 'wav', 'mov', 'mp4', 'm4a', '3gp', '3g2', 'mj2', 'flac', 'ogg', 'matroska', 'webm'}


def probe(path: Path) -> float:
    settings = get_settings()
    try:
        result = subprocess.run([settings.ffprobe_binary, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)], capture_output=True, text=True, timeout=60, check=True)
        data = json.loads(result.stdout)
        formats = set(data['format']['format_name'].split(','))
        if not formats.intersection(ALLOWED_FORMATS) or not any(s['codec_type'] == 'audio' for s in data['streams']):
            raise ValueError('Unsupported file or no audio stream')
        duration = float(data['format']['duration'])
        if not 0 < duration <= settings.max_duration_sec:
            raise ValueError(f'Audio duration must be between 0 and {settings.max_duration_sec:g} seconds')
        return duration
    except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid media file: ffprobe could not read audio') from exc


def normalize(source: Path, target: Path, duration: float = 0) -> None:
    # Decoding a multi-hour video can exceed a fixed ten-minute limit on slow disks.
    subprocess.run([get_settings().ffmpeg_binary, '-nostdin', '-v', 'error', '-y', '-i', str(source), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(target)], capture_output=True, timeout=max(600, duration), check=True)
