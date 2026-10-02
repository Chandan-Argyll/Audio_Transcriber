"""Self-contained mock evaluation CLI check; no GPU or downloads."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from pipeline.engines.mock import MockEngine
from pipeline.merge import assign_speakers, build_transcript


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        audio = folder / 'sample.wav'
        with wave.open(str(audio), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(bytes(16000 * 2 * 20))
        engine = MockEngine()
        result = engine.align(engine.transcribe(audio, 20), audio)
        result = assign_speakers(engine.diarize(audio, 20, {}), result)
        reference = build_transcript('sample', 20, result)
        audio.with_suffix('.json').write_text(json.dumps(reference), encoding='utf-8')
        csv_path = folder / 'results.csv'
        environment = {**os.environ, 'ENGINE': 'mock', 'TRANSCRIPTION_ENGINE': 'mock', 'DIARIZATION_ENGINE': 'mock', 'MOCK_DELAY_SEC': '0'}
        subprocess.run([sys.executable, '-m', 'eval.evaluate', str(folder), '--csv', str(csv_path), '--save-predictions', str(folder / 'predictions')], env=environment, check=True)
        with csv_path.open() as output:
            rows = list(csv.DictReader(output))
        assert len(rows) == 2
        assert all(float(rows[0][metric]) == 0 for metric in ('WER', 'DER', 'cpWER')), rows
        subprocess.run([sys.executable, '-m', 'eval.evaluate', str(folder), '--predictions', str(folder / 'predictions'), '--csv', str(csv_path)], env=environment, check=True)
        print('Evaluation CLI smoke test passed (inference, CSV, saved predictions, rescoring).')


if __name__ == '__main__':
    main()
