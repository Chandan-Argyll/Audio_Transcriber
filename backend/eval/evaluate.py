"""python -m eval.evaluate dataset/ --csv results.csv [--predictions predictions/]"""
import argparse
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
from app.config import get_settings
from app.schemas import Transcript
from pipeline.merge import assign_mock_speakers, build_transcript
from pipeline.preprocess import normalize, probe
from pipeline.run import engines


def normalize_text(text: str) -> str:
    import jiwer
    return jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])(text)


def scores(reference: dict[str, Any], hypothesis: dict[str, Any]) -> dict[str, float]:
    import jiwer
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    from meeteval.wer.wer.cp import cp_word_error_rate

    def text(data: dict[str, Any]) -> str:
        return normalize_text(' '.join(t['text'] for t in data['turns']))

    def per_speaker(data: dict[str, Any]) -> dict[str, str]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for turn in data['turns']:
            grouped[turn['speaker_id']].append(turn['text'])
        return {key: normalize_text(' '.join(value)) for key, value in grouped.items()}

    def annotation(data: dict[str, Any]) -> Annotation:
        output = Annotation()
        for i, turn in enumerate(data['turns']):
            if turn['end'] > turn['start']:
                output[Segment(turn['start'], turn['end']), i] = turn['speaker_id']
        return output

    uem = Timeline([Segment(0, reference['duration_sec'])])
    der = DiarizationErrorRate(collar=.25, skip_overlap=False)(annotation(reference), annotation(hypothesis), uem=uem)
    cp = cp_word_error_rate(per_speaker(reference), per_speaker(hypothesis))
    return {'WER': jiwer.wer(text(reference), text(hypothesis)), 'DER': float(der), 'cpWER': float(cp.error_rate)}


def predict(audio: Path) -> dict[str, Any]:
    duration = probe(audio)
    transcription, diarization = engines()
    with tempfile.TemporaryDirectory() as folder:
        wav = Path(folder) / 'normalized.wav'
        normalize(audio, wav)
        result = transcription.align(transcription.transcribe(wav, duration), wav)
        intervals = diarization.diarize(wav, duration, {})
        if isinstance(intervals, list):
            result = assign_mock_speakers(intervals, result)
        else:
            import whisperx
            result = whisperx.assign_word_speakers(intervals, result)
        return build_transcript(audio.stem, duration, result)


def main() -> None:
    parser = argparse.ArgumentParser(description='WER, DER (0.25s collar), cpWER evaluation')
    parser.add_argument('folder', type=Path)
    parser.add_argument('--csv', type=Path, default=Path('results.csv'))
    parser.add_argument('--predictions', type=Path, help='Score saved JSON instead of running engines')
    parser.add_argument('--save-predictions', type=Path, help='Save pipeline outputs for later comparison')
    args = parser.parse_args()
    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4', '.webm'})
    if not files:
        parser.error('No supported audio files found')
    if args.save_predictions:
        args.save_predictions.mkdir(parents=True, exist_ok=True)
    rows = []
    settings = get_settings()
    combination = f'{settings.transcription_engine or settings.engine}/{settings.diarization_engine or ("pyannote" if settings.engine == "whisperx" else settings.engine)}'
    print(f'Engine combination: {combination}')
    print(f'{"File":32} {"WER":>9} {"DER":>9} {"cpWER":>9}')
    for audio in files:
        reference = Transcript.model_validate_json(audio.with_suffix('.json').read_text(encoding='utf-8')).model_dump()
        if not reference['turns']:
            parser.error(f'{audio.name}: reference has no turns; exclude silent references from this evaluation')
        prediction = json.loads((args.predictions / f'{audio.stem}.json').read_text(encoding='utf-8')) if args.predictions else predict(audio)
        prediction = Transcript.model_validate(prediction).model_dump()
        if args.save_predictions:
            (args.save_predictions / f'{audio.stem}.json').write_text(json.dumps(prediction, indent=2), encoding='utf-8')
        values = scores(reference, prediction)
        rows.append({'file': audio.name, 'engines': combination, **values})
        print(f'{audio.name[:32]:32} {values["WER"]:9.3%} {values["DER"]:9.3%} {values["cpWER"]:9.3%}')
    average = {key: mean(row[key] for row in rows) for key in ('WER', 'DER', 'cpWER')}
    print(f'{"Macro average":32} {average["WER"]:9.3%} {average["DER"]:9.3%} {average["cpWER"]:9.3%}')
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=['file', 'engines', 'WER', 'DER', 'cpWER'])
        writer.writeheader()
        writer.writerows(rows + [{'file': 'MACRO_AVERAGE', 'engines': combination, **average}])


if __name__ == '__main__':
    main()
