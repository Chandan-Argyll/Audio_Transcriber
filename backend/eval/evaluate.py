"""WER, DER and cpWER for one or more engine combinations, side by side.

    # One recording with a CHAT reference, two combinations:
    python -m eval.evaluate --audio SBC0019.wav --reference SBC019.cha \\
        --engines parakeet_onnx+sherpa_onnx parakeet_onnx+pyannote
    # A dataset folder of audio + <stem>.json or <stem>.cha references:
    python -m eval.evaluate dataset/ --engines parakeet_onnx+sherpa_onnx --csv results.csv
    # Score saved predictions without inference:
    python -m eval.evaluate dataset/ --predictions predictions/

Combinations sharing a transcription engine reuse one transcription; each
row's processing time still includes it, so the speed factor (audio length /
processing time) reflects a full run of that combination.
"""
import argparse
import copy
import csv
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from app.config import get_settings
from app.schemas import Transcript
from eval.chat import load_chat
from pipeline.merge import assign_speakers, build_transcript
from pipeline.preprocess import normalize, probe
from pipeline.run import diarization_engine, transcription_engine

AUDIO = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4', '.webm'}
METRICS = ('WER', 'DER', 'cpWER')
COLUMNS = ['file', 'engines', 'status', *METRICS, 'audio_sec', 'processing_sec', 'speed_factor',
           'preprocess_sec', 'transcribe_sec', 'diarize_sec', 'merge_sec']


def normalize_text(text: str) -> str:
    import jiwer
    return jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])(text)


def cp_wer(reference: dict[str, str], hypothesis: dict[str, str]) -> float:
    """Concatenated minimum-permutation WER (meeteval's cpWER definition).

    Speakers are matched one-to-one to minimise total word errors; words of an
    unmatched reference speaker are deletions, of an unmatched hypothesis
    speaker insertions. Used when meeteval is not installed (it needs a C++ build).
    """
    try:
        from meeteval.wer.wer.cp import cp_word_error_rate
        return float(cp_word_error_rate(reference, hypothesis).error_rate)
    except ImportError:
        pass
    import numpy as np
    from rapidfuzz.distance import Levenshtein
    from scipy.optimize import linear_sum_assignment
    ref = [text.split() for text in reference.values()]
    hyp = [text.split() for text in hypothesis.values()]
    total = sum(len(words) for words in ref)
    size = max(len(ref), len(hyp))
    cost = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            r = ref[i] if i < len(ref) else []
            h = hyp[j] if j < len(hyp) else []
            cost[i, j] = Levenshtein.distance(r, h)
    rows, cols = linear_sum_assignment(cost)
    errors = cost[rows, cols].sum()
    return float(errors / total) if total else float(errors > 0)


def scores(reference: dict[str, Any], hypothesis: dict[str, Any]) -> dict[str, float]:
    import jiwer
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

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
    hyp_text = text(hypothesis)
    return {'WER': jiwer.wer(text(reference), hyp_text) if hyp_text else 1.0, 'DER': float(der),
            'cpWER': cp_wer(per_speaker(reference), per_speaker(hypothesis))}


def load_reference(path: Path, duration: float) -> dict[str, Any]:
    data = load_chat(path, duration) if path.suffix.lower() == '.cha' else json.loads(path.read_text(encoding='utf-8'))
    reference = Transcript.model_validate(data).model_dump()
    if not reference['turns']:
        raise SystemExit(f'{path.name}: reference has no turns; exclude silent references from this evaluation')
    return reference


def combination_issue(stt: str, diar: str) -> str | None:
    return get_settings().model_copy(update={'transcription_engine': stt, 'diarization_engine': diar}).engine_issue()


class Runner:
    """Runs engine combinations on one recording, sharing preprocessing and transcriptions."""
    def __init__(self, audio: Path, duration: float, folder: Path, options: dict[str, int]) -> None:
        self.audio, self.duration, self.options = audio, duration, options
        self.wav = folder / 'normalized.wav'
        began = time.perf_counter()
        normalize(audio, self.wav, duration)
        self.preprocess = time.perf_counter() - began
        self.transcriptions: dict[str, tuple[dict[str, Any], float]] = {}

    def run(self, stt: str, diar: str) -> tuple[dict[str, Any], dict[str, float]]:
        if stt not in self.transcriptions:
            engine = transcription_engine(stt)
            began = time.perf_counter()
            result = engine.align(engine.transcribe(self.wav, self.duration), self.wav)
            self.transcriptions[stt] = (result, time.perf_counter() - began)
        result, transcribe = self.transcriptions[stt]
        began = time.perf_counter()
        segments = diarization_engine(diar).diarize(self.wav, self.duration, self.options)
        diarize = time.perf_counter() - began
        began = time.perf_counter()
        transcript = build_transcript(self.audio.stem, self.duration, assign_speakers(segments, copy.deepcopy(result)))
        transcript.update(transcription_engine=stt, diarization_engine=diar)
        merge = time.perf_counter() - began
        total = self.preprocess + transcribe + diarize + merge
        return transcript, {'preprocess_sec': self.preprocess, 'transcribe_sec': transcribe, 'diarize_sec': diarize,
                            'merge_sec': merge, 'processing_sec': total, 'speed_factor': self.duration / total if total else 0}


def find_reference(audio: Path) -> Path | None:
    return next((p for p in (audio.with_suffix('.json'), audio.with_suffix('.cha')) if p.is_file()), None)


def print_row(row: dict[str, Any]) -> None:
    label = f'{row["file"][:24]:24} {row["engines"]:28}'
    if row['status'] != 'ok':
        print(f'{label} {row["status"]}')
        return
    timing = f'{row["processing_sec"]:9.1f} {row["speed_factor"]:7.2f}x' if 'processing_sec' in row else f'{"—":>9} {"—":>8}'
    print(f'{label} {row["WER"]:7.1%} {row["DER"]:7.1%} {row["cpWER"]:7.1%} {row["audio_sec"]:9.1f} {timing}')


def main() -> None:
    parser = argparse.ArgumentParser(description='WER, DER (0.25 s collar) and cpWER for engine combinations')
    parser.add_argument('folder', type=Path, nargs='?', help='Audio files with <stem>.json or <stem>.cha references')
    parser.add_argument('--audio', type=Path, help='Evaluate one recording instead of a folder')
    parser.add_argument('--reference', type=Path, help='Reference (.json or .cha) for --audio')
    parser.add_argument('--engines', nargs='+', metavar='STT+DIAR',
                        help='Combinations such as parakeet_onnx+sherpa_onnx parakeet_onnx+pyannote (default: configured engines)')
    parser.add_argument('--oracle-speakers', action='store_true', help='Pass the reference speaker count as num_speakers')
    parser.add_argument('--csv', type=Path, default=Path('results.csv'))
    parser.add_argument('--predictions', type=Path, help='Score saved JSON (flat, or one folder per combination) instead of running engines')
    parser.add_argument('--save-predictions', type=Path, help='Save outputs as <folder>/<stt+diar>/<stem>.json')
    args = parser.parse_args()

    if args.audio:
        if not args.reference:
            parser.error('--audio needs --reference')
        pairs = [(args.audio, args.reference)]
    elif args.folder:
        files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in AUDIO)
        pairs = [(audio, find_reference(audio)) for audio in files]
        if not pairs:
            parser.error('No supported audio files found')
        if missing := [audio.name for audio, ref in pairs if ref is None]:
            parser.error(f'No .json or .cha reference for: {", ".join(missing)}')
    else:
        parser.error('Give a dataset folder or --audio with --reference')

    combos = [tuple(c.split('+', 1)) for c in args.engines] if args.engines else [get_settings().active_engines()]
    if any(len(c) != 2 for c in combos):
        parser.error('--engines entries must look like parakeet_onnx+sherpa_onnx')

    rows: list[dict[str, Any]] = []
    print(f'{"File":24} {"Engines":28} {"WER":>7} {"DER":>7} {"cpWER":>7} {"Audio s":>9} {"Proc s":>9} {"Speed":>8}')
    for audio, reference_path in pairs:
        duration = probe(audio)
        reference = load_reference(reference_path, duration)
        options = {'num_speakers': len(reference['speakers'])} if args.oracle_speakers else {}
        with tempfile.TemporaryDirectory() as folder:
            runner = None
            for stt, diar in combos:
                label = f'{stt}+{diar}'
                row: dict[str, Any] = {'file': audio.name, 'engines': label, 'status': 'ok', 'audio_sec': duration}
                if args.predictions:
                    saved = args.predictions / label / f'{audio.stem}.json'
                    saved = saved if saved.is_file() else args.predictions / f'{audio.stem}.json'
                    prediction = json.loads(saved.read_text(encoding='utf-8'))
                elif issue := combination_issue(stt, diar):
                    row['status'] = f'skipped: {issue}'
                    rows.append(row)
                    print_row(row)
                    continue
                else:
                    runner = runner or Runner(audio, duration, Path(folder), options)
                    prediction, timings = runner.run(stt, diar)
                    row.update(timings)
                prediction = Transcript.model_validate(prediction).model_dump()
                if args.save_predictions:
                    target = args.save_predictions / label / f'{audio.stem}.json'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(prediction, indent=2), encoding='utf-8')
                row.update(scores(reference, prediction))
                rows.append(row)
                print_row(row)

    scored = [r for r in rows if r['status'] == 'ok']
    for label in dict.fromkeys(r['engines'] for r in scored):
        group = [r for r in scored if r['engines'] == label]
        average = {'file': 'MACRO_AVERAGE', 'engines': label, 'status': 'ok', 'audio_sec': sum(r['audio_sec'] for r in group),
                   **{m: mean(r[m] for r in group) for m in METRICS}}
        if all('processing_sec' in r for r in group):
            average['processing_sec'] = sum(r['processing_sec'] for r in group)
            average['speed_factor'] = average['audio_sec'] / average['processing_sec']
        rows.append(average)
        print_row(average)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
