"""CPU speaker diarization with sherpa-onnx (no PyTorch).

sherpa-onnx clustering takes either an exact speaker count or a distance
threshold, not a min/max range. Speaker options are mapped as follows:
  * num_speakers=N, or min_speakers == max_speakers == N: exactly N clusters.
  * min and/or max only: cluster with DIARIZATION_THRESHOLD; if the number of
    speakers found falls outside the range, run once more with the count set
    to the nearest bound (e.g. 9 found with max 6 -> exactly 6).
"""
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings
from pipeline.audio import read_wav, to_float
from pipeline.model_store import ensure_model

logger = logging.getLogger('thread.diarization')


def exact_count(options: dict[str, int]) -> int | None:
    if 'num_speakers' in options:
        return options['num_speakers']
    if 'min_speakers' in options and options.get('min_speakers') == options.get('max_speakers'):
        return options['min_speakers']
    return None


def nearest_allowed(found: int, options: dict[str, int]) -> int | None:
    """The bound closest to an out-of-range speaker count, or None when it is in range."""
    low, high = options.get('min_speakers'), options.get('max_speakers')
    if low is not None and found < low:
        return low
    if high is not None and found > high:
        return high
    return None


def to_segments(result: Any) -> list[dict[str, Any]]:
    return [{'start': float(r.start), 'end': float(r.end), 'speaker': f'SPEAKER_{int(r.speaker):02d}'} for r in result]


class SherpaOnnxDiarizationEngine:
    """pyannote segmentation-3.0 (ONNX) + English WeSpeaker ResNet34 (VoxCeleb) embeddings."""
    def __init__(self) -> None:
        self.pipeline: Any = None

    def _config(self, num_clusters: int) -> Any:
        import sherpa_onnx
        settings = get_settings()
        segmentation = ensure_model('segmentation') / 'model.onnx'
        embedding = ensure_model('embedding') / 'wespeaker_en_voxceleb_resnet34_LM.onnx'
        return sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
                num_threads=settings.num_threads),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=settings.num_threads),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters, threshold=settings.diarization_threshold),
            min_duration_on=0.3, min_duration_off=0.5)

    def _run(self, samples: Any, num_clusters: int) -> list[dict[str, Any]]:
        import sherpa_onnx
        config = self._config(num_clusters)
        if self.pipeline is None:
            if not config.validate():
                raise RuntimeError('Invalid sherpa-onnx diarization configuration; check MODELS_DIR')
            self.pipeline = sherpa_onnx.OfflineSpeakerDiarization(config)
        else:
            self.pipeline.set_config(config)
        reported = [0]

        def progress(done: int, total: int) -> int:
            percent = done * 100 // max(total, 1)
            if percent >= reported[0] + 10:
                reported[0] = percent
                logger.info('Diarization %d%% (%d/%d segments)', percent, done, total)
            return 0

        return to_segments(self.pipeline.process(samples, callback=progress).sort_by_start_time())

    def diarize(self, audio: Path, duration: float, options: dict[str, int]) -> list[dict[str, Any]]:
        samples = to_float(read_wav(audio))
        exact = exact_count(options)
        segments = self._run(samples, exact or -1)
        if exact is None:
            found = len({s['speaker'] for s in segments})
            if (target := nearest_allowed(found, options)) is not None:
                logger.info('Threshold clustering found %d speakers outside %s; re-clustering to %d', found, options, target)
                segments = self._run(samples, target)
        return segments
