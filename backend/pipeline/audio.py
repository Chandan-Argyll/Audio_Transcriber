from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def read_wav(audio: Path) -> np.ndarray:
    """16 kHz mono int16 samples (half the memory of float32 for multi-hour hearings)."""
    import soundfile
    samples, rate = soundfile.read(str(audio), dtype='int16', always_2d=False)
    if rate != SAMPLE_RATE or samples.ndim != 1:
        raise ValueError(f'Expected 16 kHz mono audio, got {rate} Hz with shape {samples.shape}')
    return samples


def to_float(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float32) / 32768
