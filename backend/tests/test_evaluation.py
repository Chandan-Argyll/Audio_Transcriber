"""Optional metric tests: install requirements-eval.txt to enable these."""
import copy
import pytest

pytest.importorskip('jiwer')
pytest.importorskip('pyannote.metrics')
pytest.importorskip('meeteval')
from eval.evaluate import scores


def sample() -> dict:
    return {'duration_sec': 10, 'turns': [
        {'speaker_id': 'A', 'start': 0, 'end': 4, 'text': 'Hello everyone.'},
        {'speaker_id': 'B', 'start': 5, 'end': 9, 'text': 'Thanks for joining.'},
    ]}


def test_identical_and_permuted_speaker_ids():
    reference = sample()
    hypothesis = copy.deepcopy(reference)
    hypothesis['turns'][0]['speaker_id'] = 'different-id-1'
    hypothesis['turns'][1]['speaker_id'] = 'different-id-2'
    assert scores(reference, hypothesis) == {'WER': 0, 'DER': 0, 'cpWER': 0}


def test_errors_are_measured():
    reference = sample()
    hypothesis = copy.deepcopy(reference)
    hypothesis['turns'][1]['speaker_id'] = 'A'
    hypothesis['turns'][1]['text'] = 'Thank you.'
    values = scores(reference, hypothesis)
    assert values['WER'] > 0
    assert values['DER'] > 0
    assert values['cpWER'] > 0
