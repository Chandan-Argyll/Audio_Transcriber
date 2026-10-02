"""Optional metric tests: install requirements-eval.txt to enable these."""
import copy
import pytest

pytest.importorskip('jiwer')
pytest.importorskip('pyannote.metrics')
from eval import evaluate
from eval.evaluate import cp_wer, scores


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


def test_builtin_cpwer_matches_the_meeteval_definition(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_meeteval(name, *args, **kwargs):
        if name.startswith('meeteval'):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_meeteval)
    reference = {'A': 'a b c d', 'B': 'e f'}
    # Best mapping x->B, y->A: 1 substitution (c->z) out of 6 reference words.
    assert cp_wer(reference, {'x': 'e f', 'y': 'a b z d'}) == pytest.approx(1 / 6)
    # Extra hypothesis speaker: all of its words are insertions.
    assert cp_wer(reference, {'x': 'a b c d', 'y': 'e f', 'z': 'g h'}) == pytest.approx(2 / 6)
    # Everything said by one speaker: B's words are deletions plus a mapping cost.
    assert cp_wer(reference, {'x': 'a b c d e f'}) == pytest.approx(4 / 6)
    assert cp_wer({'A': 'a'}, {}) == 1.0


def test_builtin_cpwer_agrees_with_meeteval_when_installed():
    meeteval = pytest.importorskip('meeteval')
    reference = {'A': 'the quick brown fox', 'B': 'jumps over the dog'}
    hypothesis = {'1': 'jumps over a dog', '2': 'the quick fox', '3': 'yes'}
    expected = float(meeteval.wer.wer.cp.cp_word_error_rate(reference, hypothesis).error_rate)
    assert cp_wer(reference, hypothesis) == pytest.approx(expected)
