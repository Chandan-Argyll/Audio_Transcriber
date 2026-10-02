from types import SimpleNamespace

from pipeline.engines.sherpa_diarization import exact_count, nearest_allowed, to_segments


def test_exact_counts():
    assert exact_count({'num_speakers': 4}) == 4
    assert exact_count({'min_speakers': 3, 'max_speakers': 3}) == 3
    assert exact_count({'min_speakers': 2, 'max_speakers': 5}) is None
    assert exact_count({}) is None


def test_out_of_range_counts_snap_to_the_nearest_bound():
    assert nearest_allowed(9, {'max_speakers': 6}) == 6
    assert nearest_allowed(1, {'min_speakers': 2}) == 2
    assert nearest_allowed(1, {'min_speakers': 2, 'max_speakers': 5}) == 2
    assert nearest_allowed(4, {'min_speakers': 2, 'max_speakers': 5}) is None
    assert nearest_allowed(12, {}) is None


def test_segments_use_stable_string_speaker_ids():
    raw = [SimpleNamespace(start=0.5, end=2.0, speaker=0), SimpleNamespace(start=2.0, end=3.5, speaker=11)]
    assert to_segments(raw) == [{'start': 0.5, 'end': 2.0, 'speaker': 'SPEAKER_00'},
                                {'start': 2.0, 'end': 3.5, 'speaker': 'SPEAKER_11'}]
