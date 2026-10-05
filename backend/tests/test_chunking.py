import pytest

np = pytest.importorskip('numpy')
from pipeline.chunking import FRAME_SEC, merge_speech, merge_tokens, offset_words, plan_chunks, quietest_point


def test_short_audio_is_a_single_chunk():
    assert plan_chunks([(1, 20)], 30, 60) == [(0.0, 30)]


def test_cuts_land_in_pauses_and_never_exceed_the_limit():
    # Speech every 10 s with 1 s pauses: 0-9, 10-19, ... 140-149.
    speech = [(i * 10, i * 10 + 9) for i in range(15)]
    chunks = plan_chunks(speech, 150, 60)
    assert chunks[0][0] == 0 and chunks[-1][1] == 150
    assert all(b - a <= 60 for a, b in chunks)
    assert all(a_end == b_start for (_, a_end), (b_start, _) in zip(chunks, chunks[1:]))
    for _, cut in chunks[:-1]:
        # Every cut is in the middle of a pause, never inside a speech region.
        assert not any(start < cut < end for start, end in speech)
        assert cut % 10 == pytest.approx(9.5)


def test_forced_vad_splits_are_not_used_as_cut_points():
    # A 0.05 s "gap" is a forced split inside continuous speech, not a pause.
    assert merge_speech([(0, 50), (50.05, 100), (101, 110)]) == [(0, 100), (101, 110)]
    cuts = []
    chunks = plan_chunks([(0, 50), (50.05, 100), (101, 110)], 110, 60,
                         fallback_cut=lambda a, b: cuts.append((a, b)) or 42.0)
    assert cuts == [(0.0, 60.0)]
    assert chunks[0] == (0.0, 42.0)
    assert chunks[1][1] == pytest.approx(100.5)


def test_fallback_without_pause_still_makes_progress():
    chunks = plan_chunks([(0, 200)], 200, 60, fallback_cut=lambda a, b: a)
    assert all(30 <= b - a <= 60 for a, b in chunks[:-1])
    assert chunks[-1][1] == 200


def test_silent_chunks_are_dropped():
    # Without a fallback the window end is used; chunks that cover only silence are skipped.
    assert plan_chunks([(0, 10), (130, 140)], 150, 60) == [(0.0, 60.0), (130.0, 150)]
    assert plan_chunks([(0, 10), (200, 210)], 250, 60) == [(0.0, 60.0), (165.0, 225.0)]


def test_quietest_point_finds_the_silence():
    rate = 1000
    samples = np.full(10 * rate, 10000, dtype=np.int16)
    samples[7 * rate:7 * rate + 50] = 0
    assert quietest_point(samples, rate, 0, 10) == pytest.approx(7.01, abs=0.02)


def test_subword_tokens_merge_into_words_with_punctuation():
    tokens = [' Well', ',', ' not', ' gr', 'ad', 'u', 'ate', '.', ' they', "'", 're']
    starts = [0.56, 0.72, 1.12, 1.6, 1.76, 1.84, 1.92, 1.92, 2.64, 2.72, 2.8]
    durations = [0.16, 0.08, 0.08, 0.16, 0.08, 0.08, 0.0, 0.08, 0.08, 0.0, 0.16]
    words = merge_tokens(tokens, starts, durations, log_probs=[0.0] * len(tokens))
    assert [w['word'] for w in words] == ['Well,', 'not', 'graduate.', "they're"]
    graduate = words[2]
    assert graduate['start'] == 1.6
    # Last sub-word has zero duration: the word still lasts at least one frame.
    assert graduate['end'] == pytest.approx(1.92 + FRAME_SEC)
    # Punctuation does not stretch a word.
    assert words[0]['end'] == pytest.approx(0.72)
    assert all(w['score'] == 1.0 for w in words)


def test_word_ends_never_overlap_the_next_word_or_chunk_end():
    words = merge_tokens(['▁a', '▁b'], [0.0, 0.1], [0.5, 0.5], end_limit=0.3)
    assert words[0]['end'] == 0.1
    assert words[1]['end'] == 0.3
    assert merge_tokens([' ', ''], [0, 0]) == []


def test_offset_words_moves_chunk_times_onto_the_recording_timeline():
    words = merge_tokens([' hello', ' world'], [0.5, 1.0], [0.4, 0.4])
    shifted = offset_words(words, 120.0)
    assert [(w['start'], w['end']) for w in shifted] == [(120.5, 120.9), (121.0, 121.4)]
    assert words[0]['start'] == 0.5  # Original left unchanged.


def glued(tokens: list[str]) -> list[str]:
    return [w['word'] for w in merge_tokens(tokens, [i * 0.08 for i in range(len(tokens))])]


@pytest.mark.parametrize('tokens,expected', [
    # Exact token sequences Parakeet emits: a standalone ' ' marks the next token as a word start.
    ([' in', ' ', '2', '0', '0', '6'], ['in', '2006']),
    ([' c', 'ase', ' ', '2', '4', '7', '3', '5', '1'], ['case', '247351']),
    ([' pl', 'ain', ' ', 'er', 'ror'], ['plain', 'error']),
    ([' the', ' ', 'er', 'ror'], ['the', 'error']),
    ([' argument', ' ', 'es', 'p', 'ous', 'ed', ' ', '1', '4'], ['argument', 'espoused', '14']),
    (['▁in', '▁', '2', '0', '0', '6', ','], ['in', '2006,']),
])
def test_word_start_markers_are_always_respected(tokens, expected):
    assert glued(tokens) == expected
    assert ''.join(glued(tokens)) == ''.join(t.strip(' ▁') for t in tokens)


def test_pending_word_start_keeps_the_marker_tokens_timing_out_of_the_word():
    words = merge_tokens([' in', ' ', '2', '0'], [0.0, 0.2, 0.24, 0.32], [0.16, 0.0, 0.08, 0.08])
    assert [(w['word'], w['start']) for w in words] == [('in', 0.0), ('20', 0.24)]
    assert words[0]['end'] == pytest.approx(0.16)
