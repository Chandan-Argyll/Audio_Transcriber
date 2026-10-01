from pipeline.merge import assign_mock_speakers, build_transcript


def test_overlap_and_unknown():
    result = {'segments': [{'words': [
        {'word': 'Hello', 'start': 0, 'end': 1},
        {'word': 'there', 'start': 1, 'end': 2},
        {'word': 'silence', 'start': 4, 'end': 5},
    ]}]}
    assign_mock_speakers([{'start': 0, 'end': .2, 'speaker': 'A'}, {'start': .2, 'end': 2, 'speaker': 'B'}], result)
    transcript = build_transcript('test', 5, result)
    assert [s['id'] for s in transcript['speakers']] == ['B', 'Unknown']
    assert transcript['speakers'][0]['label'] == 'Person 1'
    assert transcript['turns'][0]['text'] == 'Hello there'
    assert len(transcript['turns']) == 2


def test_first_appearance_and_missing_times():
    result = {'segments': [{'start': 0, 'words': [
        {'word': 'a', 'start': 0, 'end': 1, 'speaker': 'SPEAKER_07'},
        {'word': 'b', 'start': 1, 'end': 2, 'speaker': 'SPEAKER_00'},
        {'word': '!', 'speaker': 'SPEAKER_00'},
        {'word': 'c', 'start': 3, 'end': 4, 'speaker': 'SPEAKER_07'},
    ]}]}
    transcript = build_transcript('test', 5, result)
    assert [s['label'] for s in transcript['speakers']] == ['Person 1', 'Person 2']
    assert len(transcript['turns']) == 3
    assert transcript['turns'][1]['text'] == 'b !'


def test_unalignable_words_and_nonfinite_scores_are_preserved():
    result = {'segments': [{'start': 1, 'words': [
        {'word': '2026', 'start': None, 'end': None, 'score': float('nan')},
        {'word': 'works', 'start': 2, 'end': 3, 'speaker': 'A', 'score': .9},
    ]}]}
    assign_mock_speakers([{'start': 1, 'end': 3, 'speaker': 'A'}], result)
    transcript = build_transcript('test', 4, result)
    assert transcript['turns'][0]['text'] == '2026'
    assert transcript['turns'][0]['speaker_id'] == 'Unknown'
    assert transcript['turns'][0]['words'][0]['score'] is None
    assert transcript['turns'][0]['start'] == 1
