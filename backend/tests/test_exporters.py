import json
from pipeline.engines.mock import mock_segments
from pipeline.merge import assign_mock_speakers, build_transcript
from pipeline.exporters import export, timestamp


def test_exports_and_limits():
    result = {'segments': mock_segments(800)}
    assign_mock_speakers([{'start': 0, 'end': 800, 'speaker': 'A'}], result)
    transcript = build_transcript('test', 800, result)
    transcript['speakers'][0]['label'] = 'Alice'
    assert '[00:00:00] Alice:' in export(transcript, 'txt')
    assert json.loads(export(transcript, 'json')) == transcript
    srt = export(transcript, 'srt')
    for cue in srt.strip().split('\n\n'):
        lines = cue.splitlines()
        assert len(lines[2]) <= 84
        def seconds(value):
            h, m, s = value.replace(',', '.').split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
        start, end = lines[1].split(' --> ')
        assert seconds(end) - seconds(start) <= 7.001
    assert timestamp(3661.25, True) == '01:01:01,250'
