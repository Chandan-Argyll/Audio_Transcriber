import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import get_settings, speaker_options, vocabularies_for
from app.routers import jobs
from pipeline.merge import absorb_short_turns, snap_speaker_changes
from pipeline.speakers import merge_small_speakers


def words_from(spec: list[tuple[str, float, float, str]]) -> list[dict]:
    return [{'word': w, 'start': s, 'end': e, 'speaker': spk} for w, s, e, spk in spec]


def labels(words: list[dict]) -> list[str]:
    return [w['speaker'] for w in words]


# ---- 2. Recording types and speaker counts -------------------------------------

def test_recording_type_is_validated_stored_and_kept_out_of_diarization(client, wav_bytes, monkeypatch):
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    data = {'recording_type': 'supreme_court', 'min_speakers': '10', 'max_speakers': '14'}
    job_id = client.post('/api/jobs', files={'file': ('argument.wav', wav_bytes)}, data=data).json()['id']
    from app.models import Job, SessionLocal
    with SessionLocal() as session:
        options = session.get(Job, job_id).options
    assert options == {'recording_type': 'supreme_court', 'min_speakers': 10, 'max_speakers': 14}
    assert speaker_options(options) == {'min_speakers': 10, 'max_speakers': 14}
    assert vocabularies_for(options) == ['legal']
    assert client.post('/api/jobs', files={'file': ('a.wav', wav_bytes)}, data={'recording_type': 'podcast'}).status_code == 400
    presets = client.get('/api/health').json()['recording_presets']
    assert presets['supreme_court'] == {'label': 'Supreme Court argument', 'min_speakers': 10, 'max_speakers': 14}


def test_worker_passes_only_speaker_counts_to_diarization(client, wav_bytes, monkeypatch):
    from pipeline.engines.mock import MockEngine
    from pipeline.run import run_job
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    seen = []
    original = MockEngine.diarize
    monkeypatch.setattr(MockEngine, 'diarize', lambda self, audio, duration, options: seen.append(options) or original(self, audio, duration, options))
    data = {'recording_type': 'supreme_court', 'num_speakers': '12'}
    run_job(client.post('/api/jobs', files={'file': ('argument.wav', wav_bytes)}, data=data).json()['id'])
    assert seen == [{'num_speakers': 12}]


class FakeSherpa:
    """Records the clustering configuration sherpa-onnx receives."""
    def __init__(self, found_speakers: int = 3):
        self.clusterings: list[dict] = []
        self.found = found_speakers
        outer = self

        class Diarization:
            def __init__(self, config):
                outer.clusterings.append(config.clustering)

            def set_config(self, config):
                outer.clusterings.append(config.clustering)

            def process(self, samples, callback=None):
                wanted = outer.clusterings[-1]['num_clusters']
                count = wanted if wanted > 0 else outer.found
                return SimpleNamespace(sort_by_start_time=lambda: [SimpleNamespace(start=i, end=i + 1, speaker=i) for i in range(count)])

        def record(**kwargs):
            return SimpleNamespace(**kwargs, validate=lambda: True)
        self.module = SimpleNamespace(
            OfflineSpeakerDiarization=Diarization, OfflineSpeakerDiarizationConfig=record,
            OfflineSpeakerSegmentationModelConfig=record, OfflineSpeakerSegmentationPyannoteModelConfig=record,
            SpeakerEmbeddingExtractorConfig=record, FastClusteringConfig=lambda **kw: kw)


@pytest.fixture
def fake_sherpa(monkeypatch, tmp_path):
    from pipeline.engines import sherpa_diarization
    fake = FakeSherpa()
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', fake.module)
    monkeypatch.setattr(sherpa_diarization, 'ensure_model', lambda name: tmp_path)
    monkeypatch.setattr(sherpa_diarization, 'read_wav', lambda audio: np.zeros(16000, dtype=np.int16))
    return fake


def test_exact_speaker_count_reaches_sherpa_clustering(fake_sherpa):
    from pipeline.engines.sherpa_diarization import SherpaOnnxDiarizationEngine
    engine = SherpaOnnxDiarizationEngine()
    assert len({s['speaker'] for s in engine.diarize(None, 1, {'num_speakers': 12})}) == 12
    assert fake_sherpa.clusterings[-1]['num_clusters'] == 12
    # Later jobs reuse the loaded pipeline: the new count must arrive through set_config.
    engine.diarize(None, 1, {'num_speakers': 4})
    assert [c['num_clusters'] for c in fake_sherpa.clusterings] == [12, 4]
    engine.diarize(None, 1, {'min_speakers': 7, 'max_speakers': 7})
    assert fake_sherpa.clusterings[-1]['num_clusters'] == 7


def test_speaker_range_reclusters_to_the_nearest_bound(fake_sherpa):
    from pipeline.engines.sherpa_diarization import SherpaOnnxDiarizationEngine
    fake_sherpa.found = 3  # Threshold clustering finds 3 voices.
    segments = SherpaOnnxDiarizationEngine().diarize(None, 1, {'min_speakers': 10, 'max_speakers': 14})
    assert [c['num_clusters'] for c in fake_sherpa.clusterings] == [-1, 10]
    assert len({s['speaker'] for s in segments}) == 10


# ---- 3. Small clusters and short turns ------------------------------------------

def test_small_speakers_merge_into_the_most_similar_large_speaker():
    segments = [{'start': 0, 'end': 30, 'speaker': 'A'}, {'start': 30, 'end': 60, 'speaker': 'B'},
                {'start': 60, 'end': 64, 'speaker': 'C'}, {'start': 64, 'end': 66, 'speaker': 'D'}]
    vectors = {'A': np.array([1.0, 0.0]), 'B': np.array([0.0, 1.0]), 'C': np.array([0.1, 0.9]), 'D': np.array([0.95, 0.05])}
    merged = merge_small_speakers(segments, 10, lambda segs: vectors[segs[0]['speaker']])
    assert [s['speaker'] for s in merged] == ['A', 'B', 'B', 'A']
    assert segments[2]['speaker'] == 'C'  # Input left untouched.


def test_small_speaker_without_embedding_joins_the_nearest_speaker_in_time():
    segments = [{'start': 0, 'end': 20, 'speaker': 'A'}, {'start': 100, 'end': 120, 'speaker': 'B'},
                {'start': 121, 'end': 122, 'speaker': 'C'}]
    vectors = {'A': np.array([1.0, 0.0]), 'B': np.array([0.0, 1.0]), 'C': None}
    assert merge_small_speakers(segments, 10, lambda segs: vectors[segs[0]['speaker']])[2]['speaker'] == 'B'


def test_small_speaker_cleanup_is_skipped_when_disabled_or_nothing_is_large():
    segments = [{'start': 0, 'end': 5, 'speaker': 'A'}, {'start': 5, 'end': 8, 'speaker': 'B'}]
    explode = lambda segs: pytest.fail('no embeddings needed')
    assert merge_small_speakers(segments, 10, explode) is segments
    assert merge_small_speakers(segments, 0, explode) is segments


def test_one_and_two_word_turns_join_the_neighbour_overlapping_them_most():
    words = words_from([('So', 0.0, 0.3, 'A'), ('the', 0.3, 0.5, 'A'), ('record', 0.5, 1.0, 'A'),
                        ('shows', 1.0, 1.4, 'B'),
                        ('that', 1.4, 1.7, 'C'), ('the', 1.7, 1.9, 'C'), ('judge', 1.9, 2.3, 'C')])
    diarization = [{'start': 0, 'end': 1.2, 'speaker': 'A'}, {'start': 1.2, 'end': 2.3, 'speaker': 'C'}]
    # 'shows' (1.0-1.4) overlaps A for 0.2 s and C for 0.2 s: tie -> the closer neighbour (both touch) -> A wins
    # on equal gap by order; make C clearly larger instead:
    diarization[1]['start'] = 1.05
    assert labels(absorb_short_turns(words, diarization)) == ['A', 'A', 'A', 'C', 'C', 'C', 'C']


def test_short_unknown_turn_with_no_overlap_joins_the_closer_neighbour():
    words = words_from([('We', 0.0, 0.2, 'Unknown'), ('will', 0.25, 0.5, 'A'), ('hear', 0.5, 0.8, 'A'), ('argument', 0.8, 1.3, 'A'),
                        ('now', 1.3, 1.6, 'A'), ('Counsel', 5.0, 5.6, 'B'), ('please', 5.6, 6.0, 'B'), ('begin', 6.0, 6.4, 'B')])
    diarization = [{'start': 0.25, 'end': 1.6, 'speaker': 'A'}, {'start': 5.0, 'end': 6.4, 'speaker': 'B'}]
    assert labels(absorb_short_turns(words, diarization)) == ['A'] * 5 + ['B'] * 3


# ---- 4. Snapping speaker changes ----------------------------------------------

def test_change_inside_a_sentence_moves_to_the_sentence_end():
    words = words_from([('I', 0.0, 0.2, 'A'), ('agree.', 0.2, 0.6, 'A'), ('But', 0.6, 0.8, 'A'),
                        ('the', 0.8, 1.0, 'B'), ('record', 1.0, 1.4, 'B'), ('is', 1.4, 1.6, 'B'), ('silent.', 1.6, 2.0, 'B')])
    assert labels(snap_speaker_changes(words)) == ['A', 'A', 'B', 'B', 'B', 'B', 'B']


def test_change_moves_forward_to_a_pause_within_one_and_a_half_seconds():
    words = words_from([('the', 0.0, 0.2, 'A'), ('trial', 0.2, 0.5, 'A'), ('court', 0.5, 0.8, 'B'),
                        ('found', 0.8, 1.1, 'B'), ('Justice', 1.5, 1.9, 'B'), ('Kagan', 1.9, 2.3, 'B'), ('asked', 2.3, 2.7, 'B')])
    # 0.4 s pause before 'Justice' (1.1 -> 1.5) is 0.65 s from the change: move there.
    assert labels(snap_speaker_changes(words)) == ['A', 'A', 'A', 'A', 'B', 'B', 'B']


def test_distant_boundaries_are_ignored_and_a_real_gap_is_kept():
    words = words_from([('one', 0.0, 0.3, 'A'), ('two', 0.3, 0.6, 'A'), ('three', 0.65, 0.9, 'B'),
                        ('four', 0.9, 1.2, 'B'), ('five', 1.2, 1.5, 'B'), ('six.', 1.5, 3.5, 'B'), ('seven', 4.0, 4.3, 'B')])
    # The only strong boundary (after 'six.') is more than 1.5 s away; the 50 ms gap is real, so keep it.
    assert labels(snap_speaker_changes(words)) == ['A', 'A', 'B', 'B', 'B', 'B', 'B']


def test_a_turn_is_never_split_between_touching_words():
    words = words_from([('in', 0.0, 0.2, 'A'), ('the', 0.2, 0.3, 'A'), ('selection', 0.3, 0.9, 'B'), ('of', 0.9, 1.0, 'B'),
                        ('Terry', 1.0, 1.4, 'B'), ('Pitchford', 1.4, 2.0, 'B'), ('jury', 2.05, 2.5, 'B'), ('pool', 2.5, 2.9, 'B')])
    snapped = snap_speaker_changes(words)
    changes = [k for k in range(1, len(snapped)) if snapped[k]['speaker'] != snapped[k - 1]['speaker']]
    assert all(snapped[k]['start'] - snapped[k - 1]['end'] > 0 or snapped[k - 1]['word'].endswith('.') for k in changes)
    assert labels(snapped) == ['A'] * 6 + ['B'] * 2  # Nearest gap (50 ms before 'jury').


def test_turn_with_no_gap_anywhere_joins_its_smaller_side():
    words = words_from([('a', 0.0, 0.2, 'A'), ('b', 0.2, 0.4, 'B'), ('c', 0.4, 0.6, 'B'), ('d', 0.6, 0.8, 'B')])
    assert labels(snap_speaker_changes(words)) == ['B'] * 4


# ---- 5. Domain vocabulary -----------------------------------------------------

def correct(text: str, *names: str) -> str:
    from pipeline.vocabulary import corrector_for
    words = [{'word': w, 'start': i * 1.0, 'end': i * 1.0 + 0.9, 'score': 0.9} for i, w in enumerate(text.split())]
    return ' '.join(w['word'] for w in corrector_for(list(names) or ['legal']).correct_words(words))


def test_legal_replacements_are_whole_word_and_case_aware():
    assert correct('the veneer was struck') == 'the venire was struck'
    assert correct('Veneer members, VENEER.') == 'Venire members, VENIRE.'
    assert correct('a veneered table') == 'a veneered table'
    assert correct('during void dire, the') == 'during voir dire, the'
    assert correct('the voidier that morning') == 'the voir dire that morning'
    assert correct('Voidir was short.') == 'Voir dire was short.'


def test_aedpa_only_near_its_context_words():
    assert correct('deference under EDPA applies') == 'deference under AEDPA applies'
    assert correct('the EPA standard of review') == 'the AEDPA standard of review'
    assert correct('habeas review under EPA') == 'habeas review under AEDPA'  # 3 words away
    assert correct('habeas review under the EPA') == 'habeas review under the EPA'  # 4 words away
    assert correct('the EPA regulates emissions') == 'the EPA regulates emissions'
    assert correct('habeas one two three four EPA') == 'habeas one two three four EPA'


def test_replacement_splits_timing_and_keeps_scores():
    from pipeline.vocabulary import corrector_for
    words = [{'word': 'the', 'start': 0.0, 'end': 0.2, 'score': 0.9}, {'word': 'voidier,', 'start': 0.2, 'end': 0.8, 'score': 0.6}]
    fixed = corrector_for(['legal']).correct_words(words)
    assert [(w['word'], w['start'], w['end'], w['score']) for w in fixed] == [
        ('the', 0.0, 0.2, 0.9), ('voir', 0.2, 0.5, 0.6), ('dire,', 0.5, 0.8, 0.6)]


def test_vocabularies_come_from_recording_type_and_settings(monkeypatch, tmp_path):
    from pipeline.vocabulary import corrector_for
    assert vocabularies_for({}) == [] and vocabularies_for({'recording_type': 'meeting'}) == []
    custom = tmp_path / 'company.json'
    custom.write_text('{"replacements": [{"from": ["argil"], "to": "Argyll"}]}', encoding='utf-8')
    monkeypatch.setattr(get_settings(), 'vocabularies', str(custom))
    assert vocabularies_for({'recording_type': 'supreme_court'}) == [str(custom), 'legal']
    assert correct('argil infotech', str(custom)) == 'Argyll infotech'
    assert corrector_for(['legal']).hotwords == ['Batson', 'venire', 'voir dire', 'AEDPA', 'peremptory']
    with pytest.raises(ValueError, match='not found'):
        corrector_for(['no-such-dictionary'])


def test_hotwords_are_spelled_with_model_tokens():
    from pipeline.engines.parakeet_onnx import encode_hotwords
    vocabulary = {'▁B', 'at', 'so', 'n', '▁v', 'o', 'ir', '▁d', 'ire', '▁Bat'}
    assert encode_hotwords(['Batson', 'voir dire', 'Zebra'], vocabulary) == '▁Bat so n/▁v o ir ▁d ire'


def test_vocabulary_corrections_run_inside_jobs(client, wav_bytes, monkeypatch):
    from pipeline.engines import mock
    from pipeline.run import run_job
    monkeypatch.setattr(jobs, 'enqueue', lambda _: None)
    monkeypatch.setattr(mock, 'LINES', ['The veneer was chosen after voidier on federal habeas EPA review.'])
    general = client.post('/api/jobs', files={'file': ('a.wav', wav_bytes)}).json()['id']
    court = client.post('/api/jobs', files={'file': ('a.wav', wav_bytes)}, data={'recording_type': 'supreme_court'}).json()['id']
    run_job(general)
    run_job(court)
    general_text = client.get(f'/api/jobs/{general}/transcript').json()['turns'][0]['text']
    court_text = ' '.join(t['text'] for t in client.get(f'/api/jobs/{court}/transcript').json()['turns'])
    assert 'veneer' in general_text
    assert 'The venire was chosen after voir dire on federal habeas AEDPA review.' in court_text
