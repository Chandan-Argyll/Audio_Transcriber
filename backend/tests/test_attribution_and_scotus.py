import pytest

pytest.importorskip('jiwer')
pytest.importorskip('scipy')
from eval.attribution import mid_sentence_turns, speaker_attribution
from eval.scotus import parse_transcript


def transcript(*turns: tuple[str, str]) -> dict:
    return {'duration_sec': 0, 'turns': [{'speaker_id': s, 'start': 0, 'end': 0, 'text': t} for s, t in turns]}


def test_perfect_attribution_with_different_labels():
    reference = transcript(('KAGAN', 'What is the standard?'), ('STEWART', 'It is deferential, Justice Kagan.'))
    hypothesis = transcript(('Person 1', 'What is the standard?'), ('Person 2', 'It is deferential, Justice Kagan.'))
    result = speaker_attribution(reference, hypothesis)
    assert result['speaker_accuracy'] == 1.0
    assert result['mapping'] == {'Person 1': 'KAGAN', 'Person 2': 'STEWART'}
    assert result['purity']['Person 2'] == {'words': 5, 'dominant': 'STEWART', 'purity': 1.0, 'mapped_to': 'STEWART'}


def test_misattributed_and_extra_speakers_are_counted_wrong():
    reference = transcript(('A', 'one two three four five six'), ('B', 'seven eight nine ten'))
    # Person 1 has A's words plus two of B's; Person 3 is an extra speaker holding B's last word.
    hypothesis = transcript(('Person 1', 'one two three four five six seven eight'), ('Person 2', 'nine'), ('Person 3', 'ten'))
    result = speaker_attribution(reference, hypothesis)
    assert result['speaker_accuracy'] == pytest.approx(7 / 10)  # P1->A (6) + P2->B (1); P3 unmapped.
    assert result['purity']['Person 1']['purity'] == pytest.approx(6 / 8)
    assert result['purity']['Person 3']['mapped_to'] is None
    assert (result['hypothesis_speakers'], result['reference_speakers']) == (3, 2)


def test_alignment_ignores_inserted_and_deleted_words():
    reference = transcript(('A', 'the trial court never found'), ('B', 'that is correct'))
    hypothesis = transcript(('X', 'um the trial court found'), ('Y', 'that is correct indeed'))
    result = speaker_attribution(reference, hypothesis)
    assert result['aligned_words'] == 7 and result['speaker_accuracy'] == 1.0


def test_mid_sentence_turn_rate():
    hypothesis = transcript(('A', 'We will hear argument'), ('B', 'this morning.'), ('C', 'Thank you.'), ('A', 'Counsel?'))
    assert mid_sentence_turns(hypothesis) == pytest.approx(1 / 3)


# Synthetic text in the layout pypdf extracts from official transcripts.
OFFICIAL = """1
2
3
Official - Subject to Final Review
3
C O N T E N T S
ORAL ARGUMENT OF:            PAGE:
JOSEPH J. PERKOVICH, ESQ.
Heritage Reporting Corporation
1
2
3
4
Official - Subject to Final Review
4
P R O C E E D I N G S
 (10:27 a.m.)
 CHIEF JUSTICE ROBERTS:  We will hear
 argument this morning in Case 24-7351.
 ORAL ARGUMENT OF JOSEPH J. PERKOVICH
 ON BEHALF OF THE PETITIONER
 MR. PERKOVICH:  Mr. Chief Justice, may
it please the Court: the (d)(2) question.
Heritage Reporting Corporation
5
6
Official - Subject to Final Review
5
 JUSTICE KAGAN:  Is that right?
 (Laughter.)
 MS. FERGUSON:  Yes, step 2 matters.
 CHIEF JUSTICE ROBERTS:  Thank you,
 counsel.  The case is submitted.
 (Whereupon, at 12:17 p.m., the case
 was submitted.)
Heritage Reporting Corporation
Official - Subject to Final Review
128
definitely [1] 94:2
"""


def test_supreme_court_transcript_conversion():
    reference = parse_transcript(OFFICIAL, duration=60)
    assert [(t['speaker_id'], t['text']) for t in reference['turns']] == [
        ('CHIEF JUSTICE ROBERTS', 'We will hear argument this morning in Case 24-7351.'),
        ('MR. PERKOVICH', 'Mr. Chief Justice, may it please the Court: the (d)(2) question.'),
        ('JUSTICE KAGAN', 'Is that right?'),
        ('MS. FERGUSON', 'Yes, step 2 matters.'),
        ('CHIEF JUSTICE ROBERTS', 'Thank you, counsel. The case is submitted.'),
    ]
    assert [s['id'] for s in reference['speakers']] == ['CHIEF JUSTICE ROBERTS', 'MR. PERKOVICH', 'JUSTICE KAGAN', 'MS. FERGUSON']
    assert all(t['start'] == t['end'] == 0 for t in reference['turns'])


def test_scanned_pdf_without_text_is_rejected(tmp_path, monkeypatch):
    pypdf = pytest.importorskip('pypdf')
    from eval import scotus
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with (tmp_path / 'scan.pdf').open('wb') as handle:
        writer.write(handle)
    with pytest.raises(ValueError, match='no text layer'):
        scotus.load_scotus(tmp_path / 'scan.pdf')


def test_untimed_reference_scores_without_der_and_app_exports_load(tmp_path):
    from eval.evaluate import load_prediction, scores
    export = tmp_path / 'export.txt'
    export.write_text('[00:00:00] Person 1: We will hear argument this morning in Case 24-7351.\n'
                      '[00:00:05] Person 2: Mr. Chief Justice, may it please the Court: the (d)(2) question.\n', encoding='utf-8')
    prediction = load_prediction(export)
    reference = parse_transcript(OFFICIAL, duration=60)
    reference['turns'] = reference['turns'][:2]
    values = scores(reference, prediction)
    assert values['DER'] is None
    assert values['WER'] == 0 and values['speaker_accuracy'] == 1.0
