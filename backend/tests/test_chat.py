import os
import re
from pathlib import Path

import pytest

from eval.chat import clean, parse_chat

# Synthetic CHAT written for these tests (not corpus text), covering the SBCSAE markup.
SAMPLE = '''@UTF8
@Begin
@Participants:\tANN Ann Speaker, BOB Bob Speaker, X X Speaker, ENV ENV Environment
@Comment:\tA long header comment that
\tcontinues on a second line (with parentheses).
*ANN:\t(..) &=tsk Well \x151000_2000\x15
\t\u2308 we could go- \u2309 +... \x152000_3000\x15
*BOB:\t\u230a No \u230b . \x152500_2900\x15
*BOB:\tI made some s:tupid \u23082 comment \u23092 ? \x154000_5200\x15
*X:\t\u230a &=ex \u230b . \x155000_5500\x15
*ENV:\t&{n=MIC . \x156000_6100\x15
*ANN:\t&{l=VOX Mo:m &}l=VOX &=laugh \x157000_8000\x15
%com:\tthis dependent tier is ignored
*ANN:\t&=in &=laugh . \x158000_8500\x15
*BOB:\tmy &cheller [: cello] [% laugh] and the gradu\u23082ate \u23092 \u0294uh &-um xxx +/. \x159000_9900\x15
@End
'''


def test_markup_is_stripped():
    assert clean('(..) &=tsk Well') == 'Well'
    assert clean('&{l=F &{l=VOX Mo:m &}l=VOX') == 'Mom'
    assert clean('gradu\u23082ate \u23092 .') == 'graduate'
    assert clean('&cheller [: cello] [% laugh]') == 'cello'
    assert clean('and r- read this +/.') == 'and read this'
    assert clean('\u0294uh &-um xxx (be)cause') == 'uh um because'
    assert clean('Well\u230b2: +, A P R O X') == 'Well A P R O X'


def test_bullets_become_turns_and_non_speakers_are_ignored():
    reference = parse_chat(SAMPLE, duration=12)
    assert [s['id'] for s in reference['speakers']] == ['ANN', 'BOB']
    assert [(t['speaker_id'], t['start'], t['end'], t['text']) for t in reference['turns']] == [
        ('ANN', 1.0, 2.0, 'Well'),
        ('ANN', 2.0, 3.0, 'we could'),
        ('BOB', 2.5, 2.9, 'No'),
        ('BOB', 4.0, 5.2, 'I made some stupid comment'),
        ('ANN', 7.0, 8.0, 'Mom'),
        ('BOB', 9.0, 9.9, 'my cello and the graduate uh um'),
    ]
    assert reference['duration_sec'] == 12
    turn = reference['turns'][3]
    assert [w['w'] for w in turn['words']] == turn['text'].split()
    assert turn['words'][0]['start'] == 4.0 and turn['words'][-1]['end'] == pytest.approx(5.2)
    assert [t['id'] for t in reference['turns']] == list(range(6))


def sbc019_path() -> Path | None:
    candidates = [os.environ.get('SBC019_CHA'), Path(__file__).parent / 'fixtures' / 'SBC019.cha',
                  Path.home() / 'Downloads' / 'SBC019.cha']
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


@pytest.mark.skipif(sbc019_path() is None, reason='Set SBC019_CHA to the Santa Barbara corpus SBC019.cha (not redistributed here)')
def test_sbc019_reference():
    from eval.chat import load_chat
    reference = load_chat(sbc019_path(), duration=1309.65)
    assert {s['id'] for s in reference['speakers']} == {'FRAN', 'MELI', 'JAN', 'RON', 'BRET'}
    turns = reference['turns']
    assert len(turns) > 1000
    assert [t['text'] for t in turns[:4]] == ['Well', 'with', 'Not before I graduate', 'Well']  # 'all-' is a cut-off
    assert turns[4]['text'] == "they're really concerned about these proposals on the ballot"
    assert (turns[4]['start'], turns[4]['end']) == (3.415, 6.03)
    assert all(0 <= t['start'] < t['end'] <= 1309.65 for t in turns)
    assert [t['start'] for t in turns] == sorted(t['start'] for t in turns)
    text = ' '.join(t['text'] for t in turns)
    # No CHAT/CA markup, bullets, events or codes survive.
    assert not re.search('[&=\u2308\u2309\u230a\u230b\x15\[\]\(\)+:@\u0294_]', text)
    assert not re.search(r'\b(laugh|SNIFF|VOX|xxx)\b', text)
    assert 'stupid' in text and 'Mom' in text and 'graduate' in text
