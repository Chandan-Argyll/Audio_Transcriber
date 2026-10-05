"""Speaker attribution: how many correctly aligned words carry the right speaker.

Reference and hypothesis words are normalized one by one and aligned with jiwer.
Every aligned pair (a hit or a substitution) adds one count to a hypothesis-speaker
x reference-speaker matrix. The best one-to-one mapping (Hungarian algorithm)
gives the accuracy: the share of aligned words whose hypothesis speaker maps to
their true speaker. Unmapped hypothesis speakers (more Persons than real
speakers) count as wrong. Purity is, per hypothesis speaker, the share of its
aligned words spoken by its single most frequent real speaker. Turns starting
mid-sentence are hypothesis turns whose previous turn does not end with . ? or !.
"""
from collections import defaultdict
from typing import Any

SENTENCE_END = ('.', '?', '!')


def _words(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    from eval.evaluate import normalize_text
    tokens, speakers = [], []
    for turn in data['turns']:
        for word in (turn.get('words') or [{'w': w} for w in turn['text'].split()]):
            for token in normalize_text(word['w']).split():
                tokens.append(token)
                speakers.append(turn['speaker_id'])
    return tokens, speakers


def mid_sentence_turns(hypothesis: dict[str, Any]) -> float:
    turns = [t for t in hypothesis['turns'] if t['text'].strip()]
    if len(turns) < 2:
        return 0.0
    broken = sum(not prev['text'].rstrip().rstrip('"\')]').endswith(SENTENCE_END) for prev in turns[:-1])
    return broken / (len(turns) - 1)


def speaker_attribution(reference: dict[str, Any], hypothesis: dict[str, Any]) -> dict[str, Any]:
    import jiwer
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    ref_tokens, ref_speakers = _words(reference)
    hyp_tokens, hyp_speakers = _words(hypothesis)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    if ref_tokens and hyp_tokens:
        output = jiwer.process_words(' '.join(ref_tokens), ' '.join(hyp_tokens))
        for chunk in output.alignments[0]:
            if chunk.type in ('equal', 'substitute'):
                for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    counts[(hyp_speakers[chunk.hyp_start_idx + offset], ref_speakers[chunk.ref_start_idx + offset])] += 1
    hyp_labels = sorted({h for h, _ in counts})
    ref_labels = sorted({r for _, r in counts})
    matrix = np.array([[counts[(h, r)] for r in ref_labels] for h in hyp_labels]) if counts else np.zeros((0, 0))
    total = int(matrix.sum())
    mapping: dict[str, str] = {}
    correct = 0
    if total:
        rows, cols = linear_sum_assignment(-matrix)
        mapping = {hyp_labels[r]: ref_labels[c] for r, c in zip(rows, cols)}
        correct = int(matrix[rows, cols].sum())
    purity = {}
    for i, label in enumerate(hyp_labels):
        row = matrix[i]
        dominant = int(row.argmax())
        purity[label] = {'words': int(row.sum()), 'dominant': ref_labels[dominant],
                         'purity': float(row[dominant] / row.sum()) if row.sum() else 0.0,
                         'mapped_to': mapping.get(label)}
    return {'speaker_accuracy': correct / total if total else 0.0, 'aligned_words': total,
            'mid_sentence_turns': mid_sentence_turns(hypothesis), 'mapping': mapping, 'purity': purity,
            'reference_speakers': len({s for s in ref_speakers}), 'hypothesis_speakers': len(set(hyp_speakers))}
