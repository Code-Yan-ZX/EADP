"""Read-only reward diagnostic metrics; no answers enter the reward functions.

The official whole-string reward is Levenshtein.ratio on raw OCR output.
Local rewards partition *teacher output*, never QA answers or student errors.
Human DocVQA answer coverage is an independent evidence-coverage proxy, not QA.
"""
from __future__ import annotations

import math
import re
import unicodedata

import Levenshtein
import numpy as np


def normalize(text: str) -> str:
    """Content normalization shared by alignment and the whole-text control."""
    return "".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def teacher_fragments(text: str, width: int = 64):
    """Line-respecting chunks, capped at width; short/omitted lines remain.

    Fragments concatenate to normalize(text). No silver OCR boxes or gold
    answers are used, so this also works for outputs with no layout metadata.
    """
    normalized, spans = "", []
    for line in text.splitlines():
        line = normalize(line)
        for at in range(0, len(line), width):
            part = line[at:at + width]
            start = len(normalized)
            normalized += part
            spans.append((start, len(normalized)))
    assert normalized == normalize(text)
    return normalized, spans


def local_rewards(teacher: str, student: str, width=64, tail_fraction=0.2,
                  reverse_alignment=False):
    target, spans = teacher_fragments(teacher, width)
    output = normalize(student)
    if not target:
        raise ValueError("Empty teacher transcription has no local reward")
    owners = np.empty(len(target), dtype=np.int32)
    for i, (a, b) in enumerate(spans):
        owners[a:b] = i
    errors = np.zeros(len(spans), dtype=np.float64)
    operations = Levenshtein.editops(target[::-1], output[::-1]) if reverse_alignment else Levenshtein.editops(target, output)
    for operation, source, dest in operations:
        # Insertion at a boundary belongs to the right fragment; insertion
        # after the final character belongs to the final fragment.
        if reverse_alignment:
            source = len(target) - source if operation == 'insert' else len(target) - 1 - source
        idx = int(owners[min(max(source, 0), len(target) - 1)])
        errors[idx] += 1
    lengths = np.array([b-a for a, b in spans], dtype=np.float64)
    losses_unclipped = errors / lengths
    losses = np.minimum(losses_unclipped, 1.)
    k = max(1, math.ceil(len(losses) * tail_fraction))
    # Clipping prevents long hallucinated runs from dominating every reward;
    # the unclipped length-normalized control is retained for auditing.
    return {
        "global_raw": float(Levenshtein.ratio(teacher, student)),
        "global_normalized": float(Levenshtein.ratio(target, output)),
        "span_mean": float(1-losses.mean()),
        "span_length_weighted": float(1-np.average(losses, weights=lengths)),
        "span_length_weighted_unclipped": float(1-errors.sum()/lengths.sum()),
        "tail20": float(1-np.sort(losses)[-k:].mean()),
        "n_fragments": len(spans), "n_tail_fragments": k,
        "teacher_normalized_chars": len(target),
        "student_normalized_chars": len(output),
        "fragment_errors": errors.tolist(),
        "fragment_lengths": lengths.astype(int).tolist(),
        "fragment_losses": losses.tolist(),
    }


def answer_present(text: str, answer: str) -> bool:
    """Strict literal coverage with numeric boundaries, no fuzzy credit.

    Whitespace is collapsed, not removed: separate numbers cannot concatenate.
    Case and Unicode compatibility are normalized. Punctuation,
    decimal points, leading zeros and signs remain significant. A value inside
    a longer number is not counted; repeated values can still make coverage
    overly optimistic, so the data builder restricts selected numeric answers
    to unique occurrences in the provided OCR (not a human location gold).
    """
    def evidence_normalize(value):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()
    haystack, needle = evidence_normalize(text), evidence_normalize(answer)
    if not needle:
        return False
    if any(c.isdigit() for c in needle):
        return re.search(r"(?<![\d.,+\-−])" + re.escape(needle) + r"(?!\d|[.,]\d)", haystack) is not None
    return needle in haystack


def question_coverage(text: str, answers: list[str]) -> bool:
    return any(answer_present(text, a) for a in answers)


def self_test():
    teacher = "Invoice number 08123\nAmount 128.60\nDate 2026-11-08\n" + "other text "*80
    same = local_rewards(teacher, teacher)
    assert all(same[k] == 1 for k in ("global_raw", "span_mean", "tail20"))
    changed = local_rewards(teacher, teacher.replace("128.60", "129.60"))
    assert changed["tail20"] < changed["span_mean"] < changed["global_raw"]
    omitted = local_rewards(teacher, teacher.replace("Amount 128.60\n", ""))
    assert omitted["tail20"] < 1 and sum(omitted["fragment_errors"]) >= 12
    assert answer_present("Total: 128.60 USD", "128.60")
    assert not answer_present("Total: 1128.60 USD", "128.60")
    assert not answer_present("Total: 128.600 USD", "128.60")
    assert not answer_present("Total: -128.60 USD", "128.60")
    assert answer_present("Total: -128.60 USD", "-128.60")
    assert not answer_present("ID: 8123", "08123")
    assert not answer_present("12 34", "1234")
    assert answer_present("12 34", "12")
    assert answer_present("Total: 128.60.", "128.60")
    assert local_rewards("a\nb", "")['tail20'] == 0
    assert local_rewards("a\nb", "a\nb plus insertion")['tail20'] < 1
    print("reward metric invariants: PASS")


if __name__ == "__main__":
    self_test()
