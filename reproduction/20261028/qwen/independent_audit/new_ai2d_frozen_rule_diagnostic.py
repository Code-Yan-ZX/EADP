"""Extend the unchanged OLD-case rule to new complete AI2D; auxiliary only."""
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_qwen as audit


def main():
    old_script = HERE / 'old_ai2d_explicit_label_diagnostic.py'
    assert audit.sha(old_script) == '24bfeb42c439d71a0cdf709b818c61285ddf8f38393a8eb2a6f3902ce7caf449'
    spec = importlib.util.spec_from_file_location('unchanged_old_rule', old_script)
    old_rule = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_rule)
    intent_path = HERE / 'new_ai2d_frozen_old_rule_intent.json'
    intent = json.loads(intent_path.read_text())
    assert intent['old_rule_script']['sha256'] == audit.sha(old_script)
    terminal_path = HERE / 'public_fa2_full_run02.terminal_audit.report.json'
    terminal = json.loads(terminal_path.read_text())
    assert terminal['full_prediction_audit_passed'] and terminal['coverage_complete']
    verification_path = HERE / 'public_fa2_full_run02.verification.json'
    verification = json.loads(verification_path.read_text())
    assert verification['complete'] and verification['pending_datasets'] == []
    result = verification['datasets']['AI2D_TEST']
    canonical_path = HERE / 'AI2D_TEST.canonical.jsonl'
    assert audit.sha(canonical_path) == 'b6b879db667ac240da969ce0e6b2fc55301c7a531b69c607c3d5c64e24079049'
    raw_path = Path(result['raw_predictions']['path'])
    point_path = Path(result['per_question']['path'])
    assert audit.sha(raw_path) == result['raw_predictions']['sha256']
    assert audit.sha(point_path) == result['per_question']['sha256']
    registry = {str(p): audit.sha(p) for p in [old_script, intent_path, terminal_path, verification_path,
                                              raw_path, point_path, canonical_path, Path(__file__)]}
    canonical, predictions, pointwise = audit.jsonl(canonical_path), audit.read_table(raw_path), audit.jsonl(point_path)
    assert len(canonical) == len(predictions) == len(pointwise) == 3088
    audit.validate_predictions(canonical, predictions, 3088)
    unresolved, matched, others = [], [], []
    correct = 0
    for truth, prediction, point in zip(canonical, predictions, pointwise):
        assert str(truth['index']) == str(point['index'])
        assert prediction['prediction'] == point['raw_prediction']
        assert audit.scoring_prediction(prediction['prediction']) == point['scored_prediction']
        choices = {k:audit.native_cell(truth[k]) for k in 'ABCD' if k in truth and audit.native_cell(truth[k]) is not None}
        option = audit.independent_option(point['scored_prediction'], choices)
        assert option == point['independent_option'] == point['author_option']
        hit = int(option == truth['answer'])
        assert hit == point['independent_score'] == point['author_score']
        correct += hit
        if option in choices:
            continue
        text = point['raw_prediction']
        match = re.fullmatch(old_rule.PATTERN, text)
        label, suffix = (match.group(1), match.group(2)) if match else (None, None)
        agrees = bool(match and label in choices and suffix == truth[label])
        case = dict(index=str(truth['index']),raw_prediction=text,original_option=option,GT=truth['answer'],
                    exact_unchanged_rule_match=agrees,label=label,suffix=suffix,
                    option_text=truth.get(label) if label else None)
        unresolved.append(case)
        if agrees:
            case['label_equals_GT'] = label == truth['answer'];matched.append(case)
        else:
            others.append(case)
    n = len(canonical)
    primary = 100 * correct / n
    assert abs(primary - result['value_percent']) < 1e-12
    assert correct == 2296 and len(unresolved) == 60
    recovered = sum(case['label_equals_GT'] for case in matched)
    minimum_for_published_rounding = int((Decimal('75.65') * n / 100).to_integral_value(rounding=ROUND_CEILING))
    minimum_for_literal_paper = int((Decimal('75.7') * n / 100).to_integral_value(rounding=ROUND_CEILING))
    assert all(audit.sha(path) == digest for path,digest in registry.items())
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,API_or_GPU_used=False,
        origin=intent['origin'],unchanged_old_pattern=old_rule.PATTERN,
        original_primary=dict(correct=correct,n=n,value_percent=primary,unchanged=True,
                              metric='Released can_infer deterministic exact matching, native fallback, full denominator'),
        unresolved_count=len(unresolved),
        unrestricted_upper_if_every_unresolved_correct=dict(correct=correct+len(unresolved),n=n,value_percent=100*(correct+len(unresolved))/n),
        additional_correct_needed_for_published_rounding=minimum_for_published_rounding-correct,
        additional_correct_needed_for_literal_paper=minimum_for_literal_paper-correct,
        unchanged_rule_exact_label_content_matches=len(matched),label_GT_agreements=recovered,
        remaining_unclassified=len(others),
        auxiliary_label_respecting_score_percent=100*(correct+recovered)/n,
        conditional_upper_if_remaining_all_correct_percent=100*(correct+recovered+len(others))/n,
        assumptions=['Strict LABEL. option-text answers are interpreted according to the explicit label.',
                     'The conditional upper respects that assumption; the unrestricted upper remains valid without it.',
                     'Only the original unresolved cases are reconsidered; originally parsed answers are unchanged.'],
        unresolved_cases=unresolved,sources_before_after_unchanged=registry,
        limits=['No real author/API judge was used or certified, and no primary score or denominator was changed.',
                'The original OLD-case rule was post-hoc. Applying it unchanged here does not turn it into a preregistered primary metric.',
                'A conditional semantic bound is not proof that an arbitrary judge cannot exceed it.'])
    target = HERE / 'new_ai2d_frozen_rule_diagnostic.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    assert 'torch' not in sys.modules
    print(json.dumps({k:v for k,v in report.items() if k not in ['unresolved_cases','sources_before_after_unchanged']},ensure_ascii=False))


if __name__ == '__main__':
    main()
