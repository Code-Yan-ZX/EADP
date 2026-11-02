"""Post-hoc OLD AI2D case-form diagnostic; never changes primary scoring.

The parent proposed this exact rule after viewing the OLD unresolved cases.
It is not preregistered, author-judge output, or a new evaluation protocol.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_qwen as audit

PATTERN = r'([A-D])\.\s*(.+)'
RAW = Path('/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/stage1_roundtrip_pilot/legacy_full/acc/L_E_GATHER/AI2D_TEST_pred.tsv')
EXPECTED_RAW_SHA = '3c70f72548bd6d5c118d26e9ed294b4e6d100fbd11930281aeb495fc9cd10efd'


def main():
    bound_path = HERE / 'old_ai2d_can_infer_judge_upper_bound.json'
    bound = json.loads(bound_path.read_text())
    assert bound['raw']['path'] == str(RAW) and audit.sha(RAW) == bound['raw']['sha256'] == EXPECTED_RAW_SHA
    assert bound['main_score']['correct'] == 2302 and bound['main_score']['denominator'] == 3088
    assert bound['counts']['unresolved_any_nonvalid_inference'] == 60
    raw = audit.read_table(RAW)
    assert len(raw) == 3088
    canonical_path = HERE / 'AI2D_TEST.canonical.jsonl'
    canonical = audit.jsonl(canonical_path)
    assert len(canonical) == 3088
    by_index = {audit.normalized_id(row['index']): row for row in canonical}
    assert len(by_index) == 3088
    matched, other = [], []
    for case in bound['unresolved_answers']:
        item = raw[case['position']]
        assert str(item['index']) == case['index'] and item['prediction'] == case['raw_prediction']
        truth = by_index[audit.normalized_id(case['index'])]
        assert truth['answer'] == case['GT']
        match = re.fullmatch(PATTERN, case['raw_prediction'])
        label = match.group(1) if match else None
        suffix = match.group(2) if match else None
        strict_agreement = bool(match and label in truth and suffix == truth[label])
        record = dict(index=case['index'], original_prediction=case['raw_prediction'],
                      original_can_infer=case['inferred'], GT=truth['answer'],
                      syntax_fullmatch=match is not None, extracted_label=label, suffix=suffix,
                      canonical_option_text=truth.get(label) if label else None,
                      suffix_exactly_equals_option_text=strict_agreement)
        if strict_agreement:
            record['explicit_label_equals_GT'] = label == truth['answer']
            matched.append(record)
        else:
            record['literal_comma_separated_option_letters'] = bool(re.fullmatch(r'[A-D](?:\s*,\s*[A-D])+', case['raw_prediction']))
            other.append(record)
    n = 3088; correct = 2302
    recovered = sum(row['explicit_label_equals_GT'] for row in matched)
    assumed_label_score = 100 * (correct + recovered) / n
    remaining_all_correct = 100 * (correct + recovered + len(other)) / n
    general_upper = bound['conservative_judge_ceiling']['value_percent']
    assert len(matched) == 57 and recovered == 23 and len(other) == 3
    assert all(row['literal_comma_separated_option_letters'] for row in other)
    assert abs(general_upper - 100 * (correct + 60) / n) < 1e-12
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,API_or_GPU_used=False,
        new_FA2_outputs_read=False,frozen_auditor_or_watcher_modified=False,
        diagnostic_origin='Post-hoc: parent inspected OLD unresolved case forms and proposed this fixed fullmatch plus exact canonical-option-content rule before this independent replication. GT was visible. This was not preregistered.',
        raw=dict(path=str(RAW),sha256=audit.sha(RAW),n=n),
        canonical=dict(path=str(canonical_path),sha256=audit.sha(canonical_path),n=len(canonical)),
        original_bound_report=dict(path=str(bound_path),sha256=audit.sha(bound_path)),
        script_sha256=audit.sha(__file__),fixed_rule=dict(pattern=PATTERN,operation='re.fullmatch on unmodified raw prediction',
            content_check='Captured suffix must equal the canonical text at the captured label exactly; no trimming, case-folding, semantic guessing or rule tuning.'),
        original_main_score=dict(correct=correct,n=n,value_percent=100*correct/n,unchanged=True,
            metric='Released can_infer deterministic exact matching with original pandas semantics'),
        explicit_label_content_matches=len(matched),label_GT_agreements=recovered,
        label_GT_disagreements=len(matched)-recovered,remaining_unclassified=len(other),
        matches=matched,other_unresolved_answers=other,
        auxiliary_label_respecting_value=dict(correct=correct+recovered,n=n,value_percent=assumed_label_score,
            assumption='Treat every strict label+identical-option-text answer according to its explicit label. This is a semantic assumption, not an author-judge observation.'),
        conditional_upper_if_other_three_all_correct=dict(correct=correct+recovered+len(other),n=n,
            value_percent=remaining_all_correct,below_displayed_paper_75_7=remaining_all_correct<75.7,
            below_published_rounding_threshold_75_65=remaining_all_correct<75.65),
        general_unrestricted_unresolved_upper_retained=dict(correct=correct+60,n=n,value_percent=general_upper,
            withdrawn=False,assumption='Every one of the60 original unresolved answers is correct; no explicit-label constraint.'),
        interpretation='Under the explicit-label-respecting assumption, this format recovery plus all3 remaining cases is insufficient to reproduce75.7. The general60-correct upper remains76.489637%; this tighter conditional result does not prove an API judge cannot exceed it.',
        limits=['No author judge was invoked and no score was promoted to the primary table.',
                'This case-triggered post-hoc analysis must not be presented as a preregistered protocol.',
                'Neither the original2302/3088 score, full denominator, nor general unresolved upper is changed.'])
    target = HERE / 'old_ai2d_explicit_label_diagnostic.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(explicit_matches=len(matched),label_correct=recovered,other=len(other),
                         auxiliary_percent=assumed_label_score,conditional_upper_percent=remaining_all_correct,
                         general_upper_percent=general_upper,report_sha256=audit.sha(target)),indent=2))
    assert 'torch' not in sys.modules


if __name__ == '__main__':
    main()
