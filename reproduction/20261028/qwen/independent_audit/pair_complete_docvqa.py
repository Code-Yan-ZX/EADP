"""Pair OLD and completed new DocVQA5349 only; not a sole-FA2 causal test."""
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import audit_qwen as audit

HERE = Path(__file__).resolve().parent
OLD = Path('/media/disk2/YZX/research/audit_base_gap_20261003/archive_recovery/20261003_r2/old_predictions/old_DocVQA_VAL.xlsx')
OLD_SHA = '3066a258532725f6181b3f5c4a1d84ee230e7ebe7bb00dcf44b18c99ad3c823c'
TOLERANCE = 1e-12


def main():
    preliminary_path = HERE / 'docvqa_preterminal_20261028.json'
    preliminary = json.loads(preliminary_path.read_text())
    assert preliminary['n'] == 5349 and preliminary['prediction_validity']['coverage_complete']
    new_path = Path(preliminary['raw']['path'])
    assert audit.sha(new_path) == preliminary['raw']['sha256']
    assert audit.sha(OLD) == OLD_SHA
    truth_path = HERE / 'DocVQA_VAL.canonical.jsonl'
    assert audit.sha(truth_path) == preliminary['canonical']['sha256']
    truth = audit.jsonl(truth_path)
    old, new = audit.read_xlsx(OLD), audit.read_xlsx(new_path)
    assert len(truth) == len(old) == len(new) == 5349
    old_valid = audit.validate_predictions(truth, old, 5349)
    new_valid = audit.validate_predictions(truth, new, 5349)
    paired = []
    for canonical, left, right in zip(truth, old, new):
        gt = audit.parse_answers(canonical['answer'])
        a = audit.scoring_prediction(left['prediction'])
        b = audit.scoring_prediction(right['prediction'])
        before, after = audit.independent_anls(a, gt), audit.independent_anls(b, gt)
        delta = after - before
        outcome = 'improved' if delta > TOLERANCE else 'decreased' if delta < -TOLERANCE else 'equal'
        row = dict(index=canonical['index'], raw_old=left['prediction'], raw_new=right['prediction'],
                   native_scored_old=a, native_scored_new=b,
                   raw_answer_changed=left['prediction'] != right['prediction'],
                   native_scored_answer_changed=a != b, old_score=before, new_score=after,
                   score_delta=delta, outcome=outcome,
                   old_native_NA_conversion=a != left['prediction'],
                   new_native_NA_conversion=b != right['prediction'])
        if not row['raw_answer_changed']:
            assert outcome == 'equal'
        paired.append(row)
    old_value = 100*sum(r['old_score'] for r in paired)/5349
    new_value = 100*sum(r['new_score'] for r in paired)/5349
    assert abs(old_value-61.135577708229846) < 1e-10
    assert abs(new_value-preliminary['native_ANLS_percent']) < 1e-10
    new_rows = audit.jsonl(preliminary['independent_per_question']['path'])
    assert audit.sha(preliminary['independent_per_question']['path']) == preliminary['independent_per_question']['sha256']
    assert all(r['index'] == s['index'] and abs(r['new_score']-s['independent_score']) < TOLERANCE
               for r,s in zip(paired,new_rows))
    changed = [r for r in paired if r['raw_answer_changed']]
    counts = {key:sum(r['outcome']==key for r in paired) for key in ('improved','decreased','equal')}
    changed_counts = {key:sum(r['outcome']==key for r in changed) for key in ('improved','decreased','equal')}
    assert sum(counts.values()) == 5349 and sum(changed_counts.values()) == len(changed)
    gains = sum(max(0.,r['score_delta']) for r in paired)
    losses = sum(min(0.,r['score_delta']) for r in paired)
    assert abs(100*(gains+losses)/5349-(new_value-old_value)) < 1e-10
    paired_path = HERE / 'docvqa_full_pair_20261028.per_question.jsonl'
    paired_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in paired))
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,GPU_or_API_used=False,
        scope='Two complete ordered5349 DocVQA outputs only; no AI2D prediction prefix read or scored',
        full_denominator=5349, ordered_ID_question_GT_pairs_verified=5349,
        old_raw=dict(path=str(OLD),sha256=OLD_SHA),new_raw=preliminary['raw'],
        preliminary_full_Doc_report=dict(path=str(preliminary_path),sha256=audit.sha(preliminary_path)),
        canonical=dict(path=str(truth_path),sha256=audit.sha(truth_path)),
        old_validity=old_valid,new_validity=new_valid,
        main_metric='Unchanged released ANLS and native pandas default-NA conversion; no replacement score chosen',
        old_ANLS_percent=old_value,new_ANLS_percent=new_value,
        native_new_ANLS_percent=preliminary['native_ANLS_percent'],
        new_minus_old_percentage_points=str(Decimal(str(preliminary['native_ANLS_percent']))-Decimal('61.135577708229846')),
        raw_answers_changed=len(changed),raw_answers_unchanged=5349-len(changed),
        native_scored_answers_changed=sum(r['native_scored_answer_changed'] for r in paired),
        paired_score_counts=counts,changed_raw_answer_score_counts=changed_counts,
        score_comparison_tolerance=TOLERANCE,gross_gain_ANLS_sum=gains,gross_loss_ANLS_sum=losses,
        old_native_NA_conversions=sum(r['old_native_NA_conversion'] for r in paired),
        new_native_NA_conversions=sum(r['new_native_NA_conversion'] for r in paired),
        old_raw_empty_count=old_valid['empty_count'],new_raw_empty_count=new_valid['empty_count'],
        paired_rows=dict(path=str(paired_path),sha256=audit.sha(paired_path)),
        script_sha256=audit.sha(__file__),frozen_auditor_or_watcher_modified=False,
        whole_sequence_terminal_gate_passed=False,formal_ledger_eligible=False,
        causal_boundary='The source runtime/environment, attention backend and public-entry packaging differ across the preserved OLD run and the new author FA2 run. These paired outcomes are not a single-factor FA2 intervention; they do not attribute gains or losses solely to FA2.',
        limits=['The new public sequence still awaits AI2D and actual controller/container exit plus the frozen terminal audit.',
                'The unchanged complete benchmark denominator and per-question pairing do not prove the paper full argv.'])
    target = HERE / 'docvqa_full_pair_20261028.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['old_ANLS_percent','new_ANLS_percent','new_minus_old_percentage_points',
         'raw_answers_changed','paired_score_counts','changed_raw_answer_score_counts','old_native_NA_conversions',
         'new_native_NA_conversions','old_raw_empty_count','new_raw_empty_count']},ensure_ascii=False,indent=2))
    print('report_sha256',audit.sha(target))
    assert 'torch' not in sys.modules


if __name__ == '__main__':
    main()
