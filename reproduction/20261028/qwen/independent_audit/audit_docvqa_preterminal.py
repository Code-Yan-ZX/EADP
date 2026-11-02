"""Audit completed DocVQA only while the fixed public AI2D stage continues.

Writes separate preliminary artifacts, never the frozen terminal auditor outputs.
No AI2D prediction prefix, GPU model, API judge or generation is read/executed.
"""
from datetime import datetime, timezone
from decimal import Decimal
import ast
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import audit_qwen as audit

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / 'public_fa2_full_run02_plan.json'
RUN = HERE.parent / 'public_fa2_full_run02'
EXPECTED_PLAN = 'd1afef9e1f470d432105cb8669e25f54b9906d85b54f7685689a84b539feb467'


def proc(pid):
    try:
        text = Path('/proc', str(pid), 'stat').read_text()
        command = [x.decode() for x in Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0') if x]
        return dict(pid=pid, startticks=int(text[text.rindex(')') + 2:].split()[19]), cmdline=command)
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    assert audit.sha(PLAN) == EXPECTED_PLAN
    plan = json.loads(PLAN.read_text()); audit.source_checks(plan)
    prepared = json.loads((HERE / 'terminal_audit_preparation.json').read_text())
    for path, expected in prepared['frozen_files'].items():
        assert audit.sha(path) == expected
    initial = json.loads((HERE / (PLAN.stem + '.preparation_audit.json')).read_text())
    for path, expected in initial['metric_source_sha256'].items():
        assert audit.sha(path) == expected
    canonical_path = HERE / 'DocVQA_VAL.canonical.jsonl'
    manifest = json.loads((HERE / 'DocVQA_VAL.canonical_manifest.json').read_text())
    assert audit.sha(canonical_path) == manifest['canonical_sha256']
    assert manifest['source_sha256'] == plan['datasets']['DocVQA_VAL']['data_sha256']
    assert audit.identity(plan['datasets']['DocVQA_VAL']['data_path']) == manifest['source_stat']
    images = [r for r in json.loads((HERE / 'current_cached_images.json').read_text()) if r['dataset'] == 'DocVQA_VAL']
    assert len(images) == 1286
    for image in images:
        assert audit.identity(image['path']) == image['stat']
    paths = {p.resolve() for p in RUN.rglob(plan['model_name'] + '_DocVQA_VAL.xlsx')}
    assert len(paths) == 1
    raw = paths.pop(); raw_sha = audit.sha(raw)
    answers = audit.read_xlsx(raw)
    assert len(answers) == 5349, 'Do not score a prefix'
    canonical = audit.jsonl(canonical_path)
    validity = audit.validate_predictions(canonical, answers, 5349)
    native_path = raw.with_name(raw.stem + '_acc.csv')
    native = float(audit.read_table(native_path)[0]['Overall'])
    native_rows_path = raw.with_name(raw.stem + '_results.xlsx')
    native_rows = audit.read_xlsx(native_rows_path)
    assert len(native_rows) == 5349
    helper = audit.author_helpers()['anls_compute']
    per_question = []
    for truth, answer, scored in zip(canonical, answers, native_rows):
        assert audit.normalized_id(truth['index']) == audit.normalized_id(scored['index'])
        text = audit.scoring_prediction(answer['prediction'])
        gts = audit.parse_answers(truth['answer'])
        distance = min(helper(gt, text) for gt in gts)
        author = 1. - distance if 1. - distance >= .5 else 0.
        independent = audit.independent_anls(text, gts)
        assert abs(author - independent) < 1e-12
        distances = ast.literal_eval(scored['eval_match'])
        assert len(distances) == len(gts)
        assert abs(sum(distances)/len(distances)-float(scored['eval_score'])) < 1e-12
        native_hit = 1. - min(distances)
        native_hit = native_hit if native_hit >= .5 else 0.
        assert abs(native_hit - independent) < 1e-12
        per_question.append(dict(index=truth['index'], author_score=author,
                                 independent_score=independent, native_score=native_hit,
                                 native_eval_score_column_mean_distance=float(scored['eval_score']),
                                 raw_prediction=answer['prediction'], scored_prediction=text,
                                 prediction_class=audit.prediction_class(answer['prediction'])))
    value = 100 * sum(r['independent_score'] for r in per_question) / 5349
    assert abs(native - value) < 1e-10
    assert audit.sha(raw) == raw_sha
    worker_observations_path = HERE / 'public_fa2_full_run02.terminal_audit.observed_workers.json'
    observed = json.loads(worker_observations_path.read_text())
    doc = next(r for r in observed if r['task'] == 'DocVQA_VAL')
    ai = next(r for r in observed if r['task'] == 'AI2D_TEST')
    assert doc['identity'] == dict(pid=2364896, startticks=1331402066)
    assert proc(doc['identity']['pid']) is None
    live_ai = proc(ai['identity']['pid'])
    assert live_ai and live_ai['startticks'] == ai['identity']['startticks'] and live_ai['cmdline'] == ai['cmdline']
    assert live_ai['cmdline'][live_ai['cmdline'].index('--data') + 1] == 'AI2D_TEST'
    launcher = json.loads((RUN / 'state.json').read_text())
    assert launcher['status'] == 'running' and proc(2364766) is not None
    destination = HERE / 'docvqa_preterminal_20261028.per_question.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in per_question))
    old = Decimal('61.135577708229846'); paper = Decimal('62.8')
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), CPU_only=True, GPU_or_API_used=False,
        scope='Completed DocVQA5349 only; public AI2D generation still active. No AI2D prefix was read or scored.',
        status='complete_DocVQA_only_whole_sequence_pending', formal_ledger_eligible=False,
        whole_sequence_terminal_gate_passed=False, paper_reproduction_success_claimed=False,
        n=5349, prediction_validity=validity, native_ANLS_percent=native,
        unchanged_author_helper_ANLS_percent=100*sum(r['author_score'] for r in per_question)/5349,
        independent_ANLS_percent=value, native_per_question_checks=5349,
        native_detail_column_semantics='eval_score is mean normalized edit distance, not final ANLS. Final native hit is thresholded1-min(eval_match), per original hit_calculate.',
        native_na_prediction_conversions=sum(r['raw_prediction'] != r['scored_prediction'] for r in per_question),
        old_complete_DocVQA_ANLS_percent=float(old), new_minus_old_percentage_points=str(Decimal(str(native))-old),
        paper_reported_ANLS_percent=float(paper), local_minus_paper_percentage_points=str(Decimal(str(native))-paper),
        prediction_loader='Unchanged native pandas default-NA conversion then str(), with full5349 denominator',
        raw=dict(path=str(raw),sha256=raw_sha), native_score=dict(path=str(native_path),sha256=audit.sha(native_path)),
        native_per_question=dict(path=str(native_rows_path),sha256=audit.sha(native_rows_path)),
        independent_per_question=dict(path=str(destination),sha256=audit.sha(destination)),
        canonical=dict(path=str(canonical_path),sha256=audit.sha(canonical_path),ordered_ID_question_GT_checks=5349),
        plan=dict(path=str(PLAN),sha256=EXPECTED_PLAN), generation_recipe=plan['recipe'],
        input_evidence=dict(complete_TSV_prior_SHA=manifest['source_sha256'],current_TSV_stat_unchanged=True,
                            verified_cached_DocVQA_images=1286,current_image_stats_unchanged=True),
        backend_evidence=dict(source='Fresh author model.py:151-155 explicitly passes flash_attention_2 and dtype=auto',
            model_source_sha256=audit.sha(audit.AUTHOR/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model.py'),
            prior_four_question_loaded_runtime=initial['smoke_runtime'],
            boundary='Full-run native argv and frozen registered-class source are verified; BF16/FA2 was directly observed in the prior four-question smoke. No per-step kernel trace or post-exit full-model introspection is claimed.'),
        Doc_worker=doc, Doc_worker_actual_exit_verified=True, AI2D_current_worker=live_ai,
        whole_launcher_state=launcher, frozen_auditor_or_watcher_modified=False,
        script_sha256=audit.sha(__file__),
        limits=['This is a separate preliminary full-DocVQA audit, not the whole-sequence terminal report.',
                'The existing frozen CPU watcher must still close AI2D and actual container/controller exit before formal ledger acceptance.',
                'The public recipe does not establish the unpublished paper full argv.'])
    target = HERE / 'docvqa_preterminal_20261028.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(n=5349,native_ANLS_percent=native,independent_ANLS_percent=value,
                         validity=validity,new_minus_old=report['new_minus_old_percentage_points'],
                         local_minus_paper=report['local_minus_paper_percentage_points'],
                         AI2D_current_worker=live_ai,report_sha256=audit.sha(target)),ensure_ascii=False,indent=2))
    assert 'torch' not in sys.modules


if __name__ == '__main__':
    main()
