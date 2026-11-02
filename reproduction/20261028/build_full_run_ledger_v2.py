"""Extend the immutable nine-run ledger with the registered public K64 run.

This export reads completed evidence only. A running or unverified K64 run is
listed as pending without reading its predictions or computing a prefix score.
"""
import copy
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE / 'full_run_ledger.json'
BASE_SHA = 'cb3f29cc5197bebc735f7f99f0b9885c96208f05e960c5eafb0266f3571eed32'
NEW = HERE / 'next_textvqa/full_default_K64_public'
RUN_ID = 'NeXT_public_default_K64_alpha05_beta1_wait'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path):
    return dict(path=str(path), sha256=sha(path))


def completed_row(paper):
    run = NEW / 'execution'
    plan, note, claim = (read(NEW / name) for name in
                         ('plan.json', 'root_allocation_note.json', 'launch.claim.json'))
    proof = read(NEW / 'root_terminal_process_proof.json')
    release = read(run / 'release.json')
    verification = read(run / 'independent_verification.json')
    score = read(run / 'official.score.json')
    plan_sha = sha(NEW / 'plan.json')
    assert plan['run_id'] == note['run_id'] == release['run_id'] == RUN_ID
    assert note['authorization'] == 'explicit_root_gpu_allocation'
    assert plan_sha == note['plan_sha256'] == claim['plan_sha256']
    assert plan_sha == release['plan_sha256'] == verification['plan_sha256'] == proof['plan_sha256']
    assert sha(NEW / 'root_allocation_note.json') == claim['allocation_note_sha256'] == proof['allocation_note_sha256']
    assert proof['root_owned_exec_actual_exit_code'] == 0 and proof['all_original_processes_exited'] is True
    assert proof['controller'] == claim['controller']
    assert proof['native_worker'] == release['actual_worker_exit']['original']
    assert release['success'] and verification['success']
    assert release['actual_worker_exit']['original_exited'] and release['actual_worker_exit']['gpu_pid_absent']
    assert sha(run / 'independent_verification.json') == release['independent_verification_sha256'] == proof['independent_verification_sha256']
    assert sha(run / 'release.json') == proof['release_sha256']
    assert sha(run / 'official.score.json') == release['official_score_sha256'] == verification['official_score_sha256']
    assert verification['n_rows'] == verification['token_decoding_checks'] == verification['input_budget_observation_checks'] == verification['per_question_score_checks'] == 5000
    assert verification['full_denominator_retained'] is True
    assert verification['budget_cli'] == 64 and verification['budget_nominal'] == 320
    assert verification['paper_argv_confirmed'] is False and verification['metric_or_parameter_selection'] is False
    assert verification['CPU_only'] is True and verification['model_loaded'] is False
    assert verification['max_per_question_score_delta'] < 1e-12
    assert verification['worker_adapter']['restored_AST_identical'] is True
    assert verification['worker_adapter']['numerical_source_changed'] is False
    assert score['parameters'] == plan['parameters']
    assert score['parameters']['visual_token_num'] == 64 and score['parameters']['alpha'] == .5 and score['parameters']['beta'] == 1.
    assert score['n'] == score['token_decoding_checks'] == 5000
    assert abs(score['value'] - verification['independent_accuracy_percent']) < 1e-10
    assert score['value'] == verification['official_accuracy_percent'] == release['official_value_percent']
    for entry in verification['artifacts'].values():
        assert sha(entry['path']) == entry['sha256']
    cli = verification['fresh_author_cli']
    assert cli['exit_code'] == 0 and sha(cli['path']) == cli['sha256']
    parameters = plan['parameters']
    value = verification['independent_accuracy_percent']
    return dict(run_id=RUN_ID, model='LLaVA-NeXT-7B', task='TextVQA',
        token_budget=320, budget_cli=64, alpha=parameters['alpha'], beta=parameters['beta'],
        three_wait_nodes=True, n=5000, n_circular_groups=None, value=value, paper=paper,
        local_minus_paper=str(Decimal(str(value))-Decimal(str(paper))), unit='percentage_points',
        metric='official TextVQA accuracy', author_value=score['value'], independent_value=value,
        author_minus_independent=str(Decimal(str(score['value']))-Decimal(str(value))),
        author_independent_metric_agreement=True,
        source_recipe_identity='Public-default coverage; paper complete argv remains unverified',
        predictions=str(run / 'answers.jsonl'), prediction_sha256=sha(run / 'answers.jsonl'),
        verification=str(run / 'independent_verification.json'), verification_sha256=sha(run / 'independent_verification.json'),
        actual_retained_histogram=verification['actual_retained_histogram'],
        complete_provenance={name:artifact(path) for name,path in {
            'plan':NEW / 'plan.json', 'allocation_note':NEW / 'root_allocation_note.json',
            'author_score':run / 'official.score.json', 'release':run / 'release.json',
            'root_actual_process_exit':NEW / 'root_terminal_process_proof.json',
            'actual_masks_and_quotas':run / 'input_budget_observations.jsonl'}.items()},
        stability_boundary='Full5000 input/output tokens, actual crop masks/valid prefixes and independent pointwise scores verified. No repeated-input conflict audit or paper-recipe identity is claimed.')


def main():
    assert sha(BASE) == BASE_SHA, 'Keep the accepted nine-run ledger unchanged'
    base = read(BASE)
    assert len(base['full_runs']) == 9 and not base['pending_full_runs'] and not base['verification_failed_full_runs']
    for row in base['full_runs']:
        assert sha(row['predictions']) == row['prediction_sha256']
        assert sha(row['verification']) == row['verification_sha256']
    report = copy.deepcopy(base)
    report.update(ledger_version=2, parent_ledger=artifact(BASE), ledger_builder=artifact(Path(__file__).resolve()))
    paper_source = Path(base['paper_source']['path'])
    assert sha(paper_source) == base['paper_source']['sha256']
    reference = [row for row in read(paper_source)['rows'] if
                 row['model_key'] == 'next' and row['task'] == 'TextVQA' and row['budget_nominal'] == 320]
    assert len(reference) == 1 and reference[0]['paper_value'] == 57.0
    failure = next((p for p in (NEW / 'failure.json', NEW / 'execution/failure.json') if p.exists()), None)
    terminal = NEW / 'root_terminal_process_proof.json'
    if failure:
        report['verification_failed_full_runs'].append(dict(run_id=RUN_ID, task='TextVQA',
            status='terminal_failure_no_retry', failure=artifact(failure), prefix_scored=False))
    elif not terminal.exists():
        state_path = NEW / 'controller_state.json'
        report['pending_full_runs'].append(dict(run_id=RUN_ID, task='TextVQA',
            status='pending_full_verification', observed_controller_status=read(state_path)['status'] if state_path.exists() else 'prepared_not_launched',
            reason='All5000, independent audit and root-owned actual controller exit are required', prefix_scored=False))
    else:
        report['full_runs'].append(completed_row(reference[0]['paper_value']))
    assert report['full_runs'][:9] == base['full_runs']
    assert len({row['run_id'] for row in report['full_runs']}) == len(report['full_runs'])
    report['goal_complete'] = False
    report['remaining'] = [row['run_id'] for row in report['pending_full_runs'] + report['verification_failed_full_runs']] + base['remaining']
    (HERE / 'full_run_ledger_v2.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    keys = ['run_id','model','task','token_budget','alpha','beta','three_wait_nodes','n','n_circular_groups',
            'value','paper','local_minus_paper','unit','metric','stability_boundary','verification','verification_sha256']
    with (HERE / 'full_run_ledger_v2.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key:row[key] for key in keys} for row in report['full_runs'])
    assert 'torch' not in sys.modules
    print(json.dumps(dict(completed_full_runs=len(report['full_runs']), pending=len(report['pending_full_runs']),
        verification_failed=len(report['verification_failed_full_runs']), goal_complete=False)))


if __name__ == '__main__':
    main()
