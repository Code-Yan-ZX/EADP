"""Append one complete question-only input hypothesis to the frozen ten runs.

Metadata export only: no model, scorer, verifier or running-prefix reads.
Without the root-owned actual-exit proof the new run is pending with no value.
"""
import copy
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / 'full_run_ledger_v2.json'
NEW = HERE / 'next_textvqa/full_K64_question_only_public_comment'
RUN = NEW / 'execution'
RUN_ID = 'NeXT_full_K64_question_only_public_comment'
PLAN_SHA = '3a1bfff5c78c47d7a0676f471eadcbff89aa3b98e9407132ffeaece3dbdce7e2'
ALLOCATION_SHA = 'bdd8872140761e1bc42d7e2fda233a122461914aee2d6373837c3808561f7ebc'
SNAPSHOT_SHA = 'a88dfe5cdb733d8573dc3d414c5fa6e0a15cf3c9e871bfdecfb39ef976b72603'
IMMUTABLE = {
    ROOT / 'docs/versions/v2/eadp_comparison_20261028/comparison.json': 'fe9e3ba86ae5b353699d2b17427df392b11a2ff26ff8a2be9cc3956bd0f5a01d',
    ROOT / 'docs/versions/v2/evidence.json': '8f41a119427a78b360a244949c9f8a0e44dd6fe0e6f615390cccd3b3615b2013',
    HERE / 'full_run_ledger.json': 'cb3f29cc5197bebc735f7f99f0b9885c96208f05e960c5eafb0266f3571eed32',
    HERE / 'build_full_run_ledger.py': '0cd8df57fb4981c37cb373f95da1645f4c19b2d295d74a6fcfbf730cfc8ceae3',
    BASE: '5b19d5a62d4b05dff0c430063766649023f4f123633814352f5da9f574d9ac50',
    HERE / 'build_full_run_ledger_v2.py': '81032460c77be62828aa34a8e7f2aa18a704c69693356b78fc95881396a6db1f',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    return dict(path=str(path), sha256=sha(path))


def identity(pid):
    try:
        ticks = int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
        return dict(pid=pid, start_ticks=ticks,
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


def check_frozen_inputs(plan):
    """Check metadata SHA and strict stats; do not rehash large model payloads."""
    for entry in [plan['worker'], plan['wait_helper'], plan['questions'], plan['annotation'],
                  *plan['model_small_files'], *plan['evidence'], *plan['additional_frozen_sources']]:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    for source in plan['source_comparisons']:
        assert sha(source['author_path']) == source['author_sha256'], source['author_path']
    assert sha(plan['clip_identity']['audit']['path']) == plan['clip_identity']['audit']['sha256']
    for path, expected in plan['strict_model_stat_guards'].items():
        p = Path(path)
        s = p.stat()
        actual = dict(resolved=str(p.resolve()), bytes=s.st_size, device=s.st_dev, inode=s.st_ino,
                      mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
        assert actual == {k:v for k,v in expected.items() if k != 'prior_verified_sha256'}, path
    images = read(HERE / 'next_textvqa/full_default_K128.images.json')
    assert len(images) == plan['image_count'] == 3166
    for image in images:
        s = Path(image['path']).stat()
        assert s.st_size == image['bytes'] and s.st_mtime_ns == image['mtime_ns'], image['path']


def completed_row(paper):
    """Called only after a root proof exists; all gates precede new raw reads."""
    plan = read(NEW / 'plan.json')
    note = read(NEW / 'root_gpu_allocation.json')
    claim = read(NEW / 'launch.claim.json')
    launch = read(NEW / 'root_launch_process_snapshot.json')
    proof = read(NEW / 'root_terminal_process_proof.json')
    controller = read(NEW / 'controller_state.json')
    release = read(RUN / 'release.json')
    verification = read(RUN / 'independent_verification.json')
    native = read(RUN / 'state.json')
    assert sha(NEW / 'plan.json') == PLAN_SHA
    assert sha(NEW / 'root_gpu_allocation.json') == ALLOCATION_SHA
    assert sha(NEW / 'root_launch_process_snapshot.json') == SNAPSHOT_SHA
    assert plan['run_id'] == note['run_id'] == release['run_id'] == RUN_ID
    assert note['authorization'] == 'explicit_root_gpu_allocation'
    assert note['plan_sha256'] == claim['plan_sha256'] == proof['plan_sha256'] == PLAN_SHA
    assert release['plan_sha256'] == verification['plan_sha256'] == PLAN_SHA
    assert claim['allocation_note_sha256'] == ALLOCATION_SHA
    assert proof.get('allocation_note_sha256', proof.get('allocation_sha256')) == ALLOCATION_SHA
    assert proof['root_owned_exec_handle'] == launch['root_owned_exec_handle'] == 60045
    assert proof['root_owned_exec_actual_exit_code'] == 0 and proof['all_original_processes_exited'] is True
    assert proof['controller'] == claim['controller'] == launch['controller'] == controller['controller']
    assert proof['native_worker'] == launch['native_worker'] == controller['native_worker'] == release['actual_worker_exit']['original']
    assert proof['controller']['pid'] == 2861700 and proof['controller']['start_ticks'] == 1333486787
    assert proof['native_worker']['pid'] == 2861704 and proof['native_worker']['start_ticks'] == 1333486819
    assert controller['status'] == 'complete' and controller['native_exit_code'] == 0
    assert native['status'] == 'complete' and native['n'] == 5000
    assert release['success'] is True and verification['success'] is True
    assert release['actual_worker_exit']['original_exited'] is True and release['actual_worker_exit']['gpu_pid_absent'] is True
    for original in (proof['controller'], proof['native_worker']):
        assert identity(original['pid']) != original, ('Original process still alive', original)
    gpu_text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    gpu_pids = {int(s.strip()) for s in gpu_text.splitlines() if s.strip()}
    assert not gpu_pids.intersection([proof['controller']['pid'], proof['native_worker']['pid']])
    assert sha(RUN / 'release.json') == proof['release_sha256'] == controller['release_sha256']
    assert sha(RUN / 'independent_verification.json') == proof['independent_verification_sha256'] == release['independent_verification_sha256'] == controller['independent_verification_sha256']
    for key in ['n_rows', 'token_decoding_checks', 'input_budget_observation_checks', 'actual_guidance_checks', 'per_question_score_checks']:
        assert verification[key] == 5000
    assert release['n_rows'] == release['actual_guidance_checks'] == 5000
    assert verification['full_denominator_retained'] is True
    assert verification['budget_cli'] == 64 and verification['budget_nominal'] == 320
    for key in ['paper_argv_confirmed', 'metric_or_parameter_selection', 'model_loaded']:
        assert verification[key] is False
    assert verification['CPU_only'] is True and verification['input_hypothesis_only'] is True
    assert release['input_hypothesis_only'] is True and release['score_direction_gate'] is False
    assert verification['max_per_question_score_delta'] < 1e-12
    assert verification['worker_adapter'] == plan['worker_adapter']
    assert plan['worker_adapter']['restored_AST_identical'] is True and plan['worker_adapter']['numerical_source_changed'] is False
    hypothesis = plan['input_hypothesis']
    assert hypothesis['known_paper_argv'] is False and hypothesis['full_denominator'] == 5000
    assert hypothesis['full_LLM_OCR_prompt_or_IDS_changed'] is False and hypothesis['images_or_pruning_math_changed'] is False
    guidance = verification['guidance_input_boundary_checks']
    assert guidance['actual_guidance_checks'] == guidance['full_LLM_input_ID_checks'] == 5000
    assert guidance['GT_read_by_this_verifier_before_guidance_gate'] is False
    assert sha(guidance['reference']['path']) == guidance['reference']['sha256']
    expected_transform = plan['worker_adapter']['guidance_entry_transform']
    assert {k:guidance['actual_entry_transform'][k] for k in expected_transform} == expected_transform
    assert guidance['actual_entry_transform']['executed_eval_model_code_sha256'] == expected_transform['expected_eval_model_code_sha256']
    check_frozen_inputs(plan)
    score = read(RUN / 'official.score.json')
    assert score['n'] == score['token_decoding_checks'] == 5000
    assert score['parameters'] == plan['parameters']
    assert plan['parameters']['visual_token_num'] == 64 and plan['parameters']['alpha'] == .5 and plan['parameters']['beta'] == 1.
    assert sha(RUN / 'official.score.json') == verification['official_score_sha256'] == release['official_score_sha256'] == native['score_sha256']
    assert score['value'] == verification['official_accuracy_percent'] == release['official_value_percent']
    value = verification['independent_accuracy_percent']
    assert abs(score['value'] - value) < 1e-10
    assert value == release['independent_value_percent']
    assert Decimal(str(value)).is_finite() and Decimal('0') <= Decimal(str(value)) <= Decimal('100')
    assert float(verification['independent_exact_percent']) == value
    # First reads of new raw/pointwise artifacts occur only after full terminal gates.
    expected_names = {'answers.jsonl', 'runtime.jsonl', 'loaded_runtime.json', 'input_budget_observations.jsonl',
                      'guidance_observations.jsonl', 'per_question_scores.jsonl', 'independent_per_question_scores.jsonl'}
    assert set(verification['artifacts']) == expected_names
    for name, entry in verification['artifacts'].items():
        assert Path(entry['path']) == RUN / name and sha(entry['path']) == entry['sha256']
        if name.endswith('.jsonl'):
            with Path(entry['path']).open() as stream:
                assert sum(bool(line.strip()) for line in stream) == 5000, name
    for entry in score['artifacts'].values():
        assert sha(entry['path']) == entry['sha256']
    assert sha(RUN / 'per_question_scores.jsonl') == score['per_question_scores_sha256']
    cli = verification['fresh_author_cli']
    assert cli['exit_code'] == 0 and Path(cli['path']) == RUN / 'fresh_author_cli.txt'
    assert sha(cli['path']) == cli['sha256'] and cli['matches_exact_score_at_display_precision'] is True
    return dict(run_id=RUN_ID, model='LLaVA-NeXT-7B', task='TextVQA', token_budget=320, budget_cli=64,
        alpha=.5, beta=1., three_wait_nodes=True, n=5000, n_circular_groups=None,
        value=value, paper=paper, local_minus_paper=str(Decimal(str(value)) - Decimal(str(paper))),
        unit='percentage_points', metric='official TextVQA accuracy', author_value=score['value'], independent_value=value,
        author_minus_independent=str(Decimal(str(score['value'])) - Decimal(str(value))), author_independent_metric_agreement=True,
        input_hypothesis_only=True, guidance='question_only_before_Reference_OCR', paper_argv_confirmed=False,
        source_recipe_identity='Inactive public-source comment supports this question-only CLIP guidance hypothesis; paper argv remains unconfirmed. Complete LLM OCR input unchanged.',
        predictions=str(RUN / 'answers.jsonl'), prediction_sha256=verification['artifacts']['answers.jsonl']['sha256'],
        verification=str(RUN / 'independent_verification.json'), verification_sha256=sha(RUN / 'independent_verification.json'),
        actual_retained_histogram=verification['actual_retained_histogram'],
        complete_provenance={name:artifact(path) for name,path in {
            'plan':NEW / 'plan.json', 'allocation_note':NEW / 'root_gpu_allocation.json',
            'launch_snapshot':NEW / 'root_launch_process_snapshot.json', 'controller_state':NEW / 'controller_state.json',
            'author_score':RUN / 'official.score.json', 'release':RUN / 'release.json',
            'root_actual_process_exit':NEW / 'root_terminal_process_proof.json',
            'actual_guidance':RUN / 'guidance_observations.jsonl', 'actual_masks_and_quotas':RUN / 'input_budget_observations.jsonl',
            'independent_pointwise_scores':RUN / 'independent_per_question_scores.jsonl'}.items()},
        stability_boundary='All5000 complete actual guidance/LLM IDs/quota-mask prefixes/output decoding and author/independent pointwise score gates passed. This tests an input hypothesis, not paper recipe identity or proof of reproduction.')


def main():
    for path, expected in IMMUTABLE.items():
        assert sha(path) == expected, ('Frozen predecessor changed', str(path))
    base = read(BASE)
    assert len(base['full_runs']) == 10 and not base['pending_full_runs'] and not base['verification_failed_full_runs']
    for row in base['full_runs']:
        assert sha(row['predictions']) == row['prediction_sha256']
        assert sha(row['verification']) == row['verification_sha256']
    report = copy.deepcopy(base)
    report.update(ledger_version=3, parent_ledger=artifact(BASE), ledger_builder=artifact(Path(__file__).resolve()))
    paper_source = Path(base['paper_source']['path'])
    assert sha(paper_source) == base['paper_source']['sha256']
    reference = [row for row in read(paper_source)['rows'] if row['model_key'] == 'next' and row['task'] == 'TextVQA' and row['budget_nominal'] == 320]
    assert len(reference) == 1 and reference[0]['paper_value'] == 57.0
    registration = dict(run_id=RUN_ID, model='LLaVA-NeXT-7B', task='TextVQA', token_budget=320, budget_cli=64,
        alpha=.5, beta=1., three_wait_nodes=True, n=None, value=None, paper=57.0, local_minus_paper=None,
        input_hypothesis_only=True, paper_argv_confirmed=False, guidance='question_only_before_Reference_OCR',
        plan=dict(path=str(NEW / 'plan.json'), sha256=PLAN_SHA), prefix_scored=False,
        source_recipe_identity='Inactive public-source comment supports this input hypothesis; complete LLM OCR prompt remains unchanged.')
    failure = next((p for p in (NEW / 'failure.json', RUN / 'failure.json') if p.exists()), None)
    terminal = NEW / 'root_terminal_process_proof.json'
    if failure:
        report['verification_failed_full_runs'].append(dict(registration, status='terminal_failure_no_retry', failure=artifact(failure)))
    elif not terminal.exists():
        state_path = NEW / 'controller_state.json'
        report['pending_full_runs'].append(dict(registration, status='pending_full_verification',
            observed_controller_status=read(state_path)['status'] if state_path.exists() else 'prepared_not_launched',
            reason='Need full5000 official+independent pointwise/guidance/input/source gates and root-owned actual native/controller exit. No new raw or score file was read.'))
    else:
        report['full_runs'].append(completed_row(reference[0]['paper_value']))
    assert report['full_runs'][:10] == base['full_runs']
    assert json.dumps(report['full_runs'][:10], ensure_ascii=False) == json.dumps(base['full_runs'], ensure_ascii=False)
    assert len({row['run_id'] for row in report['full_runs']}) == len(report['full_runs'])
    report['goal_complete'] = False
    report['remaining'] = [row['run_id'] for row in report['pending_full_runs'] + report['verification_failed_full_runs']] + base['remaining']
    (HERE / 'full_run_ledger_v3.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    keys = ['status', 'run_id', 'model', 'task', 'token_budget', 'alpha', 'beta', 'three_wait_nodes', 'n', 'n_circular_groups',
            'value', 'paper', 'local_minus_paper', 'unit', 'metric', 'input_hypothesis_only', 'guidance',
            'stability_boundary', 'verification', 'verification_sha256']
    with (HERE / 'full_run_ledger_v3.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in report['full_runs']:
            writer.writerow({key:('complete' if key == 'status' else row.get(key)) for key in keys})
        for row in report['pending_full_runs'] + report['verification_failed_full_runs']:
            writer.writerow({key:row.get(key) for key in keys})
    assert 'torch' not in sys.modules
    print(json.dumps(dict(completed_full_runs=len(report['full_runs']), pending=len(report['pending_full_runs']),
        verification_failed=len(report['verification_failed_full_runs']), goal_complete=False)))


if __name__ == '__main__':
    main()
