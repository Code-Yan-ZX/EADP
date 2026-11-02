"""Append one fixed POPE beta hypothesis to the unchanged eleven-run ledger.

Metadata export only. No generation, evaluator, verifier, or prefix-score reads.
Pending registration requires a real exclusive launch claim and controller state.
Complete registration additionally requires the root-owned actual-exit proof.
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
BASE = HERE / 'full_run_ledger_v3.json'
NEW = HERE / 'pope/full_alpha1_beta2_single_control_run02'
RUN = NEW / 'execution'
RUN_ID = 'POPE_full8910_alpha1_beta2_K32_single_beta_control_run02'
PLAN_SHA = '4704a777360b042b4d1e63dd9be4458f50d37a0234bfe09d4169e675e268ea5b'
ALLOCATION_SHA = 'a9f6d6d3f87dc86dbf530388f110fa72c2529381b1543fb910f75ebab946cf0f'
LAUNCH_SHA = '2284f1a80bb97a1bbe5d96e0555c4c04f9680abe28514b5de10b257c0e7262d1'
ROOT_EXEC_HANDLE = 23602
IMMUTABLE = {
    ROOT / 'docs/versions/v2/eadp_comparison_20261028/comparison.json': 'fe9e3ba86ae5b353699d2b17427df392b11a2ff26ff8a2be9cc3956bd0f5a01d',
    ROOT / 'docs/versions/v2/evidence.json': '8f41a119427a78b360a244949c9f8a0e44dd6fe0e6f615390cccd3b3615b2013',
    HERE / 'full_run_ledger.json': 'cb3f29cc5197bebc735f7f99f0b9885c96208f05e960c5eafb0266f3571eed32',
    HERE / 'build_full_run_ledger.py': '0cd8df57fb4981c37cb373f95da1645f4c19b2d295d74a6fcfbf730cfc8ceae3',
    HERE / 'full_run_ledger_v2.json': '5b19d5a62d4b05dff0c430063766649023f4f123633814352f5da9f574d9ac50',
    HERE / 'build_full_run_ledger_v2.py': '81032460c77be62828aa34a8e7f2aa18a704c69693356b78fc95881396a6db1f',
    BASE: '1744fb69f5a42660918f12eb49ed1daf342ef2fa226ef5d10db189ed6313ff03',
    HERE / 'build_full_run_ledger_v3.py': '166d3ee00a5ad910d75771377048d52bb2e7cef15209c0600f2bce2fd63d8ec7',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
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


def validate_started(plan, ready, cpu, note, claim, state, note_sha):
    """Pure metadata gate; no pending row for an unlaunched prepared plan."""
    assert len(PLAN_SHA) == 64
    assert plan['run_id'] == note['run_id'] == RUN_ID
    assert ready['plan_sha256'] == cpu['plan_sha256'] == note['plan_sha256'] == PLAN_SHA
    assert claim['plan_sha256'] == state['plan_sha256'] == PLAN_SHA
    assert claim['controller'] == state['controller']
    for observed in (claim['controller'], state['native_worker']):
        assert set(observed) == {'pid', 'start_ticks', 'boot_id'}
        assert type(observed['pid']) is int and observed['pid'] > 0
        assert type(observed['start_ticks']) is int and observed['start_ticks'] > 0
        assert observed['boot_id'] == note['boot_id']
    assert claim['controller'] != state['native_worker']
    assert claim['allocation_note_sha256'] == state['allocation_note_sha256'] == note_sha == ALLOCATION_SHA
    assert claim['launch_attempt'] == 1 and claim['automatic_retry'] is False
    assert state['automatic_retry'] is False
    assert state['status'] in {'generating', 'native_exited', 'complete', 'terminal_failure_no_retry'}
    assert note['authorization'] == 'explicit_root_gpu_allocation'
    assert note['single_attempt'] is True and note['automatic_retry'] is False
    assert note['GPU_slot'] in {'A', 'B'} and note['GPU_device'] == 0
    assert note['maximum_concurrent_models_including_loading'] == 2
    assert len(note['allowed_other_gpu_processes']) < 2
    assert cpu['success'] is True and cpu['GPU_started'] is False
    assert plan['n_questions'] == 8910 and plan['paper_argv_confirmed'] is False
    assert plan['score_direction_used_for_acceptance'] is False
    assert plan['parameters'] == dict(visual_token_num=32, alpha=1., beta=2., entropy_keep_ratio=.2,
        temperature=0., top_p=None, num_beams=1, max_new_tokens=128, conv_mode='vicuna_v1',
        use_cache=True, source_caller_stream_wait=True)


def started_metadata():
    names = ['plan.json', 'READY.json', 'cpu_acceptance.json', 'root_gpu_allocation.json',
             'launch.claim.json', 'state.json']
    loaded = {name:read(NEW / name) for name in names}
    assert sha(NEW / 'plan.json') == PLAN_SHA
    assert loaded['root_gpu_allocation.json']['CPU_acceptance_sha256'] == sha(NEW / 'cpu_acceptance.json')
    validate_started(*(loaded[name] for name in names), sha(NEW / 'root_gpu_allocation.json'))
    launch = read(NEW / 'root_launch_process_snapshot.json')
    assert sha(NEW / 'root_launch_process_snapshot.json') == LAUNCH_SHA
    assert launch['root_owned_exec_handle'] == ROOT_EXEC_HANDLE
    assert launch['plan_sha256'] == PLAN_SHA and launch['allocation_note_sha256'] == ALLOCATION_SHA
    processes = {entry['role']:entry for entry in launch['processes']}
    assert len(launch['processes']) == len(processes) == 2
    for role, expected in [('controller', loaded['state.json']['controller']),
                           ('native_worker', loaded['state.json']['native_worker'])]:
        assert {key:processes[role][key] for key in ('pid', 'start_ticks', 'boot_id')} == expected
    return loaded['plan.json'], loaded['root_gpu_allocation.json'], loaded['launch.claim.json'], loaded['state.json']


def check_frozen_inputs(plan):
    for group in ['source_sha256', 'implementation_sha256', 'reference_sha256', 'score_only_sha256']:
        for path, expected in plan[group].items():
            assert sha(path) == expected, path
    for group in ['model_stat_guards', 'image_stat_guards']:
        for path, expected in plan[group].items():
            p = Path(path)
            s = p.stat()
            observed = dict(resolved=str(p.resolve()), bytes=s.st_size, device=s.st_dev,
                            inode=s.st_ino, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
            assert observed == {key:value for key,value in expected.items() if key != 'prior_verified_sha256'}, path


def completed_row(paper, plan, note, claim, state):
    """Read new raw only after terminal/full-count/source/score-proof gates."""
    proof = read(NEW / 'root_terminal_process_proof.json')
    launch = read(NEW / 'root_launch_process_snapshot.json')
    release = read(RUN / 'release.json')
    verification = read(RUN / 'independent_verification.json')
    assert state['status'] == 'complete' and state['native_exit_code'] == state['CPU_verifier_exit_code'] == 0
    assert proof['root_owned_exec_actual_exit_code'] == 0 and proof['all_original_processes_exited'] is True
    assert proof['root_owned_exec_handle'] == launch['root_owned_exec_handle'] == ROOT_EXEC_HANDLE
    assert sha(NEW / 'root_launch_process_snapshot.json') == LAUNCH_SHA
    assert proof['controller'] == claim['controller'] == state['controller'] == release['controller']
    assert proof['native_worker'] == state['native_worker'] == release['native_worker']
    assert proof['launch_snapshot_sha256'] == LAUNCH_SHA
    assert proof['plan_sha256'] == release['plan_sha256'] == verification['plan_sha256'] == PLAN_SHA
    note_sha = sha(NEW / 'root_gpu_allocation.json')
    assert proof.get('allocation_note_sha256', proof.get('allocation_sha256')) == note_sha
    assert release['allocation_note_sha256'] == note_sha
    assert sha(RUN / 'release.json') == proof['release_sha256'] == state['release_sha256']
    assert sha(RUN / 'independent_verification.json') == proof['independent_verification_sha256'] == release['independent_verification_sha256']
    assert release['success'] is True and release['actual_native_exited'] is True and release['native_exit_code'] == 0
    assert release['automatic_retry'] is False and release['all_generation_completed_before_scoring'] is True
    assert release['score_direction_used_for_acceptance'] is False
    assert verification['success'] is True and verification['complete'] is True
    for key in ['n_predictions', 'actual_guidance_checks', 'input_ID_checks', 'output_redecode_checks', 'budget_mask_prefix_checks']:
        assert verification[key] == 8910, key
    assert release['n_predictions'] == 8910 and verification['panel_beta2_anchor_checks'] == 16
    assert verification['full_denominator_retained'] is True and verification['score_direction_gate'] is False
    assert verification['paper_argv_confirmed'] is False and verification['all_actual_visual_counts'] == 32
    assert verification['model_loaded'] is False and verification['CUDA_initialized'] is False
    assert verification['CPU_only'] is True and verification['selector_executed'] is False
    assert verification['all_input_output_gates_before_this_verifier_GT_or_scoring'] is True
    assert verification['official_scorer_exit_code'] == 0
    assert verification['model_ids'] == {'llava-v1.5-7b':8910}
    assert verification['actual_native_exit'] == dict(original=state['native_worker'], original_exited=True, GPU_pid_absent=True)
    for original in (proof['controller'], proof['native_worker']):
        assert identity(original['pid']) != original, original
    gpu_text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    gpu_pids = {int(value.strip()) for value in gpu_text.splitlines() if value.strip()}
    assert not gpu_pids.intersection([proof['controller']['pid'], proof['native_worker']['pid']])
    check_frozen_inputs(plan)
    assert {key:value['n'] for key,value in verification['categories'].items()} == {'random':2910, 'popular':3000, 'adversarial':3000}
    for category in verification['categories'].values():
        assert sum(category[key] for key in ['TP', 'FP', 'TN', 'FN']) == category['n']
    value = verification['independent_macro_f1_percent']
    author = verification['official_macro_f1_percent']
    assert abs(value - author) < 1e-12
    assert value == release['independent_macro_f1_percent'] and author == release['official_macro_f1_percent']
    assert Decimal(str(value)).is_finite() and Decimal('0') <= Decimal(str(value)) <= Decimal('100')
    # First reads of new raw/runtime occur here, after every terminal gate above.
    names = {'predictions.jsonl', 'runtime_observations.jsonl', 'native_complete.json', 'loaded_runtime.json',
             'native_args.json', 'process_identity.json', 'wait_patch_proof.json'}
    assert set(verification['artifacts']) == names
    for name, entry in verification['artifacts'].items():
        assert Path(entry['path']) == RUN / name and sha(entry['path']) == entry['sha256']
        if name.endswith('.jsonl'):
            with Path(entry['path']).open() as stream:
                assert sum(bool(line.strip()) for line in stream) == 8910, name
    assert sha(RUN / 'predictions.jsonl') == verification['predictions_sha256'] == release['predictions_sha256']
    assert sha(RUN / 'runtime_observations.jsonl') == release['runtime_observations_sha256']
    assert sha(RUN / 'native_complete.json') == verification['native_complete_sha256']
    native = read(RUN / 'native_complete.json')
    assert native['success'] is True and native['n_predictions'] == native['n_runtime_records'] == 8910
    assert native['panel_beta2_anchor_checks'] == 16 and native['all8910_generation_completed_before_scoring'] is True
    loaded = read(RUN / 'loaded_runtime.json')
    assert loaded['same_process_compiled_code_equal'] is True
    assert loaded['actual_eval_model_structural_fingerprint'] == plan['eval_model_structural_fingerprint']
    assert loaded['unmodified_fresh_eval_model'] is True and loaded['backend'] == 'sdpa'
    for key in ['official_score_log', 'qualified_metric_verification', 'qualified_score_helper']:
        entry = verification[key]
        assert sha(entry['path']) == entry['sha256']
    qualified = read(verification['qualified_metric_verification']['path'])
    assert qualified['complete'] is True and qualified['n_predictions'] == 8910
    assert qualified['official_macro_f1_percent'] == author and qualified['independent_macro_f1_percent'] == value
    assert qualified['predictions_sha256'] == verification['predictions_sha256']
    assert qualified['categories'] == verification['categories']
    return dict(run_id=RUN_ID, model='LLaVA-1.5-7B', task='POPE', token_budget=32,
        alpha=1., beta=2., three_wait_nodes=True, n=8910, n_circular_groups=None,
        value=value, paper=paper, local_minus_paper=str(Decimal(str(value)) - Decimal(str(paper))),
        unit='percentage_points', metric='official category-macro F1', author_value=author, independent_value=value,
        author_minus_independent=str(Decimal(str(author)) - Decimal(str(value))), author_independent_metric_agreement=True,
        configuration_hypothesis_only=True, paper_argv_confirmed=False,
        source_recipe_identity='Fixed beta1-to2 public polarization hypothesis with author POPE alpha1; main paper argv unconfirmed.',
        predictions=str(RUN / 'predictions.jsonl'), prediction_sha256=verification['predictions_sha256'],
        verification=str(RUN / 'independent_verification.json'), verification_sha256=sha(RUN / 'independent_verification.json'),
        category_counts={key:value['n'] for key,value in verification['categories'].items()},
        empty_answers_preserved_in_official_denominator=verification['empty_answers_preserved_in_official_denominator'],
        EOS_count=verification['EOS_count'], non_EOS_max128_count=verification['non_EOS_max128_count'],
        complete_provenance={name:artifact(path) for name,path in {
            'plan':NEW / 'plan.json', 'allocation_note':NEW / 'root_gpu_allocation.json',
            'launch_snapshot':NEW / 'root_launch_process_snapshot.json', 'controller_state':NEW / 'state.json',
            'release':RUN / 'release.json', 'root_actual_process_exit':NEW / 'root_terminal_process_proof.json',
            'qualified_metric_verification':Path(verification['qualified_metric_verification']['path']),
            'official_score_log':Path(verification['official_score_log']['path']),
            'actual_runtime_observations':RUN / 'runtime_observations.jsonl'}.items()},
        stability_boundary='Complete8910 actual guidance/LLM IDs/output redecoding/mask-prefix and16 accepted beta2 anchors; unchanged author/independent full category macro. One configuration hypothesis; no score selection or paper-argv claim.')


def main():
    for path, expected in IMMUTABLE.items():
        assert sha(path) == expected, ('Frozen predecessor changed', str(path))
    base = read(BASE)
    assert len(base['full_runs']) == 11 and not base['pending_full_runs'] and not base['verification_failed_full_runs']
    for row in base['full_runs']:
        assert sha(row['predictions']) == row['prediction_sha256']
        assert sha(row['verification']) == row['verification_sha256']
    report = copy.deepcopy(base)
    report.update(ledger_version=4, parent_ledger=artifact(BASE), ledger_builder=artifact(Path(__file__).resolve()))
    paper_source = Path(base['paper_source']['path'])
    assert sha(paper_source) == base['paper_source']['sha256']
    matches = [row for row in read(paper_source)['rows'] if row['model_key'] == 'v15' and row['task'] == 'POPE' and row['budget_nominal'] == 32]
    assert len(matches) == 1 and matches[0]['paper_value'] == 86.7
    registration = dict(run_id=RUN_ID, model='LLaVA-1.5-7B', task='POPE', token_budget=32,
        alpha=1., beta=2., three_wait_nodes=True, n=None, value=None, paper=86.7, local_minus_paper=None,
        configuration_hypothesis_only=True, paper_argv_confirmed=False, prefix_scored=False,
        plan=dict(path=str(NEW / 'plan.json'), sha256=PLAN_SHA),
        source_recipe_identity='Fixed beta single control; paper argv unconfirmed.')
    claim_path, state_path = NEW / 'launch.claim.json', NEW / 'state.json'
    if claim_path.exists() and state_path.exists():
        plan, note, claim, state = started_metadata()
        failures = [path for path in [NEW / 'failure.json', NEW / 'worker_failure.json'] if path.exists()]
        if failures or state['status'] == 'terminal_failure_no_retry':
            report['verification_failed_full_runs'].append(dict(registration, status='terminal_failure_no_retry',
                failures=[artifact(path) for path in failures], controller_state=artifact(state_path)))
        elif not (NEW / 'root_terminal_process_proof.json').exists():
            report['pending_full_runs'].append(dict(registration, status='pending_full_verification',
                observed_controller_status=state['status'], controller_state=artifact(state_path),
                launch_claim=artifact(claim_path),
                reason='Actual launch registered. Need complete8910 source/input/output/scoring proof and root-owned controller/native exit; new raw and score files were not read.'))
        else:
            try:
                report['full_runs'].append(completed_row(86.7, plan, note, claim, state))
            except (AssertionError, KeyError, ValueError, FileNotFoundError) as error:
                report['verification_failed_full_runs'].append(dict(registration,
                    status='verification_failed', error=str(error), root_proof=artifact(NEW / 'root_terminal_process_proof.json')))
    else:
        assert not claim_path.exists() and not state_path.exists(), 'Incomplete launch metadata; do not invent a pending row'
    assert report['full_runs'][:11] == base['full_runs']
    assert json.dumps(report['full_runs'][:11], ensure_ascii=False) == json.dumps(base['full_runs'], ensure_ascii=False)
    assert len({row['run_id'] for row in report['full_runs']}) == len(report['full_runs'])
    report['goal_complete'] = False
    report['remaining'] = [row['run_id'] for row in report['pending_full_runs'] + report['verification_failed_full_runs']] + base['remaining']
    (HERE / 'full_run_ledger_v4.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    keys = ['status', 'run_id', 'model', 'task', 'token_budget', 'alpha', 'beta', 'three_wait_nodes', 'n', 'n_circular_groups',
            'value', 'paper', 'local_minus_paper', 'unit', 'metric', 'input_hypothesis_only', 'guidance',
            'configuration_hypothesis_only', 'paper_argv_confirmed', 'stability_boundary', 'verification', 'verification_sha256']
    with (HERE / 'full_run_ledger_v4.csv').open('w', newline='') as stream:
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
