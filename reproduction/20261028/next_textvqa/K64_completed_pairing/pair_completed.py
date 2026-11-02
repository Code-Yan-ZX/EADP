"""Pair accepted full-run scores only; no generation, normalization, or scoring.

Prepared only. With no argument this exits without reading any run artifact.
After root accepts all terminal evidence, pass its exact root-proof SHA256.
The completed-run gate runs before opening/hashing new prediction/point files.
Counters and integer-tenths arithmetic follow compare_full.py; scorer/CLI parts
are intentionally not imported or executed. Every output is created once.
"""
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
NEXT = OUT.parent
NEW = NEXT / 'full_default_K64_public'
RUN = NEW / 'execution'
RUN_ID = 'NeXT_public_default_K64_alpha05_beta1_wait'
PLAN_SHA = '9ef9b4c69727d7baf324c5e3b3d93fd8c7d23a899d466f35e8c2389f05405d0d'
NOTE_SHA = '77602cb7ef1ffba87aee01bebbd2dbe54c5e02dcf7f3b890f81200a07c5fe204'
OLD_REPORT = NEXT / 'small_budget_historical_recheck/K64_beta2/verification.json'
FULL_REPORT = NEXT / 'full_token_baseline_recheck/baseline_recheck.json'
LOCKED = {
    OLD_REPORT: '3f5becc3b40b9a8cb1ad2bd6c9bb52d81881862ee68c792d9b84d4f727e9873c',
    FULL_REPORT: '1433989686cc10cc2c6029a4b355f06a6a0f8fc6851cb2017f74e8dedef2ae29',
    NEXT / 'small_budget_historical_recheck/verification.json': 'a058bf638281fd159a4f18948dc1e6aabbcc9093091510d7affc594c0e23e108',
}
QUESTIONS_SHA = '9f60db6fb89e369263ed676dc26b48d00d4a462e73f0930c09d23394e9dcf3a3'
GT_SHA = '4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        result = [json.loads(line) for line in stream]
    require(len(result) == 5000, f'Full5000 required: {path}')
    return result


def artifact(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def checked_artifact(entry, registry, expected_path=None):
    path = Path(entry['path'])
    require(expected_path is None or path == expected_path, f'Unexpected artifact path: {path}')
    require(sha(path) == entry['sha256'], f'Changed accepted artifact: {path}')
    registry[str(path)] = artifact(path)
    return path


def terminal_gate(expected_root_sha, registry):
    """Only terminal metadata may be read here; never prediction/point files."""
    require(len(expected_root_sha) == 64 and all(c in '0123456789abcdef' for c in expected_root_sha), 'Exact root-confirmed proof SHA required')
    required = [NEW / name for name in ('root_terminal_process_proof.json', 'plan.json',
        'root_allocation_note.json', 'launch.claim.json', 'controller_state.json')]
    required += [RUN / name for name in ('state.json', 'release.json', 'independent_verification.json', 'official.score.json')]
    require(not (NEW / 'failure.json').exists() and not (RUN / 'failure.json').exists(), 'Terminal failure; no pairing')
    require(all(p.is_file() for p in required), 'Incomplete terminal evidence; no row/prefix reads')
    proof_path = required[0]
    require(sha(proof_path) == expected_root_sha, 'Root proof not explicitly accepted')
    proof = read(proof_path)
    require(proof['root_owned_exec_actual_exit_code'] == 0 and proof['all_original_processes_exited'] is True, 'Root actual exit gate')
    require(proof['plan_sha256'] == PLAN_SHA and proof['allocation_note_sha256'] == NOTE_SHA, 'Root run identity')
    require(sha(NEW / 'plan.json') == PLAN_SHA and sha(NEW / 'root_allocation_note.json') == NOTE_SHA, 'Frozen plan/allocation changed')
    plan, note, claim, controller = [read(NEW / name) for name in ('plan.json', 'root_allocation_note.json', 'launch.claim.json', 'controller_state.json')]
    native, release, verification, score = [read(RUN / name) for name in ('state.json', 'release.json', 'independent_verification.json', 'official.score.json')]
    require(controller['status'] == native['status'] == 'complete' and controller['native_exit_code'] == 0 and native['n'] == 5000, 'Run/controller not full and complete')
    require(plan['run_id'] == note['run_id'] == release['run_id'] == RUN_ID and note['authorization'] == 'explicit_root_gpu_allocation', 'Run allocation identity')
    require(PLAN_SHA == claim['plan_sha256'] == controller['plan_sha256'] == release['plan_sha256'] == verification['plan_sha256'], 'Plan links')
    require(NOTE_SHA == claim['allocation_note_sha256'], 'Allocation link')
    require(proof['controller'] == claim['controller'] == controller['controller'], 'Controller identity')
    require(proof['native_worker'] == controller['native_worker'] == release['actual_worker_exit']['original'], 'Native identity')
    require(release['actual_worker_exit']['original_exited'] is True and release['actual_worker_exit']['gpu_pid_absent'] is True, 'Native exit proof')
    require(release['success'] is True and verification['success'] is True and release['n_rows'] == 5000, 'Full verification failed')
    require(all(verification[key] == 5000 for key in ('n_rows', 'token_decoding_checks', 'input_budget_observation_checks', 'per_question_score_checks')), 'Partial verification; no prefix reads')
    require(verification['full_denominator_retained'] is True and verification['max_per_question_score_delta'] < 1e-12, 'Pointwise/full-denominator gate')
    require(verification['budget_cli'] == 64 and verification['budget_nominal'] == 320, 'Wrong budget')
    require(verification['worker_adapter']['restored_AST_identical'] is True and verification['worker_adapter']['numerical_source_changed'] is False, 'Native source adapter gate')
    require(verification['CPU_only'] is True and verification['model_loaded'] is False and verification['paper_argv_confirmed'] is False and verification['metric_or_parameter_selection'] is False, 'Verification scope')
    require(sha(RUN / 'release.json') == proof['release_sha256'] == controller['release_sha256'], 'Release hash links')
    require(sha(RUN / 'independent_verification.json') == proof['independent_verification_sha256'] == release['independent_verification_sha256'] == controller['independent_verification_sha256'], 'Independent report links')
    require(sha(RUN / 'official.score.json') == native['score_sha256'] == release['official_score_sha256'] == verification['official_score_sha256'], 'Official report links')
    require(score['n'] == score['token_decoding_checks'] == 5000 and score['plan']['sha256'] == PLAN_SHA, 'Official complete gate')
    require(score['parameters'] == plan['parameters'] and all(plan['parameters'][k] == v for k,v in {'visual_token_num':64, 'alpha':.5, 'beta':1., 'temperature':0., 'max_new_tokens':128}.items()), 'Recipe changed')
    require(score['value'] == verification['official_accuracy_percent'] == release['official_value_percent'] and abs(score['value'] - verification['independent_accuracy_percent']) < 1e-10, 'Accepted aggregate disagreement')
    for path in required:
        registry[str(path)] = artifact(path)
    # All full5000 + root exit gates passed. New row/hash access begins only now.
    for name, entry in verification['artifacts'].items():
        checked_artifact(entry, registry, RUN / name)
    require({'answers.jsonl', 'independent_per_question_scores.jsonl'} <= set(verification['artifacts']), 'Missing accepted pointwise keys')
    cli = verification['fresh_author_cli']
    require(cli['exit_code'] == 0, 'Accepted author CLI failed')
    checked_artifact(cli, registry)
    return plan, verification, proof


def tenth(value):
    require(isinstance(value, (int, float)) and math.isfinite(value), 'Invalid accepted score')
    units = round(value * 10)
    require(0 <= units <= 10 and abs(value - units / 10) < 1e-12, 'Not an accepted integer-tenths score')
    return units


def summarize(reference, current):
    counts, transitions = Counter(), Counter()
    gain = loss = 0
    for old, new in zip(reference, current):
        difference = new['units'] - old['units']
        change = 'improved' if difference > 0 else 'declined' if difference < 0 else 'unchanged'
        counts['score_' + change] += 1
        counts['raw_answer_changed' if new['answer'] != old['answer'] else 'raw_answer_unchanged'] += 1
        transitions[(old['units'], new['units'])] += 1
        gain += max(0, difference)
        loss += min(0, difference)
    old_sum, new_sum = sum(x['units'] for x in reference), sum(x['units'] for x in current)
    require(gain + loss == new_sum - old_sum and sum(counts[k] for k in ('score_improved','score_declined','score_unchanged')) == 5000, 'Pair aggregation')
    return dict(n_paired=5000, counts={k:counts[k] for k in ('raw_answer_changed','raw_answer_unchanged','score_improved','score_declined','score_unchanged')},
        reference_percent=str(Decimal(old_sum)/500), current_percent=str(Decimal(new_sum)/500),
        new_minus_reference_pp=str(Decimal(new_sum-old_sum)/500), total_gain_tenths=gain,
        total_loss_tenths_signed=loss, net_change_tenths=gain+loss,
        score_transitions=[dict(reference_tenths=a,new_tenths=b,n=n) for (a,b),n in sorted(transitions.items())])


def main(expected_root_sha):
    registry = {}
    require(not (OUT / 'comparison_report.json').exists() and not (OUT / 'paired_rows.jsonl').exists(), 'Pairing outputs already exist; do not overwrite')
    plan, accepted, proof = terminal_gate(expected_root_sha, registry)
    for path, digest in LOCKED.items():
        checked_artifact(dict(path=str(path), sha256=digest), registry)
    old, full = read(OLD_REPORT), read(FULL_REPORT)
    history = read(NEXT / 'small_budget_historical_recheck/verification.json')
    require([x for x in history['runs'] if x['run'] == 'K64_beta2'] == [old], 'Historical report linkage')
    require(old['success'] is True and old['n'] == old['per_question_scores_verified'] == 5000 and old['ordered_full_composite_IDs_prompts_runtime_and_all_GT_verified'] is True and old['recorded3166_image_manifest_equal_to_canonical'] is True, 'Historical beta2 qualification')
    require(full['success'] is True and full['n'] == 5000 and full['source_and_runtime_provenance_verified'] is True and full['ordered_IDs_full_prompts_all_GT_and3166images_verified'] is True, 'Historical FULL qualification')
    require((old['budget_cli'], old['alpha'], old['beta']) == (64,.5,2.), 'Historical recipe')
    for name, digest in [('questions', QUESTIONS_SHA), ('annotation', GT_SHA)]:
        path = checked_artifact(plan[name], registry)
        require(plan[name]['sha256'] == digest, 'Canonical input identity')
        for source in (history['source_registry'], full['sources_before_after_unchanged']):
            require(any(Path(x['path']) == path and x['sha256'] == digest for x in source), 'Historical/new input source mismatch')
    beta2_points = rows(checked_artifact(old['per_question_artifact'], registry))
    full_points = rows(checked_artifact(full['records'], registry))
    old_raw_path = Path(old['fresh_author_cli']['command'][-1])
    full_raw_path = Path(full['author_cli']['command'][-1])
    old_raw_entry = next(x for x in old['unchanged_input_registry'] if Path(x['path']) == old_raw_path)
    full_raw_entry = next(x for x in full['sources_before_after_unchanged'] if Path(x['path']) == full_raw_path)
    old_raw, full_raw = rows(checked_artifact(old_raw_entry, registry)), rows(checked_artifact(full_raw_entry, registry))
    questions, new_raw, new_points = rows(plan['questions']['path']), rows(RUN / 'answers.jsonl'), rows(RUN / 'independent_per_question_scores.jsonl')
    keys = [(i,str(q['question_id']),hashlib.sha256(q['text'].encode()).hexdigest()) for i,q in enumerate(questions)]
    require(len(set(keys)) == 5000 and len({(x[1],x[2]) for x in keys}) == 5000, 'Nonunique canonical keys')
    collected = {name:[] for name in ('new_beta1','historical_beta2','historical_FULL')}
    paired = []
    for i,(q,n,o,f,np,op,fp) in enumerate(zip(questions,new_raw,old_raw,full_raw,new_points,beta2_points,full_points)):
        key = keys[i]
        for point in (np,op,fp):
            require((point['position'],str(point['question_id']),point['prompt_sha256']) == key, f'No exact pointwise key at {i}; refuse hard pairing')
        require(op['image'] == q['image'], 'Historical beta2 image mismatch')
        for answer in (n,o,f):
            require(str(answer['question_id']) == key[1] and answer['prompt'] == q['text'] and isinstance(answer['text'],str), 'Raw prompt/ID mismatch')
        require(fp['full_answer'] == f['text'], 'FULL accepted point/raw mismatch')
        nu,ou,fu = tenth(np['independent_score']),op['consensus_units_tenths'],fp['full_score_tenths']
        require(tenth(np['official_score']) == nu and tenth(op['official_score']) == tenth(op['independent_score']) == ou and type(ou) is int and type(fu) is int and 0 <= fu <= 10, 'Accepted pointwise score disagreement')
        for name,units,answer in [('new_beta1',nu,n['text']),('historical_beta2',ou,o['text']),('historical_FULL',fu,f['text'])]:
            collected[name].append(dict(units=units,answer=answer))
        paired.append(dict(position=i,question_id=q['question_id'],image=q['image'],prompt_sha256=key[2],
            new_beta1_answer=n['text'],historical_beta2_answer=o['text'],historical_FULL_answer=f['text'],
            new_beta1_tenths=nu,historical_beta2_tenths=ou,historical_FULL_tenths=fu,
            beta1_minus_beta2_tenths=nu-ou,beta1_minus_FULL_tenths=nu-fu))
    comparisons = {label:summarize(collected[reference],collected['new_beta1']) for label,reference in [('beta1_vs_historical_beta2','historical_beta2'),('beta1_vs_historical_FULL','historical_FULL')]}
    require(Decimal(comparisons['beta1_vs_historical_beta2']['reference_percent']) == Decimal(old['independent_exact_percent']), 'Historical beta2 aggregate changed')
    require(Decimal(comparisons['beta1_vs_historical_FULL']['reference_percent']) == Decimal(full['independent_full_accuracy_percent']), 'Historical FULL aggregate changed')
    require(Decimal(comparisons['beta1_vs_historical_FULL']['current_percent']) == Decimal(accepted['independent_exact_percent']), 'Accepted new aggregate changed')
    registry[str(Path(__file__).resolve())] = artifact(Path(__file__).resolve())
    require(all(sha(x['path']) == x['sha256'] for x in registry.values()), 'Consumed input changed during pairing')
    require('torch' not in sys.modules and not any(x.startswith('llava') for x in sys.modules), 'Unexpected model/scorer import')
    paired_path = OUT / 'paired_rows.jsonl'
    with paired_path.open('x') as stream:
        stream.writelines(json.dumps(row,ensure_ascii=False)+'\n' for row in paired)
    report = dict(success=True,checked_utc=datetime.now(timezone.utc).isoformat(),n_rows=5000,
        direction='new K64 beta1 minus each historical reference',comparisons=comparisons,
        exact_ordered_position_questionID_full_prompt_SHA_pairing=True,
        same_canonical_question_and_GT_source_SHA=True,root_terminal_proof_sha256=expected_root_sha,
        input_registry_before_after_unchanged=list(registry.values()),paired_records=artifact(paired_path),
        scorer_invoked=False,new_metric=False,model_loaded=False,GPU_started=False,
        limitations=[
            'All scores are consumed from accepted full5000 pointwise artifacts; no answer normalization or GT scoring is performed here.',
            'Historical beta2/FULL wrappers, runtime observations and low-level scheduling differ; these paired differences are not a single-beta causal experiment.',
            'FULL point rows lack separate image/GT-key fields. Exact position+questionID+full-prompt SHA and the accepted report linking the same canonical input/GT/images provide the pairing; no fuzzy join is used.',
            'Old selected-token IDs and effective crop prefixes are unavailable, so no selected-content equality or causal token attribution is inferred.',
            'Public-default completion does not identify the paper main-table argv or establish paper reproduction. No score-direction selection or automatic retry.'])
    with (OUT / 'comparison_report.json').open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(dict(success=True,n_rows=5000,comparisons=comparisons)))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Prepared only. No run artifacts read. After root terminal acceptance, pass its exact root-proof SHA256.')
    else:
        require(len(sys.argv) == 2, 'Pass only the root-confirmed terminal-proof SHA256')
        main(sys.argv[1])
