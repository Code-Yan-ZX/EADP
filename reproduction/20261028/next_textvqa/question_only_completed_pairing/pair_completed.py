"""Summarize two accepted full5000 runs; never evaluate or normalize answers.

Default invocation reads no run artifact. An explicit root-confirmed SHA of the
new terminal proof is required. Both complete metadata/exit gates precede any
prediction, per-question-score or observation-file read/hash.
"""
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
NEXT = OUT.parent
N = 5000
BOOT = '1b0bf851-085c-43c5-9855-503a2909543b'
OLD = NEXT / 'full_default_K64_public'
NEW = NEXT / 'full_K64_question_only_public_comment'
OLD_SPEC = dict(directory=OLD, run_id='NeXT_public_default_K64_alpha05_beta1_wait',
    note_filename='root_allocation_note.json', root_allocation_field='allocation_note_sha256',
    plan_sha256='9ef9b4c69727d7baf324c5e3b3d93fd8c7d23a899d466f35e8c2389f05405d0d',
    note_sha256='77602cb7ef1ffba87aee01bebbd2dbe54c5e02dcf7f3b890f81200a07c5fe204',
    root_proof_sha256='100454d855cd853632a389693312cff5de9702fd5124a512a292e3ef3b16eae4',
    root_exec=63618, controller=dict(pid=2734202, start_ticks=1332948026),
    native_worker=dict(pid=2734206, start_ticks=1332948051))
NEW_SPEC = dict(directory=NEW, run_id='NeXT_full_K64_question_only_public_comment',
    note_filename='root_gpu_allocation.json', root_allocation_field='allocation_sha256',
    plan_sha256='3a1bfff5c78c47d7a0676f471eadcbff89aa3b98e9407132ffeaece3dbdce7e2',
    note_sha256='bdd8872140761e1bc42d7e2fda233a122461914aee2d6373837c3808561f7ebc',
    root_exec=60045, controller=dict(pid=2861700, start_ticks=1333486787, boot_id=BOOT),
    native_worker=dict(pid=2861704, start_ticks=1333486819, boot_id=BOOT))
LOCKED = {
    NEXT / 'guidance_question_boundary_exact_token_rows.jsonl': 'eb48187c9cc0a1879f17ce6ea9bbf80a4cb16ebeeb89b969ca421e6d26dc22a6',
    NEXT / 'guidance_question_boundary_exact_tokenization.json': 'fcab501744bb1cbda1fc46aaae288f59f98d8d49881b3be27255573b21af43b6',
    NEXT / 'llm_tokens_current.json': 'f83c3de9c6d4aef8a1d61b89a8a6d4be81097e8648b00883e6448c2a0495e2b6',
    NEXT / 'llm_tokens_author_pins.json': '4475870d4d0dff7bbc24b8243cba418e332a48bf5f13a98099a60820290abd5f',
}


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
    result = [json.loads(line) for line in Path(path).read_text().splitlines()]
    require(len(result) == N, f'Full{N} required: {path}')
    return result


def artifact(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def checked(entry, registry, expected_path=None):
    path = Path(entry['path'])
    require(expected_path is None or path == expected_path, f'Unexpected artifact path: {path}')
    require(sha(path) == entry['sha256'], f'Accepted artifact changed: {path}')
    registry[str(path)] = artifact(path)
    return path


def current_identity(pid):
    try:
        return dict(pid=pid, start_ticks=int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


def original_alive(identity):
    current = current_identity(identity['pid'])
    return current is not None and all(current[k] == v for k, v in identity.items())


def gpu_pids():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def check_launch_identity(launch, spec):
    require(launch['root_owned_exec_handle'] == spec['root_exec'] and launch['plan_sha256'] == spec['plan_sha256']
        and launch[spec['root_allocation_field']] == spec['note_sha256'], 'Launch plan/allocation identity')
    if spec['directory'] == OLD:
        require({r['role'] for r in launch['processes']} == {'controller', 'native_worker'}
            and len(launch['processes']) == 2, 'Old actual launch roles')
        for role in ['controller', 'native_worker']:
            process = next(r for r in launch['processes'] if r['role'] == role)
            require({k: process[k] for k in spec[role]} == spec[role] and process['boot_id'] == BOOT, 'Old launch identity mismatch')
    else:
        require(launch['controller'] == spec['controller'] and launch['native_worker'] == spec['native_worker'], 'New launch identity mismatch')


def terminal_gate(spec, root_sha, registry):
    """Read only terminal metadata here. Never hash/open a raw/point/obs file."""
    require(len(root_sha) == 64 and all(c in '0123456789abcdef' for c in root_sha), 'Explicit root-confirmed SHA required')
    directory, run = spec['directory'], spec['directory'] / 'execution'
    names = ['root_terminal_process_proof.json', 'plan.json', spec['note_filename'],
        'launch.claim.json', 'controller_state.json', 'root_launch_process_snapshot.json']
    required = [directory / n for n in names]
    required += [run / n for n in ['state.json', 'release.json', 'independent_verification.json', 'official.score.json']]
    require(not (directory / 'failure.json').exists() and not (run / 'failure.json').exists(), 'Failure retained; pairing is not eligible')
    require(all(p.is_file() for p in required), 'Incomplete terminal/root proof; refuse all prefix reads')
    require(sha(required[0]) == root_sha, 'Root terminal proof not explicitly accepted')
    root, plan, note, claim, controller, launch = [read(directory / n) for n in names]
    native, release, verification, official = [read(run / n) for n in ['state.json', 'release.json', 'independent_verification.json', 'official.score.json']]
    require(root['root_owned_exec_handle'] == spec['root_exec'] and root['root_owned_exec_actual_exit_code'] == 0
        and root['all_original_processes_exited'] is True, 'Root PTY not actually successful/exited')
    require(root['controller'] == claim['controller'] == controller['controller'] == spec['controller'], 'Controller identity mismatch')
    require(root['native_worker'] == controller['native_worker'] == release['actual_worker_exit']['original'] == spec['native_worker'], 'Native identity mismatch')
    check_launch_identity(launch, spec)
    require(not original_alive(spec['controller']) and not original_alive(spec['native_worker']), 'Original process still present')
    require(spec['native_worker']['pid'] not in gpu_pids(), 'Native GPU PID still present')
    psha, nsha = spec['plan_sha256'], spec['note_sha256']
    require(sha(directory / 'plan.json') == root['plan_sha256'] == claim['plan_sha256'] == controller['plan_sha256']
        == release['plan_sha256'] == verification['plan_sha256'] == psha, 'Plan SHA linkage')
    require(sha(directory / spec['note_filename']) == root[spec['root_allocation_field']]
        == claim['allocation_note_sha256'] == nsha, 'Allocation SHA linkage')
    require(plan['run_id'] == note['run_id'] == release['run_id'] == spec['run_id']
        and note['authorization'] == 'explicit_root_gpu_allocation', 'Run identity/allocation')
    require(controller['status'] == native['status'] == 'complete' and controller['native_exit_code'] == 0 and native['n'] == N, 'Native/controller incomplete')
    require(claim['launch_attempt'] == 1 and claim['automatic_retry'] is False and controller['automatic_retry'] is False, 'Unexpected retry')
    require(release['success'] is True and verification['success'] is True and release['n_rows'] == N, 'Terminal verification failed')
    require(release['actual_worker_exit']['original_exited'] is True and release['actual_worker_exit']['gpu_pid_absent'] is True, 'Missing native exit proof')
    require(all(verification[k] == N for k in ['n_rows', 'token_decoding_checks', 'input_budget_observation_checks', 'per_question_score_checks']), 'Partial5000 verification')
    require(verification['full_denominator_retained'] is True and verification['max_per_question_score_delta'] < 1e-12, 'Full denominator/pointwise disagreement')
    require(verification['CPU_only'] is True and verification['model_loaded'] is False
        and verification['paper_argv_confirmed'] is False and verification['metric_or_parameter_selection'] is False, 'Verification scope changed')
    require(sha(run / 'release.json') == root['release_sha256'] == controller['release_sha256'], 'Release SHA')
    require(sha(run / 'independent_verification.json') == root['independent_verification_sha256']
        == release['independent_verification_sha256'] == controller['independent_verification_sha256'], 'Independent report SHA')
    require(sha(run / 'official.score.json') == native['score_sha256'] == release['official_score_sha256'] == verification['official_score_sha256'], 'Official report SHA')
    require(official['n'] == official['token_decoding_checks'] == N and official['plan']['sha256'] == psha, 'Official full-run metadata')
    require(official['parameters'] == plan['parameters'], 'Official recipe mismatch')
    require(official['value'] == verification['official_accuracy_percent'] == release['official_value_percent']
        and abs(official['value'] - verification['independent_accuracy_percent']) < 1e-10, 'Accepted aggregates disagree')
    require(verification['fresh_author_cli']['exit_code'] == 0
        and verification['fresh_author_cli']['matches_exact_score_at_display_precision'] is True, 'Author CLI gate')
    if directory == NEW:
        require(verification['actual_guidance_checks'] == release['actual_guidance_checks'] == N
            and verification['input_hypothesis_only'] is True, 'Question-only actual guidance gate')
        require(verification['guidance_input_boundary_checks']['actual_guidance_checks'] == N
            and verification['guidance_input_boundary_checks']['full_LLM_input_ID_checks'] == N, 'Guidance/LLM coverage')
    for path in required:
        registry[str(path)] = artifact(path)
    return dict(spec=spec, plan=plan, verification=verification, official=official, root=root)


def freeze_sources(context, registry):
    """Called after BOTH terminal gates, before paired row access; no scorer."""
    plan, directory = context['plan'], context['spec']['directory']
    for item in plan['additional_frozen_sources'] + plan['evidence'] + plan['model_small_files']:
        checked(item, registry)
    checked(plan['worker'], registry)
    checked(plan['wait_helper'], registry)
    checked(plan['questions'], registry)
    # Annotation identity is compared via accepted plan/report hashes; GT is not opened here.
    for item in plan['source_comparisons']:
        checked(dict(path=item['author_path'], sha256=item['author_sha256']), registry)
    for name, entry in context['verification']['artifacts'].items():
        checked(entry, registry, directory / 'execution' / name)
    for entry in context['official']['artifacts'].values():
        checked(entry, registry)
    checked(context['verification']['fresh_author_cli'], registry)
    needed = {'answers.jsonl', 'runtime.jsonl', 'loaded_runtime.json', 'input_budget_observations.jsonl', 'independent_per_question_scores.jsonl'}
    require(needed <= context['verification']['artifacts'].keys(), 'Missing accepted artifact set')
    if directory == NEW:
        require('guidance_observations.jsonl' in context['verification']['artifacts'], 'Missing actual-guidance artifact')
        for path, guard in plan['strict_model_stat_guards'].items():
            p = Path(path); s = p.stat()
            actual = dict(resolved=str(p.resolve()), bytes=s.st_size, device=s.st_dev, inode=s.st_ino,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
            require(actual == {k: v for k, v in guard.items() if k != 'prior_verified_sha256'}, 'Strict model stat changed')
    for name in needed | ({'guidance_observations.jsonl'} if directory == NEW else set()):
        context[name] = read(directory / 'execution' / name) if name.endswith('.json') else rows(directory / 'execution' / name)


def units(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid accepted pointwise value')
    result = round(value * 10)
    require(0 <= result <= 10 and abs(value - result / 10) < 1e-12, 'Accepted score is not integer tenths')
    return result


def tensor_list_sha(values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def validate_observation(observation, point, runtime, reference, position):
    require(observation['position'] == position and observation['input_shape'] == [1, reference['n_tokens']]
        and observation['input_dtype'] == 'torch.int64' and observation['input_ids_sha256'] == reference['token_ids_sha256'], 'LLM IDs mismatch')
    quotas = observation['allocated_quotas']
    require(observation['crop_count'] == 5 and observation['initial_quotas'] == [64] * 5
        and len(quotas) == 5 and all(type(q) is int and 1 <= q <= 576 for q in quotas), 'Physical crop quota gate')
    require(quotas == observation['greedy_quotas'] == observation['effective_mask_counts'] == point['native_quotas'], 'Quota/prefix/point disagreement')
    require(observation['effective_mask_shape'] == [5, 576] and len(observation['effective_mask_indices'])
        == len(observation['greedy_effective_prefixes']) == 5, 'Mask shape')
    for q, mask, prefix in zip(quotas, observation['effective_mask_indices'], observation['greedy_effective_prefixes']):
        require(len(mask) == len(set(mask)) == len(prefix) == len(set(prefix)) == q
            and mask == sorted(prefix) and all(type(i) is int and 0 <= i < 576 for i in prefix), 'Mask/effective prefix membership')
    require(sum(quotas) == observation['returned_retained_tokens'] == runtime['actual_visual_tokens_retained'], 'Actual retained total')
    require(observation['importance_shape'] == [5, 576] and observation['importance_finite'] is True
        and observation['projected_feature_shape'] == [5, 576, 4096] and observation['projected_feature_dtype'] == 'torch.float16'
        and observation['post_generation_tensor_reads_only'] is True, 'Observation scope/dtype')


def summarize(paired):
    counts, transitions, old_hist, new_hist = Counter(), Counter(), Counter(), Counter()
    gain = loss = 0
    for row in paired:
        difference = row['question_only_score_tenths'] - row['full_guidance_score_tenths']
        counts['score_improved' if difference > 0 else 'score_declined' if difference < 0 else 'score_unchanged'] += 1
        for key in ['raw_answer_changed', 'output_token_IDs_changed', 'quota_vector_changed', 'retained_total_changed', 'final_mask_changed', 'effective_ordered_prefix_changed']:
            counts[key] += int(row[key])
        gain += max(0, difference); loss += min(0, difference)
        transitions[(row['full_guidance_score_tenths'], row['question_only_score_tenths'])] += 1
        old_hist[sum(row['full_guidance_quotas'])] += 1
        new_hist[sum(row['question_only_quotas'])] += 1
    n = len(paired)
    old_sum = sum(r['full_guidance_score_tenths'] for r in paired)
    new_sum = sum(r['question_only_score_tenths'] for r in paired)
    require(n > 0 and gain + loss == new_sum - old_sum
        and sum(counts[k] for k in ['score_improved', 'score_declined', 'score_unchanged']) == n, 'Aggregation mismatch')
    # Tenths / n *100 = total units *10/n, exact Decimal, no new scoring.
    percent = lambda total: str(Decimal(total) * 10 / Decimal(n))
    names = ['score_improved', 'score_declined', 'score_unchanged', 'raw_answer_changed', 'output_token_IDs_changed',
        'quota_vector_changed', 'retained_total_changed', 'final_mask_changed', 'effective_ordered_prefix_changed']
    return dict(n=n, counts={key: counts[key] for key in names}, raw_answer_unchanged=n - counts['raw_answer_changed'],
        output_token_IDs_unchanged=n - counts['output_token_IDs_changed'],
        full_guidance_percent=percent(old_sum), question_only_percent=percent(new_sum),
        question_only_minus_full_guidance_pp=percent(new_sum - old_sum),
        full_guidance_score_tenths_sum=old_sum, question_only_score_tenths_sum=new_sum,
        gained_tenths=gain, lost_tenths_signed=loss, net_tenths=gain + loss,
        full_guidance_retained_histogram=dict(sorted(old_hist.items())), question_only_retained_histogram=dict(sorted(new_hist.items())),
        transitions=[dict(full_guidance_tenths=a, question_only_tenths=b, n=count) for (a, b), count in sorted(transitions.items())])


def main(expected_root_sha):
    require(not (OUT / 'comparison_report.json').exists() and not (OUT / 'paired_rows.jsonl').exists(), 'Outputs already exist; do not overwrite')
    registry = {}
    # This order deliberately cannot touch either run's raw/points/observations
    # until the new run and its root-owned PTY have also passed every gate.
    old = terminal_gate(OLD_SPEC, OLD_SPEC['root_proof_sha256'], registry)
    new = terminal_gate(NEW_SPEC, expected_root_sha, registry)
    require(old['plan']['parameters'] == new['plan']['parameters'], 'Single-guidance recipe mismatch')
    for key in ['author_commit', 'waits', 'model_path', 'model_payloads', 'model_small_files', 'source_comparisons',
        'questions', 'annotation', 'image_folder', 'image_count', 'clip_identity']:
        require(old['plan'][key] == new['plan'][key], f'Common source/model/input gate differs: {key}')
    for path, digest in LOCKED.items():
        checked(dict(path=str(path), sha256=digest), registry)
    freeze_sources(old, registry); freeze_sources(new, registry)
    for name in ['entry_file', 'entry_sha256', 'gpu', 'torch_default_dtype', 'package_versions', 'backend_flags', 'loaded', 'wait_patch', 'first_generate_input']:
        require(old['loaded_runtime.json'][name] == new['loaded_runtime.json'][name], f'Actual runtime differs: {name}')
    for context in [old, new]:
        require(context['loaded_runtime.json']['plan_sha256'] == context['spec']['plan_sha256']
            and context['loaded_runtime.json']['worker_sha256'] == context['plan']['worker']['sha256'], 'Loaded plan/worker linkage')
    old_args, new_args = [dict(c['loaded_runtime.json']['args']) for c in (old, new)]
    old_args.pop('answers_file'); new_args.pop('answers_file')
    require(old_args == new_args, 'Actual native args differ beyond isolated output')
    transform = new['loaded_runtime.json']['guidance_entry_transform']
    static = new['plan']['worker_adapter']['guidance_entry_transform']
    require({k: transform[k] for k in static} == static and transform['executed_eval_model_code_sha256']
        == static['expected_eval_model_code_sha256'] and static['inverse_restores_original'] is True, 'Actual guidance entry code gate')
    references = rows(NEXT / 'guidance_question_boundary_exact_token_rows.jsonl')
    questions = rows(old['plan']['questions']['path'])
    llm = read(NEXT / 'llm_tokens_current.json')['rows']
    require(llm == read(NEXT / 'llm_tokens_author_pins.json')['rows'] and len(llm) == N, 'Existing full LLM references disagree')
    paired = []
    for i, (question, reference, llm_reference) in enumerate(zip(questions, references, llm)):
        qid = str(question['question_id']); prompt_sha = hashlib.sha256(question['text'].encode()).hexdigest()
        require(reference['position'] == llm_reference['position'] == i and str(reference['question_id']) == str(llm_reference['question_id']) == qid
            and reference['unchanged_full_LLM_prompt_sha256'] == prompt_sha
            and reference['full_guidance_matches_both_existing_reference_stacks'] is True, 'Exact canonical/reference alignment')
        points = []
        for context in [old, new]:
            answer, point, runtime, observation = [context[name][i] for name in ['answers.jsonl', 'independent_per_question_scores.jsonl', 'runtime.jsonl', 'input_budget_observations.jsonl']]
            require(str(answer['question_id']) == str(point['question_id']) == str(runtime['question_id']) == qid
                and point['position'] == runtime['position'] == i and point['prompt_sha256'] == runtime['prompt_sha256'] == prompt_sha
                and runtime['image'] == question['image'] and answer['prompt'] == question['text'] and isinstance(answer['text'], str), 'Full ordered ID/prompt/image mismatch')
            require(runtime['image_tensor_shape'] == [1, 5, 3, 336, 336] and runtime['images_dtype'] == 'torch.float16', 'Actual image tensor shape/dtype')
            validate_observation(observation, point, runtime, llm_reference, i)
            require(units(point['official_score']) == units(point['independent_score']), 'Accepted pointwise author/independent mismatch')
            points.append((answer, point, runtime, observation))
        guidance = new['guidance_observations.jsonl'][i]
        require(guidance['position'] == i and str(guidance['question_id']) == qid and guidance['prompt_sha256'] == prompt_sha
            and guidance['guidance_sha256'] == reference['question_only_guidance']['guidance_sha256']
            and guidance['post_generation_observation'] is True, 'Actual question-only guidance mismatch')
        a, b = points
        old_masks, new_masks = a[3]['effective_mask_indices'], b[3]['effective_mask_indices']
        old_prefix, new_prefix = a[3]['greedy_effective_prefixes'], b[3]['greedy_effective_prefixes']
        removed = [len(set(x) - set(y)) for x, y in zip(old_masks, new_masks)]
        added = [len(set(y) - set(x)) for x, y in zip(old_masks, new_masks)]
        paired.append(dict(position=i, question_id=question['question_id'], image=question['image'], prompt_sha256=prompt_sha,
            full_LLM_input_IDs_sha256=llm_reference['token_ids_sha256'], full_guidance_token_length=reference['full_guidance']['ids_length'],
            fixed_stratum='full_guidance_gt77' if reference['full_guidance']['ids_length'] > 77 else 'full_guidance_le77',
            full_guidance_sha256=reference['full_guidance']['guidance_sha256'], actual_question_only_guidance_sha256=guidance['guidance_sha256'],
            full_guidance_answer=a[0]['text'], question_only_answer=b[0]['text'], raw_answer_changed=a[0]['text'] != b[0]['text'],
            full_guidance_score_tenths=units(a[1]['independent_score']), question_only_score_tenths=units(b[1]['independent_score']),
            full_guidance_quotas=a[3]['allocated_quotas'], question_only_quotas=b[3]['allocated_quotas'],
            quota_vector_changed=a[3]['allocated_quotas'] != b[3]['allocated_quotas'],
            retained_total_changed=a[3]['returned_retained_tokens'] != b[3]['returned_retained_tokens'],
            final_mask_changed=old_masks != new_masks, per_crop_final_mask_changed=[x != y for x, y in zip(old_masks, new_masks)],
            effective_ordered_prefix_changed=old_prefix != new_prefix,
            per_crop_removed_tokens=removed, per_crop_added_tokens=added,
            full_guidance_masks_json_sha256=tensor_list_sha(old_masks), question_only_masks_json_sha256=tensor_list_sha(new_masks),
            full_guidance_effective_prefix_json_sha256=tensor_list_sha(old_prefix), question_only_effective_prefix_json_sha256=tensor_list_sha(new_prefix),
            output_token_IDs_changed=a[2]['output_token_ids'] != b[2]['output_token_ids'],
            full_guidance_output_IDs_json_sha256=tensor_list_sha(a[2]['output_token_ids']), question_only_output_IDs_json_sha256=tensor_list_sha(b[2]['output_token_ids'])))
    require(len({(r['question_id'], r['prompt_sha256']) for r in paired}) == N, 'Composite keys not unique')
    all_summary = summarize(paired)
    strata = {name: summarize([r for r in paired if r['fixed_stratum'] == name]) for name in ['full_guidance_gt77', 'full_guidance_le77']}
    require(strata['full_guidance_gt77']['n'] == 881 and sum(s['n'] for s in strata.values()) == N, 'Frozen pre-outcome strata coverage')
    for key in ['full_guidance_score_tenths_sum', 'question_only_score_tenths_sum', 'gained_tenths', 'lost_tenths_signed', 'net_tenths']:
        require(sum(s[key] for s in strata.values()) == all_summary[key], 'Strata do not reconstruct complete result')
    for context, key in [(old, 'full_guidance_percent'), (new, 'question_only_percent')]:
        require(Decimal(all_summary[key]) == Decimal(context['verification']['independent_exact_percent']), 'Consumed full score aggregate changed')
    registry[str(Path(__file__).resolve())] = artifact(Path(__file__).resolve())
    require(all(sha(item['path']) == item['sha256'] for item in registry.values()), 'Consumed input changed during pairing')
    require('torch' not in sys.modules and not any(name.startswith(('llava', 'transformers')) for name in sys.modules), 'Unexpected model/scorer import')
    path = OUT / 'paired_rows.jsonl'
    with path.open('x') as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + '\n' for row in paired)
    report = dict(success=True, checked_utc=datetime.now(timezone.utc).isoformat(), n_paired=N,
        direction='question-only guidance minus accepted full-guidance K64 control', summary=all_summary,
        fixed_pre_outcome_guidance_length_strata=strata,
        exact_ordered_ID_prompt_image_LLM_input_ID_pairing=True, actual_recorded_runtime_flags_equal=True,
        new_root_terminal_proof_sha256=expected_root_sha, original_root_terminal_proof_sha256=OLD_SPEC['root_proof_sha256'],
        runtime_equality_keys=['package_versions', 'backend_flags', 'loaded', 'wait_patch', 'native_args_except_answers_file', 'first_generate_input'],
        input_registry_before_after_unchanged=list(registry.values()), paired_records=artifact(path),
        scorer_invoked=False, answer_normalization_invoked=False, GT_parsed=False,
        model_loaded=False, GPU_started=False, new_metric=False, score_direction_gate=False,
        limitations=[
            'Only sums accepted full5000 independent pointwise scores in integer tenths; no answer normalization, GT evaluation or new scoring.',
            'Question-only guidance is supported by an inactive public comment, not a uniquely identified paper argv, confirmed repair or evidence of paper-level causation.',
            'Both full runs retain all5000 inputs and positive/negative changes. Strata use frozen full-guidance token lengths, never outcomes.',
            'Mask comparisons use actual per-crop effective sets and retained ordered prefixes; discarded maxT suffixes are absent. No unrecorded per-question projected-feature or pixel-byte equality is inferred.',
            'Package/backend/source/model guards and input-ID observations match the recorded controls; this does not certify unrecorded low-level scheduling or efficiency.'])
    with (OUT / 'comparison_report.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(dict(success=True, n_paired=N, summary=all_summary, strata=strata)))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Prepared only; no run artifacts read. After full terminal acceptance, pass the exact new root-proof SHA256.')
    else:
        require(len(sys.argv) == 2, 'Pass only the root-confirmed new terminal-proof SHA256')
        main(sys.argv[1])
