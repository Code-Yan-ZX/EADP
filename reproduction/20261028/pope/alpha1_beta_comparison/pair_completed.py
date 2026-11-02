"""Pair the two accepted full POPE runs, only after the root terminal gate.

No GT is opened, no score is computed, and no GPU/model is used.  Official
yes/no parsing is used solely to summarize changes between already accepted
raw predictions; category metrics come from previously accepted score files.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

POPE = Path(__file__).resolve().parents[1]
OLD = POPE / 'run02_source_wait_only_k32'
NEW = POPE / 'full_alpha1_beta2_single_control_run02'
OUT = Path(__file__).resolve().parent / 'comparison.json'
QUESTION = POPE / 'inputs/llava_pope_test.jsonl'
PLAN_SHA = '4704a777360b042b4d1e63dd9be4458f50d37a0234bfe09d4169e675e268ea5b'
OLD_PROOF_SHA = '6590416b99cf7b99e78ee7176ab192b37fb86ce0cc4b93532c9c301eeaa85c5a'
OLD_SUMMARY_SHA = '22bdd3a03a19fb7f94e4b5ae24564df1d40a831be36e11140c2cd62a00db3f31'
OLD_FINAL_SHA = '15f84c054ab366216cc5cfc2271fd3a423d6c698366883b22a828699daa9b82f'
OLD_OFFICIAL_SHA = '4f654bf8264af909ddab4159fe456435340645899ac9acb08db6adf568f8f736'
QUESTION_SHA = '1d3b41f0dced74b4f361c659787ef673497cfac54a6be7bde00eb405bab03c70'
CATEGORIES = {'random': 2910, 'popular': 3000, 'adversarial': 3000}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    raw = Path(path).read_bytes()
    assert raw.endswith(b'\n')
    return [json.loads(line) for line in raw.splitlines()]


def official_yes(text):
    # Literal semantics of the released official POPE evaluator.
    assert isinstance(text, str)
    words = text.split('.')[0].replace(',', '').split(' ')
    return not any(word in words for word in ('No', 'not', 'no'))


def official_categories(text):
    # Read previously produced official stdout; do not invoke the evaluator.
    pattern = (r'Category: (random|popular|adversarial), # samples: (\d+)\s+'
               r'TP\s+FP\s+TN\s+FN\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)'
               r'.*?F1 score:\s*([0-9.eE+-]+)')
    matches = re.findall(pattern, text, re.S)
    assert len(matches) == 3 and len({m[0] for m in matches}) == 3
    return {name: dict(n=int(n), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
                       f1_percent=float(f1)*100)
            for name,n,tp,fp,tn,fn,f1 in matches}


def accepted_categories(independent, official):
    assert set(independent) == set(official) == set(CATEGORIES)
    for name,n in CATEGORIES.items():
        a,b = independent[name],official[name]
        assert a['n'] == b['n'] == n
        assert all(a[k] == b[k] for k in ('TP','FP','TN','FN'))
        assert abs(a['f1_percent']-b['f1_percent']) < 1e-12
    return {name:{k:independent[name][k] for k in ('n','TP','FP','TN','FN','f1_percent')}
            for name in CATEGORIES}


def paired_summary(questions, baseline, variant, observations, category_sizes):
    assert len(questions) == len(baseline) == len(variant) == len(observations) == sum(category_sizes.values())
    assert len({q['question_id'] for q in questions}) == len(questions)
    summary = {name:Counter() for name in category_sizes}
    for i,(q,a,b,obs) in enumerate(zip(questions,baseline,variant,observations)):
        assert q['question_id'] == a['question_id'] == b['question_id'] == obs['question_id']
        assert q['text'] == a['prompt'] == b['prompt']
        assert q['image'] == obs['image'] and obs['position'] == i
        assert b['text'] == obs['text'] and q['category'] in category_sizes
        old,new = official_yes(a['text']), official_yes(b['text'])
        group = summary[q['category']]
        group['n'] += 1
        group['raw_text_changed'] += a['text'] != b['text']
        group['yes_no_changed'] += old != new
        group[('yes' if old else 'no') + '_to_' + ('yes' if new else 'no')] += 1
    for name,n in category_sizes.items():
        c=summary[name]
        assert c['n'] == n
        assert sum(c[k] for k in ('yes_to_yes','yes_to_no','no_to_yes','no_to_no')) == n
        assert c['yes_no_changed'] == c['yes_to_no']+c['no_to_yes']
    keys=('n','raw_text_changed','yes_no_changed','yes_to_yes','yes_to_no','no_to_yes','no_to_no')
    by_category={name:{k:summary[name][k] for k in keys} for name in category_sizes}
    total={k:sum(by_category[name][k] for name in by_category) for k in keys}
    return dict(total=total,by_category=by_category)


def terminal_gate(root_proof_sha):
    # No prediction, observation, label or score file is opened before this
    # explicit actual-root-exit gate and completed-release chain succeeds.
    root_path=NEW/'root_terminal_process_proof.json'
    assert sha(root_path) == root_proof_sha
    root=read(root_path)
    assert root['root_owned_exec_handle'] == 23602 and root['root_owned_exec_actual_exit_code'] == 0
    assert root['all_original_processes_exited'] is True and root['plan_sha256'] == PLAN_SHA == sha(NEW/'plan.json')
    assert root['allocation_note_sha256'] == sha(NEW/'root_gpu_allocation.json')
    assert root['launch_snapshot_sha256'] == sha(NEW/'root_launch_process_snapshot.json')
    assert root['controller_state_sha256'] == sha(NEW/'state.json')
    assert root['release_sha256'] == sha(NEW/'execution/release.json')
    assert root['independent_verification_sha256'] == sha(NEW/'execution/independent_verification.json')
    state=read(NEW/'state.json'); release=read(NEW/'execution/release.json'); report=read(NEW/'execution/independent_verification.json')
    assert state['status']=='complete' and state['native_exit_code']==state['CPU_verifier_exit_code']==0
    assert state['release_sha256']==root['release_sha256'] and state['plan_sha256']==PLAN_SHA
    assert state['controller']==root['controller']==release['controller']
    assert state['native_worker']==root['native_worker']==release['native_worker']
    for role in ('controller','native_worker'):
        checks=[p for p in root['process_checks'] if p['role']==role]
        assert len(checks)==1 and checks[0]['original']==root[role] and checks[0]['original_exited'] is True
    launch=read(NEW/'root_launch_process_snapshot.json')
    for role in ('controller','native_worker'):
        matches=[p for p in launch['processes'] if p['role']==role]
        assert len(matches)==1 and {k:matches[0][k] for k in ('pid','start_ticks','boot_id')}==root[role]
    assert launch['root_owned_exec_handle']==23602 and launch['plan_sha256']==PLAN_SHA
    assert release['success'] and release['n_predictions']==8910 and release['actual_native_exited']
    assert release['native_exit_code']==0 and release['all_generation_completed_before_scoring']
    assert release['plan_sha256']==PLAN_SHA and release['allocation_note_sha256']==root['allocation_note_sha256']
    assert release['independent_verification_sha256']==root['independent_verification_sha256']
    assert report['success'] and report['complete'] and report['n_predictions']==8910
    assert all(report[k]==8910 for k in ('actual_guidance_checks','input_ID_checks','output_redecode_checks','budget_mask_prefix_checks'))
    assert report['panel_beta2_anchor_checks']==16 and report['official_scorer_exit_code']==0
    assert report['full_denominator_retained'] and report['plan_sha256']==PLAN_SHA
    assert report['actual_native_exit']['original']==root['native_worker'] and report['actual_native_exit']['original_exited']
    assert sha(OLD/'completion_and_release_proof.json')==OLD_PROOF_SHA
    old=read(OLD/'completion_and_release_proof.json')
    assert old['status']=='all_completion_and_exit_gates_passed' and old['prediction_count']==8910
    assert old['generation_exit_code']==old['official_score_exit_code']==0 and old['same_worker_still_alive'] is False
    assert old['independent_and_official_macro_match'] and old['all_source_sha_match']
    return root,release,report,old


def check_old_metadata(old):
    """Only accepted completed old metadata/log; no prediction or GT access."""
    assert sha(OLD/'completion_and_release_proof.json') == OLD_PROOF_SHA
    assert old == read(OLD/'completion_and_release_proof.json')
    assert old['status']=='all_completion_and_exit_gates_passed' and old['prediction_count']==8910
    assert old['generation_exit_code']==old['official_score_exit_code']==0 and old['same_worker_still_alive'] is False
    assert old['independent_and_official_macro_match'] and old['all_source_sha_match']
    assert sha(OLD/'independently_verified_full_summary.json')==OLD_SUMMARY_SHA
    assert old['full_result_summary']['sha256']==OLD_FINAL_SHA==sha(OLD/'full_result_summary.json')
    old_summary=read(OLD/'independently_verified_full_summary.json')
    old_final=read(OLD/'full_result_summary.json')
    assert old_summary['run_id']==old_final['run_id']=='run02_source_wait_only_k32'
    assert old_summary['n_predictions']==old_final['n_predictions']==8910
    assert old_final['official_scorer_exit_code']==0
    assert old_final['all_8910_question_ids_order_verified'] and old_final['gpu_worker_exit_verified']
    assert all(old_final['source_sha_unchanged_checks'].values())
    assert abs(old_summary['official_macro_f1_percent']-old_final['official_macro_f1_percent'])<1e-12
    assert abs(old_final['official_macro_f1_percent']-old_final['independent_macro_f1_percent'])<1e-12
    assert old_final['categories']==old_summary['categories']
    assert old_summary['predictions']==old_final['predictions']
    assert old_final['predictions']['sha256']==old_final['predictions_sha256']==old['predictions_sha256']
    assert old_summary['source_plan']==old_final['source_plan']
    assert old_final['source_plan']['sha256']==sha(OLD/'run_plan.json')
    old_plan=read(OLD/'run_plan.json')
    assert old_plan['input_audit']['sha256']=='32e6fccb886a6ab9b0db6569e785ca01fc4f71eab4db9896731d5a37055af326'
    assert old_summary['official_scorer_stdout']==old_final['official_scorer_stdout']
    assert old_final['official_scorer_stdout']['sha256']==OLD_OFFICIAL_SHA==sha(OLD/'official_score.log')
    categories=accepted_categories(old_final['categories'],official_categories((OLD/'official_score.log').read_text()))
    return old_final,categories


def main(proof_sha):
    assert re.fullmatch('[0-9a-f]{64}',proof_sha) and not OUT.exists()
    root,release,report,old=terminal_gate(proof_sha)
    # Both sides passed terminal gate; now inspect already accepted score files.
    old_final,old_categories=check_old_metadata(old)
    metric_info=report['qualified_metric_verification']; log_info=report['official_score_log']
    assert Path(metric_info['path'])==NEW/'execution/official_and_reference_score/full_prediction_verification.json'
    assert Path(log_info['path'])==NEW/'execution/official_and_reference_score/official_score.log'
    assert sha(metric_info['path'])==metric_info['sha256'] and sha(log_info['path'])==log_info['sha256']
    metric=read(metric_info['path'])
    assert metric['complete'] and metric['n_predictions']==8910 and metric['official_scorer_exit_code']==0
    assert metric['all_question_ids_order_verified'] and metric['all_raw_prompts_verified'] and metric['all_pinned_image_question_positions_verified']
    assert metric['score_log_sha256']==log_info['sha256']
    assert abs(metric['official_macro_f1_percent']-metric['independent_macro_f1_percent'])<1e-12
    assert all(abs(metric[k]-report[k])<1e-12 and abs(metric[k]-release[k])<1e-12
               for k in ('official_macro_f1_percent','independent_macro_f1_percent'))
    assert metric['categories']==report['categories'] and metric['predictions_sha256']==report['predictions_sha256']==release['predictions_sha256']
    new_categories=accepted_categories(metric['categories'],official_categories(Path(log_info['path']).read_text()))
    assert metric['canonical_questions_sha256']==QUESTION_SHA==sha(QUESTION)
    assert old['predictions_sha256']==old_final['predictions_sha256']==sha(OLD/'predictions.jsonl')
    assert sha(NEW/'execution/predictions.jsonl')==release['predictions_sha256']
    assert sha(NEW/'execution/runtime_observations.jsonl')==release['runtime_observations_sha256']
    questions=jsonl(QUESTION); baseline=jsonl(OLD/'predictions.jsonl')
    variant=jsonl(NEW/'execution/predictions.jsonl'); observations=jsonl(NEW/'execution/runtime_observations.jsonl')
    comparison=paired_summary(questions,baseline,variant,observations,CATEGORIES)
    result=dict(success=True,full_8910_paired=True,comparison=comparison,
        baseline=dict(alpha=1,beta=1,macro_f1_percent=old_final['independent_macro_f1_percent'],categories=old_categories,
                      predictions_sha256=old_final['predictions_sha256'],terminal_proof_sha256=OLD_PROOF_SHA),
        variant=dict(alpha=1,beta=2,macro_f1_percent=metric['independent_macro_f1_percent'],categories=new_categories,
                     predictions_sha256=metric['predictions_sha256'],terminal_proof_sha256=proof_sha),
        variant_minus_baseline_macro_f1_percentage_points=metric['independent_macro_f1_percent']-old_final['independent_macro_f1_percent'],
        category_f1_changes_percentage_points={k:new_categories[k]['f1_percent']-old_categories[k]['f1_percent'] for k in CATEGORIES},
        official_yes_no_rule='first sentence; remove comma; split on space; No/not/no means no; otherwise yes',
        input_identity='Both full prediction orders and prompts match the same qualified canonical question IDs; new recorded image matches that canonical image. Old raw predictions contain no image field.',
        metrics_are_existing_accepted_official_and_independent_records=True,GT_read=False,scoring_rerun=False,
        limitation='Old full baseline runtime was not fully matched field by field; full paired results alone do not establish beta causality. Verified beta propagation is limited to the accepted fixed16 native panel.')
    with OUT.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'success':True,'output':str(OUT),'n':comparison['total']['n']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-proof-sha256',required=True)
    main(parser.parse_args().root_proof_sha256)
