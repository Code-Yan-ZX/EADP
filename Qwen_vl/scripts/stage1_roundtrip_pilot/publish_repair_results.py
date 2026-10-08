"""Publish only complete repaired-protocol pairs, without touching the live index.

--prepare/--validate are CPU-only and never commit/push. --publish uses a
detached worktree and a normal fast-forward push to the fixed origin branch.
--watch retries every 30 seconds; no generation process or source is changed.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).parent
OUT = ROOT/'Qwen_vl/outputs/audit_followup_20261008'
BATCH = OUT/'rerun_batch'
LOCAL = BATCH/'group_publish'
DEST = Path('Qwen_vl/outputs/anchor_completion_validation/records_20261003/audit_followup_20261008/published_results')
BRANCH = 'codex/anchor-completion-validation'
REF = 'refs/heads/' + BRANCH
STATE = LOCAL/'publish_state.json'
MAX_BLOB = 89 * 1024 * 1024
EXCLUDED = {'Qwen_vl/scripts/stage1_roundtrip_pilot/llava_round2_driver.sh'}
TRACKED = [
 'Qwen_vl/outputs/anchor_completion_validation/records_20261003/STATUS_20261003.md',
 'docs/project_handoff.md',
 *['Qwen_vl/scripts/stage1_roundtrip_pilot/'+name for name in (
 'gen_sqa_cqm_i.py','gqa_score.py','llava_eval_arm_model_vqa.py','llava_eval_arm_science.py',
 'llava_lane_a_v15.sh','llava_lane_b_next.sh','llava_next2_driver.sh','llava_next_driver.sh',
 'llava_round2b_driver.sh','next_budget_manifest.py','next_sqa_official_and_perf2_driver.sh',
 'sqa_arms_official_driver.sh','sqa_full_official_driver.sh','vizwiz_val_score.py')]]
NEW = ['docs/evaluation_reproduction_audit_20261008.md',
 *['Qwen_vl/scripts/stage1_roundtrip_pilot/'+name for name in (
 'build_repair_run_summary.py','check_sqa_reuse.py','mme_canonical_score.py','mme_convert_canonical.py',
 'monitor_repair_hourly.py','official_score.py','run_hourly_repair_agent.py','run_repair_queue.py',
 'run_sqa_shipped_control.py','run_textvqa_official.py','textvqa_runtime_launcher.py',
 'vizwiz_official_normalization.py','publish_repair_results.py')]]
ALLOWLIST = TRACKED + NEW


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def command(argv, cwd=ROOT):
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {argv[0]}: {result.stderr[-3000:]}')
    return result.stdout.strip()


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS/(name+'.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def checked_source_maps(protocol):
    verified = {}
    for name in ('source_sha256','source_and_input_sha256'):
        mapping = protocol.get(name)
        if not isinstance(mapping, dict):
            continue
        for filename, expected in mapping.items():
            path = Path(filename)
            if not path.is_absolute():
                path = (ROOT/'LLaVA'/filename).resolve()
            if sha(path) != expected:
                raise ValueError('Frozen source/input SHA mismatch: '+str(path))
            verified[str(path)] = expected
    return verified


def add_files(files, paths):
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        if path.suffix.lower() not in ('.py','.sh','.json','.jsonl','.txt','.csv','.md','.log'):
            raise ValueError('Disallowed artifact type: '+str(path))
        if any(token in str(path) for token in ('agent.events', '/.codex/', '/hourly_monitor/')):
            raise ValueError('Private monitor/account artifact excluded: '+str(path))
        try:
            member = str(path.relative_to(ROOT))
        except ValueError:
            # Official reference source outside this checkout is stored by
            # content hash; no absolute symlink or account file is archived.
            member = 'external_reference/'+sha(path)+'/'+path.name
        files[member] = path.read_bytes()


def validate_job(job, summary_module):
    output = Path(job['output'])
    rows = [json.loads(line) for line in output.open()]
    expected = 2017 if job['task']=='sqa' else 5000
    if len(rows) != expected or any(str(r.get('text','')).strip() in ('','FAILED') for r in rows):
        raise ValueError('Incomplete/FAILED prediction: '+str(output))
    files = {}
    add_files(files, [output,output.with_suffix('.log')])
    protocol_path = Path(str(output)+'.protocol.json')
    protocol = read(protocol_path) if protocol_path.exists() else {}
    verified = checked_source_maps(protocol)
    add_files(files, verified)
    add_files(files, [protocol_path,Path(job['question_file'])])
    if job['task']=='textvqa':
        exact, dependencies = summary_module.text_score(job, rows)
        queue = module('run_repair_queue')
        queue.check_textvqa_score(job['score_log'],5000)
        for key, value in {'max_new_tokens':128,'alpha':.5,'beta':2.,'temperature':0,'conv_mode':'vicuna_v1'}.items():
            if protocol.get(key) != value:
                raise ValueError('TextVQA frozen parameter mismatch: '+key)
        arm, _, budget = summary_module.arm_fields(job)
        if protocol.get('visual_token_num') != budget:
            raise ValueError('TextVQA protocol budget mismatch')
        if protocol.get('question_file_sha256') != sha(job['question_file']):
            raise ValueError('TextVQA input hash mismatch')
        if protocol.get('wrapper_sha256') != sha(SCRIPTS/'llava_eval_arm_model_vqa.py'):
            raise ValueError('TextVQA wrapper hash mismatch')
        trace = Path(str(output)+'.runtime.jsonl')
        if trace.exists():
            traced_job = dict(job, generate_command=['python','wrapper','--visual_token_num',str(budget)])
            queue.check_predictions(traced_job)
            add_files(files,[trace])
        elif not (job['model']=='next' and arm in ('EGATHER','LRMAIN025')):
            raise ValueError('Missing required TextVQA runtime trace')
        add_files(files,[job['score_log'],*dependencies,SCRIPTS/'llava_eval_arm_model_vqa.py'])
        score = dict(accuracy=exact,n=5000,metric='official TextVQA accuracy',runtime_trace_available=trace.exists())
    else:
        helper = module('run_sqa_shipped_control')
        helper.check_science_identity(); helper.check_questions()
        questions = read(job['question_file'])
        expected_prompts = {}
        for question in questions:
            prompt = '<image>\n'+question['conversations'][0]['value'].replace('<image>','').strip()
            expected_prompts[str(question['id'])] = prompt+"\nAnswer with the option's letter from the given choices directly."
        ids = [str(r['question_id']) for r in rows]
        if len(set(ids)) != 2017 or set(ids) != set(expected_prompts) or any(r['prompt']!=expected_prompts[str(r['question_id'])] for r in rows):
            raise ValueError('SQA input/prompt identity failure')
        result_path = Path(job.get('result',str(output.with_suffix('.result.json'))))
        previous = read(result_path)
        scorer = ROOT/'LLaVA/llava/eval/eval_science_qa.py'
        base = OUT/'sqa_image_only_base'
        with tempfile.TemporaryDirectory(prefix='eadp-score-') as temporary:
            fresh = Path(temporary)/'result.json'
            command([sys.executable,str(scorer),'--base-dir',str(base),'--result-file',str(output),
                     '--output-file',str(Path(temporary)/'output.json'),'--output-result',str(fresh)])
            current = read(fresh)
        if current != previous or current['count']!=2017:
            raise ValueError('Official SQA rescore differs from archived score')
        if protocol:
            if protocol.get('fixed_subset_sha256') != sha(job['question_file']) or protocol.get('official_evaluator_sha256')!=sha(scorer):
                raise ValueError('SQA protocol input/scorer hash mismatch')
            if isinstance(protocol.get('source_sha256'), str) and protocol['source_sha256'] != sha(helper.SOURCE):
                raise ValueError('SQA original question source hash mismatch')
            if protocol.get('frozen') != dict(alpha=.5,beta=2.,lambda_completion=.25,
                                             conv_mode='vicuna_v1',temperature=0,
                                             max_new_tokens=1024,single_pred_prompt=True):
                raise ValueError('SQA frozen generation parameters changed')
            budget = summary_module.arm_fields(job)[2]
            if any(r.get('metadata',{}).get('visual_token_budget_parameter')!=budget or not isinstance(r.get('metadata',{}).get('actual_visual_tokens_retained'),int) or r['metadata']['actual_visual_tokens_retained']<=0 for r in rows):
                raise ValueError('SQA runtime budget failure')
        elif job.get('arm')!='FULL':
            raise ValueError('Missing required SQA protocol')
        add_files(files,[result_path,output.with_suffix('.output.json'),output.with_suffix('.score.txt'),
                         scorer,base/'pid_splits.json',base/'problems.json',helper.SOURCE,
                         OUT/'sqa_official_zip_identity.json',OUT/'sqa_input_comparison.json'])
        score = dict(accuracy=current['acc'],n=2017,metric='ScienceQA IMG',runtime_metadata_available=bool(protocol))
    score.update(output=str(output),prediction_sha256=sha(output),verified_source_sha256=verified)
    return files, score


def validate_pope():
    directory = OUT/'rerun_fast'
    report_path = directory/'pope_v15_EADP32_streamwait.fair_comparison.json'
    report = read(report_path)
    if report.get('success') is not True or report.get('n')!=8910:
        raise ValueError('Incomplete POPE fair comparison')
    files, scores = {}, []
    from official_score import score as official_score
    qf = ROOT/'LLaVA/playground/data/eval/pope/llava_pope_test.jsonl'
    questions = [json.loads(line) for line in qf.open()]
    qby = {str(r['question_id']):r for r in questions}
    for method,stem in [('AnchorZip32','pope_v15_AZ32_streamwait'),('EADP32','pope_v15_EADP32_streamwait')]:
        output = directory/(stem+'.jsonl');protocol_path=directory/(stem+'.protocol.json')
        rows=[json.loads(line) for line in output.open()];ids=[str(r['question_id']) for r in rows]
        if len(rows)!=8910 or len(set(ids))!=8910 or set(ids)!=set(qby) or any(r['prompt']!=qby[str(r['question_id'])]['text'] or r.get('metadata',{}).get('actual_visual_tokens_retained')!=32 or str(r.get('text','')).strip() in ('','FAILED') for r in rows):
            raise ValueError('POPE complete input/runtime identity failure')
        protocol=read(protocol_path); verified=checked_source_maps(protocol)
        if protocol.get('generation_exit_code')!=0 or sha(directory/'run_pope_streamwait_control.py')!=protocol['launcher_source_sha256']:
            raise ValueError('POPE protocol generation/source failure')
        exact=official_score('pope',output,question_file=qf,annotation_dir=qf.parent)
        if exact['categories']!=report['scores'][method]['categories']:
            raise ValueError('POPE official rescore differs')
        add_files(files,[output,protocol_path,directory/(stem+'.official.score.json'),directory/(stem+'.log'),*verified])
        scores.append(dict(method=method,n=8910,metric='POPE as-asked average F1',accuracy=exact['summary']['average_f1'],prediction_sha256=sha(output)))
    for filename,expected in report['sha256'].items():
        if sha(filename)!=expected:
            raise ValueError('POPE report source hash failure: '+filename)
        add_files(files,[filename])
    for stem in ['pope_v15_AZ32','pope_v15_AZ32_streamwait']:
        add_files(files,[directory/(stem+suffix) for suffix in ('.jsonl','.protocol.json','.official.score.json','.replay_comparison.json','.comparison.json','.images.json')])
    add_files(files,[directory/'run_pope_streamwait_control.py',directory/'resume_pope_streamwait_stage2.py',
                     directory/'score_pope_streamwait_control.py',directory/'score_pope_streamwait_baseline.py',
                     directory/'pope_streamwait_stage2_recovery.json',report_path])
    return files,scores


def group_definitions(jobs):
    by = {(j['model'],j['task'],j['arm']):j for j in jobs}
    groups=[]
    for model in ('v15','next'):
        for task in ('textvqa','sqa'):
            for budget in (128,64,32):
                if task=='textvqa':
                    e='EGATHER' if budget==128 else f'EGATHER_K{budget}'
                    az={128:'LRMAIN025',64:'LRMAIN0125',32:'LRMAIN00625'}[budget]
                else:e,az=f'E_GATHER_K{budget}',f'AZ_K{budget}'
                groups.append((f'{model}_{task}_K{budget}',[by[(model,task,e)],by[(model,task,az)]]))
            groups.append((f'{model}_{task}_FULL',[by[(model,task,'FULL')]]))
    groups.append(('v15_pope_K32_streamwait',None))
    return groups


def bundle(group_id, files, scores):
    entries={name:dict(sha256=digest(data),bytes=len(data),archived_as='regular_file') for name,data in sorted(files.items())}
    identity=digest(json.dumps(entries,sort_keys=True).encode())
    manifest=dict(group_id=group_id,identity=identity,prepared_utc=now(),scores=scores,files=entries,
                  source_symlinks_dereferenced=True,contains_weights_or_images=False)
    directory=LOCAL/'prepared'/group_id;directory.mkdir(parents=True,exist_ok=True)
    archive=directory/'evidence.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for name,data in {**files,'MANIFEST.json':json.dumps(manifest,indent=2).encode()}.items():
            info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644
            tar.addfile(info,io.BytesIO(data))
    blobs=[]
    if archive.stat().st_size<MAX_BLOB:
        blobs=[archive]
    else:
        with archive.open('rb') as stream:
            index=0
            while chunk:=stream.read(MAX_BLOB):
                path=directory/f'evidence.tar.gz.part{index:03d}';path.write_bytes(chunk);blobs.append(path);index+=1
    manifest['archive_sha256']=sha(archive)
    manifest['blobs']=[dict(filename=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in blobs]
    atomic(directory/'manifest.json',manifest)
    return dict(group_id=group_id,identity=identity,manifest=manifest,directory=str(directory),blobs=[str(p) for p in blobs])


def prepare(state):
    monitor=json.loads(command([sys.executable,str(SCRIPTS/'monitor_repair_hourly.py'),'--once']))
    if monitor['needs_attention']:
        raise ValueError('Independent monitor requires attention; publication deferred')
    summary=read(BATCH/'repaired_results_summary.json')
    if len(summary['rows'])!=28 or summary['errors']:
        raise ValueError('Unexpected summary rows/errors')
    summarizer=module('build_repair_run_summary')
    completed={row['output'] for row in summary['rows'] if row['status']=='complete'}
    candidates=[];pending=[]
    for group_id,jobs in group_definitions(summarizer.entries()):
        if group_id in state.get('published_groups',{}):
            continue
        if jobs is None:
            if not (OUT/'rerun_fast/pope_v15_EADP32_streamwait.fair_comparison.json').exists():
                pending.append(group_id);continue
            files,scores=validate_pope()
        else:
            if not all(job['output'] in completed for job in jobs):
                pending.append(group_id);continue
            files={};scores=[]
            for job in jobs:
                members,score=validate_job(job,summarizer);files.update(members);scores.append(score)
        candidates.append(bundle(group_id,files,scores))
    common_files={}
    for filename in ['paper_reference.json','mme_official_gt_rescore.json','sqa_input_identity.json',
                     'sqa_official_zip_identity.json','sqa_input_comparison.json','textvqa_guidance_manifest.json',
                     'latest_method_comparison.json','latest_method_comparison.csv','latest_method_comparison_validation.json']:
        add_files(common_files,[OUT/filename])
    for filename in ['lane1_v15_plan.json','lane2_next_plan.json','rerun_schedule_manifest.json',
                     'repaired_results_summary.json','repaired_results_summary.csv']:
        add_files(common_files,[BATCH/filename])
    if not state.get('common_audit_published'):
        common=bundle('common_audit',common_files,[])
    else:common=None
    return dict(prepared_utc=now(),available_groups=candidates,pending_groups=pending,
                common_audit=common,all_experiments_complete=monitor['all_complete'],allowlist=ALLOWLIST)


def remote_head():
    text=command(['git','ls-remote','--heads','origin',REF])
    if not text:
        raise ValueError('Expected origin branch does not exist')
    return text.split()[0]


def finish_pending(state):
    """Recover a successful push even if the previous process lost its reply."""
    publication=state.get('pending_publication')
    commit=state.get('pending_commit')
    if not commit or not publication:
        return state
    reached=remote_head()
    if reached==publication['parent_remote_head']:
        command(['git','cat-file','-e',commit+'^{commit}'])
        command(['git','push','origin',commit+':'+REF])
        reached=remote_head()
    if reached!=commit:
        raise ValueError('Remote changed while a publication was pending; manual review required')
    state.setdefault('published_groups',{})
    for package in publication['groups']:
        state['published_groups'][package['group_id']]=dict(identity=package['identity'],
            commit=commit,scores=package['scores'],published_utc=now())
    state.update(status='published',remote_head=commit,local_head=publication['local_head'],
                 common_audit_published=True,pending_commit=None,pending_groups=[],
                 pending_publication=None,updated_utc=now())
    atomic(STATE,state)
    return state


def publish(plan,state):
    if not plan['available_groups']:
        return state
    if command(['git','branch','--show-current'])!=BRANCH:
        raise ValueError('Live branch is not the authorized branch')
    index=Path(command(['git','rev-parse','--git-path','index']))
    if not index.is_absolute():index=ROOT/index
    before_index=sha(index);local_head=command(['git','rev-parse','HEAD']);base=remote_head()
    expected_base=state.get('remote_head',local_head)
    if base!=expected_base:
        raise ValueError('Origin has an unrecognized new commit; refusing to overwrite audit sources')
    for package in plan['available_groups']+([plan['common_audit']] if plan['common_audit'] else []):
        for blob in package['manifest']['blobs']:
            path=Path(package['directory'])/blob['filename']
            if sha(path)!=blob['sha256'] or path.stat().st_size!=blob['bytes'] or blob['bytes']>=90*1024*1024:
                raise ValueError('Prepared archive hash/size changed')
    if subprocess.run(['git','cat-file','-e',base+'^{commit}'],cwd=ROOT,capture_output=True).returncode:
        command(['git','fetch','origin',REF])
    parent=Path(tempfile.mkdtemp(prefix='eadp-publish-'));worktree=parent/'worktree'
    try:
        command(['git','worktree','add','--detach',str(worktree),base])
        selected=[]
        for relative in ALLOWLIST:
            if relative in EXCLUDED:
                raise ValueError('Excluded user file in allowlist')
            source=ROOT/relative
            if source.is_file():
                target=worktree/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(source.read_bytes());selected.append(relative)
        bundles=plan['available_groups']+([plan['common_audit']] if plan['common_audit'] else [])
        for package in bundles:
            destination=worktree/DEST/package['group_id'];destination.mkdir(parents=True,exist_ok=True)
            for source in [Path(package['directory'])/'manifest.json',*[Path(p) for p in package['blobs']]]:
                target=destination/source.name;shutil.copyfile(source,target);selected.append(str(target.relative_to(worktree)))
        groups=[p['group_id'] for p in plan['available_groups']]
        latest=worktree/DEST/'publication_manifest.json'
        history=dict(state.get('published_groups',{}))
        for package in plan['available_groups']:
            history[package['group_id']]=dict(identity=package['identity'],scores=package['manifest']['scores'])
        atomic(latest,dict(updated_utc=now(),new_groups=groups,published_groups=history,
                          local_head_preserved=local_head,parent_remote_head=base,
                          allowlist=ALLOWLIST,excluded_user_files=sorted(EXCLUDED)))
        selected.append(str(latest.relative_to(worktree)))
        command(['git','add','-f','--',*selected],cwd=worktree)
        staged=command(['git','diff','--cached','--name-only'],cwd=worktree).splitlines()
        if any(path not in selected for path in staged):
            raise ValueError('Unexpected staged path in isolated worktree')
        body=parent/'commit-message.txt'
        body.write_text('Publish completed EADP repair groups\n\nGroups: '+', '.join(groups)+'\n\nComplete predictions, official score checks, runtime/protocol evidence and SHA256 manifests are archived per group. The live checkout index and user edits are preserved.\n')
        command(['git','commit','-F',str(body)],cwd=worktree)
        commit=command(['git','rev-parse','HEAD'],cwd=worktree)
        if remote_head()!=base:
            raise ValueError('Origin branch advanced during packaging; retry with current remote HEAD')
        state.update(status='pushing',pending_commit=commit,pending_groups=groups,local_head=local_head,
                     pending_publication=dict(parent_remote_head=base,local_head=local_head,
                         groups=[dict(group_id=p['group_id'],identity=p['identity'],scores=p['manifest']['scores'])
                                 for p in plan['available_groups']]),updated_utc=now())
        atomic(STATE,state)
        command(['git','push','origin',commit+':'+REF],cwd=worktree)
        reached=remote_head()
        if reached!=commit:
            raise ValueError('Remote branch did not confirm published commit')
        if sha(index)!=before_index or command(['git','rev-parse','HEAD'])!=local_head:
            raise ValueError('Live HEAD/index changed unexpectedly; inspect before retry')
        state.setdefault('published_groups',{})
        for package in plan['available_groups']:
            state['published_groups'][package['group_id']]=dict(identity=package['identity'],commit=commit,scores=package['manifest']['scores'],published_utc=now())
        state.update(status='published',remote_head=commit,local_head=local_head,live_index_preserved=True,
                     common_audit_published=True,pending_commit=None,pending_groups=[],pending_publication=None,updated_utc=now())
        atomic(STATE,state)
        return state
    finally:
        if worktree.exists():
            command(['git','worktree','remove','--force',str(worktree)])
        shutil.rmtree(parent,ignore_errors=True)


def main():
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--validate',action='store_true')
    modes.add_argument('--publish',action='store_true');modes.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=int,default=30)
    args=parser.parse_args();LOCAL.mkdir(parents=True,exist_ok=True)
    lock=(LOCAL/'publish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(STATE) if STATE.exists() else dict(created_utc=now(),published_groups={})
    while True:
        try:
            if args.publish or args.watch:
                state=finish_pending(state)
            plan=prepare(state);atomic(LOCAL/'prepared_plan.json',plan)
            if args.publish or args.watch:
                state=publish(plan,state)
            else:
                state.update(status='prepared',available_groups=[p['group_id'] for p in plan['available_groups']],
                             pending_groups=plan['pending_groups'],updated_utc=now());atomic(STATE,state)
            print(json.dumps(dict(status=state.get('status','prepared'),available_groups=[p['group_id'] for p in plan['available_groups']],
                                  published_groups=list(state.get('published_groups',{})),pending_groups=plan['pending_groups'],
                                  remote_head=state.get('remote_head'),all_experiments_complete=plan['all_experiments_complete'])),flush=True)
            if len(state.get('published_groups',{}))==17 and plan['all_experiments_complete']:
                state.update(status='complete',finished_utc=now());atomic(STATE,state)
                atomic(LOCAL/'publication.finished.json',dict(success=True,remote_head=state['remote_head'],n_groups=17,completed_utc=now()))
                return
        except Exception as error:
            state.update(status='failed_retryable',error=str(error),updated_utc=now());atomic(STATE,state)
            traceback.print_exc()
            if not args.watch:raise
        if not args.watch:return
        time.sleep(max(10,args.interval))


if __name__=='__main__':
    main()
