"""Publish only complete repaired-protocol pairs, without touching the live index.

--prepare/--validate are CPU-only and never commit/push. --publish uses a
detached worktree and a normal fast-forward push to the fixed origin branch.
--watch retries every 30 seconds; no generation process or source is changed.
"""
from __future__ import annotations

import argparse
import csv
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
 'vizwiz_official_normalization.py','publish_repair_results.py','test_publish_repair_results.py')]]
ALLOWLIST = TRACKED + NEW
METADATA_ARTIFACTS = ['llava_paper_configuration_ambiguities.json','mme_dependency_impact.json',
                      'audit_provenance_manifest.json','sqa_shipped_control_analysis.json']
CONTROL_GROUPS = {
    'next_textvqa_K32_streamwait': ('EADP_beta2', 'AZ_beta2'),
    'next_textvqa_K32_default_beta1': ('EADP_beta1',),
}


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


def diagnosis_directory():
    return OUT/'next_gap_diagnosis_20261008'


def load_control_worker():
    path=diagnosis_directory()/'stability/next_text_full_streamwait.py'
    spec=importlib.util.spec_from_file_location('publication_full_control',path)
    loaded=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def control_registration():
    """A prepared plan is not a request to generate or wait for new results."""
    directory=diagnosis_directory()/'stability'
    state_path=directory/'next_text_full_streamwait.controller.state.json'
    if not state_path.is_file():
        return None
    state=read(state_path)
    if not state.get('started_utc'):
        return None
    datetime.datetime.fromisoformat(state['started_utc'])
    methods=state.get('registered_methods',state.get('selected_arms',[]))
    if not methods or len(set(methods))!=len(methods) or not set(methods)<=set(sum(CONTROL_GROUPS.values(),())):
        raise ValueError('Invalid registered full-control methods')
    if state.get('selected_arms',methods)!=methods:
        raise ValueError('Control registration aliases differ')
    if set(methods)&set(CONTROL_GROUPS['next_textvqa_K32_streamwait']) and not set(CONTROL_GROUPS['next_textvqa_K32_streamwait'])<=set(methods):
        raise ValueError('Register both beta2 comparison arms together')
    expected=[group for group,arms in CONTROL_GROUPS.items() if set(arms)<=set(methods)]
    if state.get('publication_expected_groups',expected)!=expected:
        raise ValueError('Control publication group contract differs')
    plan_path=directory/'next_text_full_streamwait.controller.plan.json'
    if not plan_path.is_file():
        raise ValueError('Registered full-control plan is missing')
    plan=read(plan_path)
    if state.get('plan_sha256') and state['plan_sha256']!=sha(plan_path):
        raise ValueError('Registered controller plan hash changed')
    controller_path=diagnosis_directory()/'schedule_next_full_controls.py'
    if plan.get('controller_sha256')!=sha(controller_path):
        raise ValueError('Registered controller source hash changed')
    if any(arm not in plan.get('jobs',{}) or plan['jobs'][arm].get('arm')!=arm for arm in methods):
        raise ValueError('Registered controller plan omits an arm')
    return dict(started_utc=state['started_utc'],registered_methods=methods,
                expected_groups=expected,attempt=state.get('attempt',plan.get('attempt',1)),
                plan_path=str(plan_path),plan_sha256=sha(plan_path),plan=plan,
                controller_path=str(controller_path))


def control_paths(registration,arm):
    worker=load_control_worker()
    job=registration['plan']['jobs'][arm]
    paths=worker.paths(arm,job['attempt'])
    # The worker owns fixed names; neither registration nor a report can redirect
    # archive output to arbitrary user or account files.
    directory=(diagnosis_directory()/'stability').resolve()
    if any(Path(path).parent.resolve()!=directory for path in paths.values()):
        raise ValueError('Full-control path outside authorized stability directory')
    if (job['output_paths']!={name:str(path) for name,path in paths.items()}
            or job['control_protocol']!=str(paths['control_protocol'])
            or job['control_protocol_sha256']!=sha(paths['control_protocol'])):
        raise ValueError('Full-control paths or frozen protocol differ from registration')
    return worker,paths


def control_group_ready(registration,arms):
    return all(control_paths(registration,arm)[1]['finished'].is_file() for arm in arms)


def validate_control(registration,arm):
    worker,paths=control_paths(registration,arm)
    for name in ('prediction','runtime','native_protocol','control_protocol','score','state','finished'):
        if not paths[name].is_file():
            raise ValueError('Missing completed full-control evidence: '+name)
    protocol=read(paths['control_protocol']);finished=read(paths['finished']);state=read(paths['state']);stored=read(paths['score'])
    beta=1. if arm=='EADP_beta1' else 2.
    anchorzip=arm=='AZ_beta2'
    expected=dict(alpha=.5,beta=beta,visual_token_num=32,temperature=0,
                  top_p=None,num_beams=1,max_new_tokens=128,conv_mode='vicuna_v1',anchorzip=anchorzip)
    if protocol.get('arm')!=arm or protocol.get('question_count')!=5000 or protocol.get('attempt')!=registration['plan']['jobs'][arm]['attempt'] or protocol.get('index_trace') is not False:
        raise ValueError('Full-control arm/protocol/observation mismatch')
    if any(protocol.get('parameters',{}).get(key)!=value for key,value in expected.items()):
        raise ValueError('Full-control frozen parameters mismatch')
    if protocol.get('launcher_sha256')!=sha(Path(worker.__file__)):
        raise ValueError('Full-control worker source hash mismatch')
    ast_validation=protocol.get('AST_validation',{})
    if ast_validation.get('restoration_ast_identical') is not True or ast_validation.get('record_stream_added') is not False or ast_validation.get('inserted_nodes')!=[
            'caller_stream = torch.cuda.current_stream(device=self.device)',
            'image_stream.wait_stream(caller_stream)','text_stream.wait_stream(caller_stream)']:
        raise ValueError('Full-control wait-only AST contract mismatch')
    if (finished.get('success') is not True or finished.get('complete') is not True
            or finished.get('n')!=5000 or finished.get('generation_exit_code')!=0
            or finished.get('arm')!=arm or state.get('generation_exit_code')!=0
            or state.get('complete') is not True or not state.get('started_utc')):
        raise ValueError('Full-control completion/exit gate failed')
    if finished.get('score_sha256')!=sha(paths['score']) or finished.get('prediction_sha256')!=sha(paths['prediction']):
        raise ValueError('Full-control finished source hash mismatch')
    verified=checked_source_maps(protocol)
    native=read(paths['native_protocol']);verified.update(checked_source_maps(native))
    worker.verify_images(protocol)
    with Path(protocol['parameters']['question_file']).open() as stream:
        questions=[json.loads(line) for line in stream]
    exact=worker.validate_and_score(paths['prediction'],paths['runtime'],questions,protocol['parameters'])
    if any(stored.get(key)!=value for key,value in exact.items()) or stored.get('arm')!=arm or stored.get('control_protocol_sha256')!=sha(paths['control_protocol']) or stored.get('image_manifest_sha256')!=protocol.get('image_manifest_sha256'):
        raise ValueError('Full-control exact official score/source differs')
    files={}
    add_files(files,[*paths.values(),*verified,Path(worker.__file__),
                     protocol['image_manifest'],registration['plan_path'],registration['controller_path']])
    for path in checked_source_maps(registration['plan']):
        add_files(files,[path])
    score=dict(method='AnchorZip' if anchorzip else 'EADP',control_arm=arm,beta=beta,
               accuracy=exact['accuracy_percent'],n=5000,metric='official TextVQA accuracy',
               runtime_trace_available=True,output=str(paths['prediction']),
               prediction_sha256=sha(paths['prediction']),verified_source_sha256=verified,
               protocol='full_NeXT_TextVQA_K32_wait_only_control',
               interpretation='published-script default sensitivity control; paper configuration unverified' if beta==1. else 'same wait-only repair applied to both methods')
    return files,score


def diagnostic_metadata():
    """Only the coordinator's hashed, explicit complete-evidence selection."""
    report_path=OUT/'next_gap_diagnosis_report.json'
    if not report_path.is_file():
        return {}
    report=read(report_path)
    if report.get('panel_complete') is not True:
        return {}
    files={};allowed_root=diagnosis_directory().resolve()
    for entry in report.get('files',[]):
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Diagnostic selection has an unsafe path')
        source=ROOT/relative
        if not source.resolve().is_relative_to(allowed_root) or any(token in str(relative).lower() for token in ('private','environ','agent.events','/hourly_monitor/','.state.','.lock')):
            raise ValueError('Diagnostic selection outside allowed public evidence')
        if source.stat().st_size>=90*1024*1024 or sha(source)!=entry['sha256'] or source.stat().st_size!=entry['bytes']:
            raise ValueError('Diagnostic selected source hash/size mismatch')
        add_files(files,[source])
    result_path=diagnosis_directory()/'stability/next_text_K32_EADP_stream_panel128.result.json'
    result=read(result_path)
    protocol_path=Path(result['protocol']);protocol=read(protocol_path)
    manifest_path=Path(protocol['manifest']);manifest=read(manifest_path)
    records_path=Path(protocol['output_paths']['records'])
    required=[result_path,protocol_path,manifest_path,records_path]
    if any(str(path.relative_to(ROOT)) not in files for path in required):
        raise ValueError('Diagnostic report omits complete panel evidence')
    if (result.get('success') is not True or result.get('complete') is not True
            or result.get('summary',{}).get('n')!=128 or result['summary'].get('n_generations')!=512
            or protocol.get('index_trace') is not False or manifest.get('n')!=128
            or sha(protocol_path)!=result['protocol_sha256'] or sha(manifest_path)!=result['manifest_sha256']
            or sha(records_path)!=result['records_sha256']):
        raise ValueError('Diagnostic panel completion/source gate failed')
    with records_path.open() as stream:
        records=[json.loads(line) for line in stream]
    variants={'as_is_1','as_is_2','wait_1','wait_2'}
    if len(records)!=128 or any(set(row.get('arms',{}))!=variants for row in records):
        raise ValueError('Diagnostic panel has incomplete repetitions')
    if sha(diagnosis_directory()/'stability/next_text_stream_panel.py')!=protocol['worker_sha256']:
        raise ValueError('Completed panel worker hash changed')
    checked_source_maps(protocol)
    add_files(files,[report_path])
    return files


def publication_complete(plan,state):
    return plan['all_experiments_complete'] and set(plan['expected_groups'])<=set(state.get('published_groups',{}))


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
    original_groups=group_definitions(summarizer.entries())
    expected=[group_id for group_id,_ in original_groups]
    registration=control_registration()
    dynamic=registration['expected_groups'] if registration else []
    previous=set(state.get('expected_groups',[]))&set(CONTROL_GROUPS)
    if not previous<=set(dynamic):
        raise ValueError('Previously registered diagnostic controls disappeared')
    expected.extend(dynamic)
    for group_id,jobs in original_groups:
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
    for group_id in dynamic:
        if group_id in state.get('published_groups',{}):
            continue
        arms=CONTROL_GROUPS[group_id]
        if not control_group_ready(registration,arms):
            pending.append(group_id);continue
        files={};scores=[]
        for arm in arms:
            members,score=validate_control(registration,arm);files.update(members);scores.append(score)
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
                common_audit=common,all_experiments_complete=monitor['all_complete'],allowlist=ALLOWLIST,
                expected_groups=expected,control_registration=registration)


def remote_head():
    text=command(['git','ls-remote','--heads','origin',REF])
    if not text:
        raise ValueError('Expected origin branch does not exist')
    return text.split()[0]


def index_fingerprint():
    # Stage mode/blob/path identifies user staging; index stat cache does not.
    return digest(command(['git','ls-files','--stage','-z']).encode())


def readable_results(history):
    stream=io.StringIO(newline='')
    columns=['group_id','model','task','budget','EADP','AnchorZip','AZ_minus_EADP','FULL','n','metric','manifest']
    writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
    lines=['# Published repaired-protocol results','',
           'Only complete, independently scored comparison groups appear here. ',
           'Read [paired_results.csv](paired_results.csv) for scores and each group manifest for prediction, input and source SHA256 evidence.',
           '', 'TextVQA uses all 5000 official validation questions. ScienceQA uses the 2017 image questions in the shipped CQM-A input. POPE uses the three-category mean F1 over 8910 predictions with the controlled CUDA stream wait.',
           '', 'For NeXT, K128/K64/K32 denotes the per-crop budget parameter (nominal 640/320/160 for five crops). Actual retained tokens are recorded per sample in the runtime evidence.',
           '', '| Group | EADP | AnchorZip | AZ − EADP | FULL | Evidence |',
           '|---|---:|---:|---:|---:|---|']
    control_notes=[]
    for group_id,entry in sorted(history.items()):
        scores=entry['scores'];parts=group_id.split('_');full=group_id.endswith('_FULL')
        if group_id=='next_textvqa_K32_default_beta1':
            if len(scores)!=1 or scores[0].get('method')!='EADP' or scores[0].get('beta')!=1.:
                raise ValueError('Default beta1 control mislabeled')
            control_notes.extend(['', 'EADP K32 beta1 script-default sensitivity control: '
                          f"{scores[0]['accuracy']:.3f}% over 5000 questions, with the same stream wait. "
                          'This is a separate EADP control; its parameters are not verified to be the paper configuration. '
                          '[manifest]('+group_id+'/manifest.json).'])
            continue
        e=az=None
        if full:full_value=scores[0]['accuracy']
        else:
            full_value=None
            if scores[0].get('method','').startswith('AnchorZip'):az,e=scores
            else:e,az=scores
        manifest=group_id+'/manifest.json'
        row=dict(group_id=group_id,model=parts[0],task=parts[1],budget=parts[2],
                 EADP='' if e is None else e['accuracy'],AnchorZip='' if az is None else az['accuracy'],
                 AZ_minus_EADP='' if e is None else az['accuracy']-e['accuracy'],
                 FULL='' if full_value is None else full_value,n=scores[0]['n'],metric=scores[0]['metric'],manifest=manifest)
        writer.writerow(row)
        values=[row[k] for k in ('EADP','AnchorZip','AZ_minus_EADP','FULL')]
        display=['' if value=='' else f'{value:.3f}' for value in values]
        lines.append('| '+group_id+' | '+' | '.join(display)+' | [manifest]('+manifest+') |')
    lines.extend(control_notes)
    return stream.getvalue(), '\n'.join(lines)+'\n'


def metadata_snapshot(plan,state):
    """Capture exact publish bytes before Git so concurrent edits remain visible."""
    history=dict(state.get('published_groups',{}))
    for package in plan['available_groups']:
        history[package['group_id']]=dict(identity=package['identity'],scores=package['manifest']['scores'])
    files={};sources={};artifacts={}
    for relative in ALLOWLIST:
        if relative in EXCLUDED:
            raise ValueError('Excluded user file in allowlist')
        source=ROOT/relative
        data=source.read_bytes() if source.is_file() else None
        sources[relative]=None if data is None else digest(data)
        if data is not None:files[relative]=data
    for name in METADATA_ARTIFACTS:
        source=OUT/name;data=source.read_bytes() if source.is_file() else None
        artifacts[name]=None if data is None else dict(sha256=digest(data),bytes=len(data))
        if data is not None:files[str(DEST/'audit_metadata'/name)]=data
    for relative,data in diagnostic_metadata().items():
        source=Path(relative)
        if relative==str((OUT/'next_gap_diagnosis_report.json').relative_to(ROOT)):
            name='next_gap_diagnosis_report.json'
        else:
            name=str(source.relative_to(OUT.relative_to(ROOT)))
        artifacts[name]=dict(sha256=digest(data),bytes=len(data))
        files[str(DEST/'audit_metadata'/name)]=data
    registration=plan.get('control_registration')
    if registration:
        # No mutable progress/state bytes in the metadata fingerprint.
        stable={key:registration[key] for key in ('started_utc','registered_methods','expected_groups','attempt','plan_sha256')}
        data=json.dumps(stable,sort_keys=True,indent=2).encode()
        files[str(DEST/'audit_metadata/full_control_registration.json')]=data
        artifacts['full_control_registration.json']=dict(sha256=digest(data),bytes=len(data))
    table,readme=readable_results(history)
    files[str(DEST/'paired_results.csv')]=table.encode()
    files[str(DEST/'README.md')]=readme.encode()
    files[str(DEST/'audit_metadata/manifest.json')]=json.dumps(
        {name:entry for name,entry in artifacts.items() if entry is not None},ensure_ascii=False,indent=2).encode()
    memo=dict(allowlist=list(ALLOWLIST),metadata_artifacts=list(METADATA_ARTIFACTS),
              sources=sources,artifacts=artifacts,paired_csv_sha256=digest(table.encode()),
              readme_sha256=digest(readme.encode()))
    return digest(json.dumps(memo,sort_keys=True).encode()),files,history


def success(state,**updates):
    state.pop('error',None)
    state.update(status='published',updated_utc=now(),**updates)
    atomic(STATE,state)
    return state


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
    return success(state,remote_head=commit,local_head=publication['local_head'],
                   common_audit_published=True,pending_commit=None,pending_groups=[],
                   pending_publication=None,metadata_fingerprint=publication.get('metadata_fingerprint'))


def publish(plan,state):
    fingerprint,metadata_files,history=metadata_snapshot(plan,state)
    if (not plan['available_groups'] and plan['common_audit'] is None
            and not state.get('pending_commit') and state.get('metadata_fingerprint')==fingerprint):
        return success(state,no_op=True,last_checked_utc=now(),pending_groups=plan['pending_groups'])
    if command(['git','branch','--show-current'])!=BRANCH:
        raise ValueError('Live branch is not the authorized branch')
    index=Path(command(['git','rev-parse','--git-path','index']))
    if not index.is_absolute():index=ROOT/index
    before_index=sha(index);before_staging=index_fingerprint()
    local_head=command(['git','rev-parse','HEAD']);base=remote_head()
    expected_base=state.get('remote_head') or local_head
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
        for relative,data in metadata_files.items():
            target=worktree/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data);selected.append(relative)
        bundles=plan['available_groups']+([plan['common_audit']] if plan['common_audit'] else [])
        for package in bundles:
            destination=worktree/DEST/package['group_id'];destination.mkdir(parents=True,exist_ok=True)
            for source in [Path(package['directory'])/'manifest.json',*[Path(p) for p in package['blobs']]]:
                target=destination/source.name;shutil.copyfile(source,target);selected.append(str(target.relative_to(worktree)))
        groups=[p['group_id'] for p in plan['available_groups']]
        command(['git','add','-f','--',*selected],cwd=worktree)
        if not command(['git','diff','--cached','--name-only'],cwd=worktree):
            return success(state,metadata_fingerprint=fingerprint,no_op=True,last_checked_utc=now())
        latest=worktree/DEST/'publication_manifest.json'
        atomic(latest,dict(updated_utc=now(),new_groups=groups,published_groups=history,
                          local_head_preserved=local_head,parent_remote_head=base,
                          allowlist=ALLOWLIST,excluded_user_files=sorted(EXCLUDED),
                          expected_groups=plan.get('expected_groups',[])))
        selected.append(str(latest.relative_to(worktree)))
        command(['git','add','-f','--',*selected],cwd=worktree)
        staged=command(['git','diff','--cached','--name-only'],cwd=worktree).splitlines()
        if any(path not in selected for path in staged):
            raise ValueError('Unexpected staged path in isolated worktree')
        body=parent/'commit-message.txt'
        body.write_text(('Publish completed EADP repair groups' if groups else 'Refresh repaired-result tools and readable scores')+'\n\nGroups: '+(', '.join(groups) or 'metadata refresh; existing groups preserved')+'\n\nComplete predictions, official score checks, runtime/protocol evidence and SHA256 manifests are archived per group. The live checkout index and user edits are preserved.\n')
        command(['git','commit','-F',str(body)],cwd=worktree)
        commit=command(['git','rev-parse','HEAD'],cwd=worktree)
        if remote_head()!=base:
            raise ValueError('Origin branch advanced during packaging; retry with current remote HEAD')
        state.update(status='pushing',pending_commit=commit,pending_groups=groups,local_head=local_head,
                     pending_publication=dict(parent_remote_head=base,local_head=local_head,
                         metadata_fingerprint=fingerprint,
                         groups=[dict(group_id=p['group_id'],identity=p['identity'],scores=p['manifest']['scores'])
                                 for p in plan['available_groups']]),updated_utc=now())
        atomic(STATE,state)
        command(['git','push','origin',commit+':'+REF],cwd=worktree)
        reached=remote_head()
        if reached!=commit:
            raise ValueError('Remote branch did not confirm published commit')
        after_index=sha(index);after_staging=index_fingerprint()
        if after_staging!=before_staging or command(['git','rev-parse','HEAD'])!=local_head:
            raise ValueError('Live HEAD/index changed unexpectedly; inspect before retry')
        state.setdefault('published_groups',{})
        for package in plan['available_groups']:
            state['published_groups'][package['group_id']]=dict(identity=package['identity'],commit=commit,scores=package['manifest']['scores'],published_utc=now())
        return success(state,remote_head=commit,local_head=local_head,live_index_preserved=True,
                     live_index_byte_sha256_before=before_index,live_index_byte_sha256_after=after_index,
                     live_index_stat_cache_refreshed=before_index!=after_index,
                     live_staging_sha256_before=before_staging,live_staging_sha256_after=after_staging,
                     common_audit_published=True,pending_commit=None,pending_groups=[],pending_publication=None,
                     metadata_fingerprint=fingerprint,no_op=False)
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
            state['expected_groups']=plan['expected_groups']
            if args.publish or args.watch:
                state=publish(plan,state)
            else:
                state.update(status='prepared',available_groups=[p['group_id'] for p in plan['available_groups']],
                             pending_groups=plan['pending_groups'],updated_utc=now());atomic(STATE,state)
            print(json.dumps(dict(status=state.get('status','prepared'),available_groups=[p['group_id'] for p in plan['available_groups']],
                                  published_groups=list(state.get('published_groups',{})),pending_groups=plan['pending_groups'],
                                  remote_head=state.get('remote_head'),all_experiments_complete=plan['all_experiments_complete'])),flush=True)
            if publication_complete(plan,state):
                state.update(status='complete',finished_utc=now());atomic(STATE,state)
                atomic(LOCAL/'publication.finished.json',dict(success=True,remote_head=state['remote_head'],
                    n_groups=len(plan['expected_groups']),expected_groups=plan['expected_groups'],completed_utc=now()))
                return
        except Exception as error:
            state.update(status='failed_retryable',error=str(error),updated_utc=now());atomic(STATE,state)
            traceback.print_exc()
            if not args.watch:raise
        if not args.watch:return
        time.sleep(max(10,args.interval))


if __name__=='__main__':
    main()
