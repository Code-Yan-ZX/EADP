"""Prepare/check on CPU; execute once only with a separate root allocation note."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=BASE.parents[2]
CONTROL=BASE/'streamwait_control'
PYTHON='/home/dell/miniconda3/envs/llava_pruner/bin/python'
PRED=OUT/'v15_FULL_oldEN_streamwait.jsonl'
read=lambda p:json.loads(Path(p).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def stat(path):
    p=Path(path);s=p.stat()
    return dict(resolved=str(p.resolve()),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def current_ticks(pid):
    try:return int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return None


def gpu_pids():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return sorted({int(x.strip()) for x in text.splitlines() if x.strip()})


def check_plan(plan):
    assert plan['method']=='FULL' and plan['native_cli_visual_token_num']==0 and plan['actual_expected_retained_tokens']==576
    assert plan['parameters']['K']==0 and plan['parameters']['alpha']==.5 and plan['parameters']['beta']==1
    baseline=read(CONTROL/'protocol.json')
    parameters=copy.deepcopy(plan['parameters']);parameters['K']=128
    assert parameters==baseline['parameters'] and plan['dataset']==baseline['dataset']
    assert plan['dataset']['n_rows']==4377 and plan['dataset']['n_circular_groups']==1176
    assert sha(plan['dataset']['path'])==plan['dataset']['sha256']
    old,new=copy.deepcopy(baseline['command']),copy.deepcopy(plan['command'])
    for command in (old,new):
        command[1:3]=['ISOLATED_WORKER','ISOLATED_OUTPUT']
        command[command.index('--answers-file')+1]='ISOLATED_PREDICTIONS'
    assert new[new.index('--visual_token_num')+1]=='0'
    new[new.index('--visual_token_num')+1]='128'
    assert old==new
    for item in plan['source_files']:assert sha(item['path'])==item['sha256'],item['path']
    for path,guard in plan['model_stat_guards'].items():
        assert stat(path)=={k:v for k,v in guard.items() if k!='prior_verified_sha256'},path
    patcher=module('full_control_wait_builder',CONTROL/'patch_only.py')
    assert patcher.build_patch()[1]==plan['runtime_patch']
    verifier=module('full_control_CPU_verifier_adapter',OUT/'verify_full.py');verifier.adapted_core()
    assert 'torch' not in sys.modules,'CPU plan check must not import torch/model'


def prepare():
    assert not (OUT/'protocol.json').exists() and not (OUT/'state.json').exists() and not PRED.exists()
    search=read(BASE/'full_baseline_search/search_report.json')
    assert search['status']=='bounded_search_complete_no_qualified_v15_7B_oldEN_FULL_found'
    assert not search['eligible_oldEN4377_7B_prediction_candidates']
    baseline=read(CONTROL/'protocol.json');proof=read(CONTROL/'independent_verification.json')
    assert proof['success'] and proof['n_rows']==4377 and proof['n_circular_groups']==1176
    plan=copy.deepcopy(baseline)
    plan['prepared_utc']=datetime.now(timezone.utc).isoformat()
    plan['protocol']='Native FULL bypass K0, expected576 retained; same oldEN4377/three waits/greedy1024'
    plan['method']='FULL';plan['native_cli_visual_token_num']=0;plan['actual_expected_retained_tokens']=576
    prior_runtime=read(CONTROL/'observed/loaded_model.json')
    plan['runtime_versions']={k:prior_runtime[k] for k in ('torch','transformers')}
    plan['parameters']['K']=0
    command=[PYTHON,str(OUT/'worker.py'),str(OUT/'observed')]+baseline['command'][3:]
    command[command.index('--answers-file')+1]=str(PRED)
    command[command.index('--visual_token_num')+1]='0';plan['command']=command
    arch=ROOT/'_reproduction/author_eadp/LLaVA/llava/model/llava_arch.py'
    tree=ast.parse(arch.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LlavaMetaForCausalLM')
    encode=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='encode_images')
    bypass=next(n for n in encode.body if isinstance(n,ast.If) and ast.unparse(n.test)=='visual_token_num <= 0')
    assert isinstance(bypass.body[-1],ast.Return) and ast.unparse(bypass.body[-1].value)=='(image_features, index_masks)'
    plan['native_FULL_bypass']=dict(source=str(arch),source_sha256=sha(arch),line=bypass.lineno,
        original_bypass_AST_sha256=hashlib.sha256(ast.dump(bypass,include_attributes=False).encode()).hexdigest(),
        original_source=ast.unparse(bypass),source_arithmetic_modified=False,
        difference_from_K576_facility='K0 directly computes CLIP vision and mm_projector then an all-True mask. No dense/global CLIP similarity, entropy, smoothing, or facility selection executes. K576 would still execute that pruning arithmetic and is not this control.',
        text_tower_weights='Loaded by the unchanged builder, but text-guidance forward is unused by the source K0 bypass.')
    guard_plan=ROOT/'reproduction/20261028/pope/gpu_formula_facility_panel/run_plan.json'
    plan['model_stat_guards']=read(guard_plan)['model_stat_guards']
    references=['image_manifest.json','cpu_dependency_input_equivalence.json','cpu_llm_dependency_equivalence.json','cpu_inputs_current.json','cpu_llm_tokens_current.json']
    for name in references:shutil.copyfile(BASE/name,OUT/name)
    comparison=read(BASE/'comparison_reference.json')
    evidence=read(ROOT/'docs/versions/v2/evidence.json')['records']
    matches=[r for r in evidence if r.get('origin')=='eadp_paper_reported' and r.get('model')=='v15' and r.get('task')=='MMB_EN' and r.get('method')=='FULL']
    assert len(matches)==1 and matches[0]['value']==64.7 and matches[0]['budget_nominal']==576
    comparison['paper']=matches[0]
    comparison['comparison_boundary']='FULL native K0 bypass versus completed same oldEN4377 K128 beta1/three-wait control. Same fixed inputs and public decoding; not recovered paper argv or external-server judge.'
    save('comparison_reference.json',comparison)
    refs=[OUT/name for name in references]+[OUT/'comparison_reference.json',OUT/'run_full.py',OUT/'worker.py',OUT/'observe_full.py',OUT/'verify_full.py',
          BASE/'verify_completed.py',BASE/'public_mcq_metric_check.py',BASE/'full_baseline_search/search_report.json',CONTROL/'protocol.json',CONTROL/'independent_verification.json',CONTROL/'observed/loaded_model.json',guard_plan]
    utils=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval'
    refs += [utils/'utils/matching_util.py',utils/'dataset/utils/multiple_choice.py']
    plan['source_files'] += [dict(path=str(p),sha256=sha(p)) for p in refs]
    plan['observer_reuse']=dict(path=str(OUT/'observe_full.py'),sha256=sha(OUT/'observe_full.py'),
        basis=str(BASE/'observe_native.py'),extensions='Post-generation FULL mask/shape checks, actual input token SHA and saved output token IDs; no numerical replacements.')
    plan['planned_primary_metric']='released_public_exact_matching_circular'
    plan['same_input_pruned_control']=dict(protocol=str(CONTROL/'protocol.json'),protocol_sha256=sha(CONTROL/'protocol.json'),
        verification=str(CONTROL/'independent_verification.json'),verification_sha256=sha(CONTROL/'independent_verification.json'),K=128,value_percent=proof['circular_accuracy_percent'])
    plan['launch_policy']=dict(status='prepared_not_started',GPU_allocated=False,watcher_or_queue_created=False,
        root_allocation_note_required=True,single_attempt=True,automatic_retry=False,global_max_GPU_models=2)
    plan['runtime_observations']=dict(native_parameter=0,actual_retained=576,all4377_masks_all_true=True,
        complete_input_token_SHA=True,all4377_output_token_redecodes=True,pre_generation_tensor_reads=False)
    plan['limits']=['This is a new unpruned FULL-path calibration, not a replacement for any EADP pruning result.',
        'Only native visual_token_num128->0 switches the source-defined bypass; same model/data/full prompts/greedy1024/three waits. There is no parameter search.',
        'Bypass omits unused CLIP text/similarity/facility computations; K576 selection is a different entry and is not used.',
        'The released public exact-matching circular helper is the fixed primary; the unchanged local first-letter metric is also retained even if the parsers disagree. No higher score is selected.',
        'No external evaluation server or API is called; matching identity to the paper server and unpublished argv remains unverified.',
        'Prior independently hashed model payloads are reused only with all17 stat guards unchanged. Current runtime observations do not backfill historical environments.']
    save('protocol.json',plan);check_plan(plan)
    save('READY.json',dict(status='prepared_not_started',GPU_loaded=False,watcher_created=False,protocol_sha256=sha(OUT/'protocol.json'),
        root_allocation_note_required=True,run_command=[PYTHON,str(OUT/'run_full.py'),'run','--allocation-note','ROOT_REVIEWED_FUTURE_ALLOCATION.json']))
    print(json.dumps(read(OUT/'READY.json'),indent=2))


def run(note_path):
    plan=read(OUT/'protocol.json');check_plan(plan)
    assert sha(OUT/'protocol.json')==read(OUT/'READY.json')['protocol_sha256']
    note=read(note_path)
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']=='v15_FULL_oldEN4377_streamwait'
    assert note['protocol_sha256']==sha(OUT/'protocol.json') and note['GPU_slot'] in ('A','B')
    assert gpu_pids()==sorted(note['allowed_other_gpu_pids']) and len(gpu_pids())<2
    assert Path(sys.executable).resolve()==Path(PYTHON).resolve()
    assert not PRED.exists() and not (OUT/'state.json').exists()
    with (OUT/'launch.claim.json').open('x') as stream:json.dump(dict(pid=os.getpid(),start_ticks=current_ticks(os.getpid()),allocation=note,automatic_retry=False),stream,indent=2)
    save('allocation.json',dict(note=note,note_sha256=sha(note_path),actual_gpu_pids_before_launch=gpu_pids()))
    for key in list(os.environ):
        if key.startswith(('USE_LLAVA_ARCH_','ANCHORZIP_','EADP_','ENTROPY_')):os.environ.pop(key)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    runner=module('frozen_MMB_FULL_base_runner',BASE/'run_author.py')
    runner.OUT,runner.PRED,runner.DATA=OUT,PRED,Path(plan['dataset']['path']);runner.command=lambda:plan['command']
    original_score=runner.score
    def score():
        original_score()
        result=read(OUT/'local_circular_score.json')
        result.update(paper_reported=64.7,local_minus_paper=result['value']-64.7,
                      reference_correction='FULL64.7 replaces inherited EADP62.7 reference only; grouping/parsing/metric arithmetic unchanged.')
        save('local_circular_score.json',result)
    runner.score=score
    sys.path.insert(0,str(CONTROL))
    converter=module('frozen_MMB_FULL_converter',CONTROL/'run_control.py')
    converter.OUT,converter.PRED=OUT,PRED;runner.convert_submission=converter.convert_submission
    try:
        runner.run()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(BASE/'vendor_scoring'),PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([PYTHON,str(OUT/'verify_full.py')],env=env,check=True)
        check_plan(plan)
        state=read(OUT/'state.json');native=state['native_process'];now=current_ticks(native['pid'])
        assert now!=native['start_ticks'] and native['pid'] not in gpu_pids()
        proof=read(OUT/'independent_verification.json');assert proof['success'] and proof['mask_observation_checks']==4377
        save('release.json',dict(success=True,n_rows=4377,n_groups=1176,method='FULL',actual_retained_tokens=576,
            native_worker_exit=dict(original=native,original_exited=True,current_start_ticks_if_reused=now,GPU_pid_absent=True),
            independent_verification_sha256=sha(OUT/'independent_verification.json'),protocol_sha256=sha(OUT/'protocol.json'),
            raw_sha256=sha(PRED),controller_identity=state['runner_process'],
            controller_exit_must_be_confirmed_after_run_returns=True,primary_metric=plan['planned_primary_metric'],
            primary_value_percent=proof['primary_value_percent'],automatic_retry=False))
        state.update(status='complete',FULL_independent_verification_sha256=sha(OUT/'independent_verification.json'),release_sha256=sha(OUT/'release.json'))
        save('state.json',state)
    except BaseException as error:
        save('failure.json',dict(status='terminal_failure_no_retry',error=repr(error),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','check','run'],nargs='?',default='check');parser.add_argument('--allocation-note',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='check':check_plan(read(OUT/'protocol.json'));print(json.dumps(dict(CPU_validation=True,GPU_loaded=False,would_launch=False)))
    else:
        try:
            assert args.allocation_note is not None,'A separate root allocation note is required';run(args.allocation_note)
        except BaseException as error:
            if (OUT/'launch.claim.json').exists() and not (OUT/'failure.json').exists():
                save('failure.json',dict(status='terminal_failure_no_retry',error=repr(error),automatic_retry=False))
            raise
