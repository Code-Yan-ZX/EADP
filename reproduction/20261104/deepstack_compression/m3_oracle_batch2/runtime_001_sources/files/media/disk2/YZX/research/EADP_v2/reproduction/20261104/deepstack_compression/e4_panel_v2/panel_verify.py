"""Future independent full E4 panel acceptance; no references or scores."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;E4=HERE.parent/'e4'
sys.path.insert(1,str(E4))
from identity_worker import PLAN,sha,read_jsonl
from panel_v2_imports import load_local
_gate=load_local('panel_gate');validated_dev=_gate.validated_dev;validate_smoke_gate=_gate.validate_smoke_gate
validate_budget_supervisor=_gate.validate_budget_supervisor;validate_smoke_review=_gate.validate_smoke_review;validate_deadline=_gate.validate_deadline
check_native=load_local('native_panel_gate').check_native
from verify_k256 import validate_budget_summary,compare_record
from completion import alive


def validate_terminal(run,started,terminal,launch,exited):
    if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or
        terminal.get('remaining_process_group')!=[] or terminal.get('worker_pid')!=started.get('worker_pid')):
        raise ValueError('panel supervisor failed')
    validate_budget_supervisor(started,run,terminal['worker_pid'])
    validate_deadline(json.loads((HERE/'budget_plan.json').read_text()),launch['started_unix'])
    controller=launch.get('controller');container=exited.get('container_identity')
    if (not isinstance(controller,dict) or not isinstance(container,dict) or controller!=exited.get('controller') or
        controller.get('pid')!=terminal.get('worker_pid') or not container.get('container_id') or not isinstance(container.get('host_init'),dict)):
        raise ValueError('panel identity mismatch')
    for ident in (controller,container['host_init']):
        if (not isinstance(ident.get('pid'),int) or ident['pid']<=0 or not isinstance(ident.get('start_ticks'),int)
            or ident['start_ticks']<=0 or not ident.get('boot_id') or alive(ident)):raise ValueError('panel process still alive or unknown')
    if (exited.get('returncode')!=0 or exited.get('container_exit_code')!=0 or exited.get('timed_out') is not False or
        exited.get('interrupted') is not None or exited.get('container_stopped_verified') is not True or
        exited.get('client_exited_while_container_running') is not False or exited.get('gpu_pids_after')!=[]):
        raise ValueError('panel container did not exit cleanly')
    command=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    scripts=[(cwd/x).resolve() for x in command if Path(x).name=='panel_launch.py']
    if scripts!=[(HERE/'panel_launch.py').resolve()] or command.count('--run-id')!=1:raise ValueError('wrong panel launcher')
    index=command.index('--run-id')
    if index+1>=len(command) or run.name!='panel_'+command[index+1]:raise ValueError('wrong panel output')
    dock=launch.get('command',[])
    if (dock.count(str(HERE/'panel_worker.py'))!=1 or dock.count('--out')!=1 or
        Path(dock[dock.index('--out')+1]).resolve()!=run or launch.get('worker_sha256')!=sha(HERE/'panel_worker.py') or
        launch.get('launcher_sha256')!=sha(HERE/'panel_launch.py') or
        launch.get('budget_plan_sha256')!=sha(HERE/'budget_plan.json')):raise ValueError('wrong panel source or command')
    if launch.get('approval_bundle')!=load_local('panel_launch').check_panel_approval():raise ValueError('panel approval changed')
    inspected=subprocess.run(['docker','inspect',container['container_id']],capture_output=True,text=True,timeout=10,check=True)
    state=json.loads(inspected.stdout)[0]
    if state['Id']!=container['container_id'] or state['State']['Running'] or state['State']['ExitCode']!=0:
        raise ValueError('independent panel container inspection failed')


def main():
    p=argparse.ArgumentParser()
    for name in ('e1-acceptance','e1-started','e1-terminal','smoke-dir','smoke-proof','smoke-started','smoke-terminal',
                 'run-dir','run-started','run-terminal','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();e1,rows=validated_dev(a.e1_acceptance,a.e1_started,a.e1_terminal)
    validate_smoke_gate(a.e1_acceptance,a.smoke_dir,a.smoke_proof,a.smoke_started,a.smoke_terminal)
    smoke_review=validate_smoke_review(a.smoke_proof,a.smoke_terminal)
    native=check_native()
    run=a.run_dir.resolve();launch=json.loads((run/'launch.json').read_text());exited=json.loads((run/'exit.json').read_text())
    receipt=run/'host_native_gate.json'
    if launch.get('host_native_gate')!={'path':str(receipt),'sha256':sha(receipt)} or json.loads(receipt.read_text())!={
        'native_gate':native,'native_gate_source_sha256':sha(HERE/'native_panel_gate.py'),'host_live_docker_inspection':True}:
        raise ValueError('host native inspection receipt mismatch')
    validate_terminal(run,json.loads(a.run_started.read_text()),json.loads(a.run_terminal.read_text()),launch,exited)
    frozen=json.loads((run/'frozen.json').read_text());done=json.loads((run/'worker_complete.json').read_text())
    keys=('task','qid','group_id','message_sha256','image_sha256')
    if (frozen.get('stage')!='full' or frozen.get('n_questions')!=384 or frozen.get('selected')!=[{k:r[k] for k in keys} for r in rows]
        or frozen.get('e1_acceptance_sha256')!=sha(a.e1_acceptance) or frozen.get('smoke_proof_sha256')!=sha(a.smoke_proof)
        or frozen.get('plan_sha256')!=sha(PLAN) or frozen.get('inputs_sha256')!=e1['files']['inputs']['sha256'] or
        frozen.get('native_gate')!=native or frozen.get('smoke_review_gate')!=smoke_review
        or frozen.get('budgets')!=[128,256,512,1024] or frozen.get('ds_on')!=[False,True] or frozen.get('k256_reuse') is not False
        or frozen.get('gt_read') is not False or frozen.get('scoring') is not False or frozen.get('timing_claims') is not False
        or frozen.get('tensor_retention')!='summary_only_runtime_checked_no_tensor_reconstruction' or (run/'arrays').exists()
        or done.get('n_questions')!=384 or done.get('n_natural_generations')!=3072):raise ValueError('panel frozen protocol mismatch')
    for path,digest in frozen.get('code_sha256',{}).items():
        if sha(path)!=digest:raise ValueError('panel source changed')
    sources=[HERE/x for x in ('panel_worker.py','panel_launch.py','panel_gate.py','panel_verify.py','native_panel_gate.py','panel_v2_imports.py','budget_plan.json')]
    sources += [E4/x for x in ('smoke_contract.py','budget_runner.py','budget_adapter.py','decode_observer.py',
                 'off_adapter.py','skip_deepstack.py','identity_records.py','identity_worker.py','verify_k256.py')]
    for path in sources:
        if frozen.get('code_sha256',{}).get(str(path))!=sha(path):raise ValueError('missing panel source binding')
    records=read_jsonl(run/'records.jsonl');generated=read_jsonl(run/'generated.jsonl')
    expected=[(r['task'],str(r['qid']),k,on) for r in rows for k in (128,256,512,1024) for on in (False,True)]
    if [(r.get('task'),str(r.get('qid')),r.get('budget'),r.get('ds_on')) for r in records]!=expected:
        raise ValueError('panel arm inventory/order mismatch')
    fields=('task','qid','group_id','budget','ds_on','output_ids','eos_ids','generation_scalars','forced')
    if len(generated)!=len(records) or any(g!={k:r[k] for k in fields} for g,r in zip(generated,records)):
        raise ValueError('panel generation/observation mismatch')
    e1rows=read_jsonl(e1['files']['journal']['path'])
    e1map={(r['task'],str(r['qid']),r['arm']):r for r in e1rows if r['forced'] is False}
    if len(e1map)!=1536:raise ValueError('E1 inventory mismatch')
    bykey={(r['task'],str(r['qid'])):r for r in rows};text_hashes={};output_mismatch=0
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(json.loads(PLAN.read_text())['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
    for r in records:
        if r.get('forced') is not False:raise ValueError('full panel must use natural outputs')
        item=bykey[(r['task'],str(r['qid']))]
        if any(r.get(k)!=item[k] for k in keys):raise ValueError('panel input mismatch')
        ids=r.get('output_ids');eos=r.get('eos_ids')
        if (not isinstance(ids,list) or not ids or len(ids)>2048 or not isinstance(eos,list) or
            r.get('eos')!=bool(ids[-1] in eos) or r.get('truncated')!=bool(len(ids)==2048 and ids[-1] not in eos) or
            not (r['eos'] or r['truncated'])):raise ValueError('invalid panel generation endpoint')
        text=validate_budget_summary(r,e1map[(r['task'],str(r['qid']),'P3-D0')],tokenizer)
        question=(r['task'],str(r['qid']))
        if question in text_hashes and text_hashes[question]!=text:raise ValueError('panel prefill text differs')
        text_hashes[question]=text
        if r['budget']==256:
            arm='P3-D1' if r['ds_on'] else 'P3-D0';reference=e1map[(r['task'],str(r['qid']),arm)]
            local=copy.deepcopy(reference)
            for field in ('output_ids','answer','eos_ids','eos','truncated'):local[field]=r[field]
            compare_record(r,local,item)
            if any(r[field]!=reference[field] for field in ('output_ids','answer','eos_ids','eos','truncated')):output_mismatch+=1
    files=[a.e1_acceptance,a.e1_started,a.e1_terminal,a.smoke_proof,a.smoke_started,a.smoke_terminal,a.run_started,a.run_terminal,
           run/'launch.json',run/'exit.json',run/'frozen.json',run/'worker_complete.json',run/'records.jsonl',run/'generated.jsonl']
    result={'status':'passed','stage':'full','n_questions':384,'all_arms_validated':3072,'k256_reuse':False,
            'k256_e1_output_differences':output_mismatch,'GT_opened':False,'scoring_performed':False,
            'tensor_retention':'summary_only_runtime_checked_no_tensor_reconstruction','full_tensor_reconstruction':False,
            'files_sha256':{str(p.resolve()):sha(p) for p in files},'verifier_sha256':sha(__file__)}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
