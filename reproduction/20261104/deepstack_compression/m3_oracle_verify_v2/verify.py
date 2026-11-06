"""No-GT acceptance after all 128 questions and real container exit."""
import argparse,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
V2=Path(__file__).resolve().parent;ROOT=V2.parent;HERE=ROOT/'m3_oracle_exec';SMOKE=ROOT/'m3_gpu_smoke'
sys.path[:0]=[str(HERE),str(SMOKE),str(ROOT/'e4')]
import e1_admission,e1_baseline,runtime_gate,verify_contract,container_identity
from identity_worker import PLAN

def read(path):return json.loads(Path(path).read_text())
def lines(path):
    with Path(path).open() as f:return [json.loads(x) for x in f]

def check_terminal(run,started,terminal,launch,exited,plan):
    if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or
        started.get('worker_pid')!=terminal.get('worker_pid') or started.get('process_group')!=started.get('worker_pid') or
        started.get('timeout_seconds')!=plan['outer_timeout_seconds'] or started.get('stale_seconds')!=plan['progress_stale_seconds'] or
        started.get('snapshot_seconds')!=plan['snapshot_seconds'] or started.get('progress_paths')!=[str((run/'panel/panel_events.jsonl').resolve())]):
        raise ValueError('runtime supervisor not clean or mismatched')
    command=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    if ([(cwd/x).resolve() for x in command if Path(x).name=='runtime_launch.py']!=[(HERE/'runtime_launch.py').resolve()] or
        command.count('--run-id')!=1 or run.name!='runtime_'+command[command.index('--run-id')+1]):raise ValueError('wrong runtime launcher')
    container=exited.get('container_identity') or {};controller=launch.get('controller') or {}
    if (controller!=exited.get('controller') or controller.get('pid')!=terminal.get('worker_pid') or not container.get('container_id') or
        exited.get('returncode')!=0 or exited.get('container_exit_code')!=0 or exited.get('timed_out') is not False or
        exited.get('interrupted') is not None or exited.get('container_stopped_verified') is not True or
        exited.get('client_exited_while_container_running') is not False or exited.get('errors')!=[] or exited.get('gpu_pids_after')!=[]):
        raise ValueError('runtime container exit invalid')
    completion_path=ROOT/'e1_analysis/completion.py'
    spec=importlib.util.spec_from_file_location('m3_verify_v2_completion',completion_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);alive=module.alive
    for ident in (controller,container.get('host_init') or {}):
        if not isinstance(ident.get('pid'),int) or ident['pid']<=0 or not ident.get('start_ticks') or not ident.get('boot_id') or alive(ident):
            raise ValueError('runtime process still alive or identity missing')
    dock=launch.get('command',[])
    if (dock.count(str(HERE/'runtime_worker.py'))!=1 or dock.count('--execute')!=1 or dock.count('--out')!=1 or
        Path(dock[dock.index('--out')+1]).resolve()!=run or launch.get('worker_sha256')!=runtime_gate.sha(HERE/'runtime_worker.py') or
        launch.get('launcher_sha256')!=runtime_gate.sha(HERE/'runtime_launch.py')):raise ValueError('wrong runtime container source')
    q=subprocess.run(['docker','inspect',container['container_id']],capture_output=True,text=True,timeout=20,check=True)
    state=json.loads(q.stdout)[0]
    container_identity.validate(run,launch,exited,state)

def main():
    ap=argparse.ArgumentParser()
    for name in ('run-dir','run-started','run-terminal','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();run=a.run_dir.resolve();plan=read(HERE/'runtime_plan.json')
    approval=runtime_gate.validate(host=True,require_future=False)
    launch=read(run/'launch.json');exited=read(run/'exit.json');receipt=run/'host_receipt.json'
    if launch.get('host_receipt')!={'path':str(receipt),'sha256':runtime_gate.sha(receipt)} or read(receipt)!=approval or launch.get('approval')!=approval:
        raise ValueError('runtime admission receipt mismatch')
    check_terminal(run,read(a.run_started),read(a.run_terminal),launch,exited,plan)
    frozen=read(run/'frozen.json');done=read(run/'panel/worker_complete.json');sample_bytes=Path(plan['sample_path']).read_bytes();sample=json.loads(sample_bytes)
    if (frozen.get('stage')!='oracle_panel' or frozen.get('approval')!=approval or frozen.get('plan_sha256')!=runtime_gate.sha(HERE/'runtime_plan.json') or
        frozen.get('sample_sha256')!=plan['sample_sha256'] or frozen.get('GT_opened') is not False or frozen.get('scoring') is not False or
        done.get('status')!='completed' or done.get('questions')!=128 or done.get('new_generations')!=768 or done.get('scoring_performed') is not False or
        hashlib.sha256(sample_bytes).hexdigest()!=plan['sample_sha256']):raise ValueError('frozen/completion contract invalid')
    e1=e1_admission.validate();proofpath=ROOT/'e1/e1full001_completion.json';proof_bytes=proofpath.read_bytes();proof=json.loads(proof_bytes)
    if frozen['e1_completion_sha256']!=e1.proof_sha or frozen['e1_review_sha256']!=e1.review_sha or frozen['weights_identity_sha256']!=e1.weights_sha:
        raise ValueError('E1 provenance changed')
    j=Path(proof['files']['journal']['path']).read_bytes();inp=Path(proof['files']['inputs']['path']).read_bytes()
    events=lines(run/'panel/panel_events.jsonl');starts=[x for x in events if x['event']=='question_started'];finishes=[x for x in events if x['event']=='question_completed']
    if len(starts)!=128 or len(finishes)!=128 or events[-1].get('event')!='panel_completed' or any(x['event']=='panel_failed' for x in events):
        raise ValueError('panel event inventory incomplete')
    source_plan=read(proof['files']['plan']['path']);recipe=source_plan['recipe']
    if recipe.get('post_process',False) is not False:raise ValueError('unsupported output post-processing')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
    def decode(ids):return tokenizer.batch_decode([ids],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    results=[];question_files={}
    for i,sample_row in enumerate(sample['examples']):
        begin=starts[i];finish=finishes[i];key=(sample_row['task'],str(sample_row['qid']),sample_row['group_id'])
        if (begin.get('index')!=i or finish.get('index')!=i or (begin.get('task'),begin.get('qid'),begin.get('group_id'))!=key or
            (finish.get('task'),finish.get('qid'),finish.get('group_id'))!=key or begin.get('journal')!=finish.get('journal') or finish.get('new_generations')!=6):
            raise ValueError('panel sample sequence mismatch')
        path=run/'panel'/begin['journal']
        if path.parent.resolve()!=(run/'panel').resolve() or not path.is_file():raise ValueError('invalid question journal path')
        raw,row,provenance=e1_baseline.select(proof_bytes,j,inp,sample_row,e1);record=json.loads(raw)
        result=verify_contract.validate(row,record,lines(path),decode)
        if result['baseline_record_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('baseline record replaced')
        question_files[str(path)]=runtime_gate.sha(path);results.append(result)
    out={'status':'passed','stage':'oracle_full','n_questions':128,'new_generations':768,'scoring_performed':False,
         'reference_sources_opened_by_verifier':False,'result_files_sha256':question_files,'run_start_sha256':runtime_gate.sha(a.run_started),
         'run_terminal_sha256':runtime_gate.sha(a.run_terminal),'launch_sha256':runtime_gate.sha(run/'launch.json'),
         'exit_sha256':runtime_gate.sha(run/'exit.json'),'panel_events_sha256':runtime_gate.sha(run/'panel/panel_events.jsonl'),
         'worker_complete_sha256':runtime_gate.sha(run/'panel/worker_complete.json'),'approval_sha256':approval['approval_sha256'],
         'verifier_sha256':runtime_gate.sha(__file__),'scope':'complete DEV oracle inventory, log replay and token redecoding; no scoring; unarchived tensors not independently reconstructed',
         'questions':results}
    with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();__import__('os').fsync(f.fileno())
    return 0
if __name__=='__main__':sys.exit(main())
