"""Acceptance gate. This module never opens any dataset reference answers."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
E1=HERE.parent/'e1'
sys.path.insert(0,str(E1))
from contract import ARMS, validate_quartet

TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
PROOF_FILES=frozenset({'frozen','worker_complete','journal','generated_arms','quartets','panel','terminal',
    'container_exit','launch','inputs','input_proof','contract','runner','launcher_source','position_adapter',
    'e0_adapter','plan','tokenizer_tokenizer.json','tokenizer_tokenizer_config.json','tokenizer_vocab.json',
    'dataset_summary','dataset_verification','dataset_builder','dataset_verifier'})
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read_jsonl(p):
    raw=Path(p).read_bytes()
    if raw and not raw.endswith(b'\n'):raise ValueError('incomplete JSONL tail')
    return [json.loads(line) for line in raw.splitlines()]
def alive(identity):
    if not identity or not isinstance(identity,dict):return False
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if identity.get('boot_id')!=boot:return False
    try:
        raw=Path('/proc',str(identity['pid']),'stat').read_text()
        fields=raw[raw.rfind(')')+2:].split()
        return int(fields[19])==identity['start_ticks'] and fields[0]!='Z'
    except (OSError,ValueError,IndexError,KeyError):return False

def validate_terminal(terminal,exited,launch=None,run_dir=None):
    if terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[]:
        raise ValueError('supervisor did not prove clean completion')
    if ('interrupted' not in exited or exited.get('returncode')!=0 or exited.get('timed_out') is not False or exited.get('interrupted') is not None):
        raise ValueError('launcher/container returned failure or was interrupted')
    if exited.get('client_exited_while_container_running') is not False:
        raise ValueError('docker client may have exited before the container')
    if exited.get('container_stopped_verified') is not True or exited.get('gpu_pids_after')!=[]:
        raise ValueError('container stop or released GPU was not proven')
    ident=exited.get('container_identity')
    if not isinstance(ident,dict) or not ident.get('container_id') or not ident.get('host_init'):
        raise ValueError('missing observed container identity')
    for process in (ident['host_init'],exited.get('controller')):
        if (not isinstance(process,dict) or not isinstance(process.get('pid'),int) or process['pid']<=0 or
            not isinstance(process.get('start_ticks'),int) or process['start_ticks']<=0 or
            not isinstance(process.get('boot_id'),str) or not process['boot_id']):
            raise ValueError('incomplete original process identity')
    if alive(ident['host_init']) or alive(exited.get('controller')):
        raise ValueError('recorded original container/controller still alive')
    if launch is not None:
        before=launch.get('controller'); after=exited.get('controller')
        if not before or not after or any(before.get(k)!=after.get(k) for k in ('pid','start_ticks','boot_id')):
            raise ValueError('launcher and exit identity differ')
        if terminal.get('worker_pid')!=before.get('pid'):
            raise ValueError('supervisor did not wait for this launcher')
        command=launch.get('command',[])
        def arg(name):
            if command.count(name)!=1:raise ValueError('launch command missing or repeated '+name)
            return command[command.index(name)+1]
        if arg('--stage')!='full' or Path(arg('--out')).resolve()!=Path(run_dir).resolve():
            raise ValueError('launch command did not target this full run')
        if arg('--name')!=exited.get('name'):raise ValueError('container name mismatch')

def validate_records(panel,records,completed_quartets,expected_per_task=128):
    dev=[r for r in panel if r.get('split')=='DEV']
    if Counter(x['task'] for x in dev)!={task:expected_per_task for task in TASKS}:
        raise ValueError('DEV panel counts differ')
    keys={(r['task'],str(r['qid'])):r for r in dev}
    if len(keys)!=len(dev) or len({r['group_id'] for r in dev})!=len(dev):raise ValueError('duplicate DEV id/group')
    grouped=defaultdict(list)
    for r in records:
        key=(r['task'],str(r['qid']))
        if key not in keys or r['group_id']!=keys[key]['group_id']:raise ValueError('record outside frozen DEV panel')
        if r.get('forced') is not False:raise ValueError('full panel must contain natural arms only')
        grouped[(key,r['forced'])].append(r)
    natural=[]
    for key in keys:
        items=grouped.get((key,False),[])
        if len(items)!=4:raise ValueError('missing or repeated natural arm')
        if [x['arm'] for x in items]!=list(ARMS):raise ValueError('natural arm order/inventory')
        validate_quartet(items)
        for r in items:
            if not isinstance(r.get('eos'),bool) or not isinstance(r.get('truncated'),bool) or not isinstance(r.get('eos_ids'),list):
                raise ValueError('missing EOS/truncation evidence')
            observed=bool(r['output_ids'] and r['output_ids'][-1] in r['eos_ids'])
            if r['eos']!=observed or r['truncated']!=bool(not observed and len(r['output_ids'])>=2048):
                raise ValueError('EOS or length state inconsistent')
            if not r['output_ids'] or len(r['output_ids'])>2048 or not (r['eos'] or r['truncated']):
                raise ValueError('natural generation has no valid terminal token or length')
        natural.extend(items)
    marker=Counter((r['task'],str(r['qid']),r['group_id'],r.get('forced')) for r in completed_quartets)
    expected={(r['task'],str(r['qid']),r['group_id'],False) for r in dev}
    if set(marker)!=expected or any(marker[x]!=1 for x in expected):
        raise ValueError('natural quartet acceptance markers absent, duplicated or unexpected')
    return natural

def validate_generated(records,generated):
    if len(records)!=len(generated):raise ValueError('generation and observation journal lengths differ')
    keys=('task','qid','group_id','arm','forced','output_ids','eos_ids')
    for rec,observed in zip(records,generated):
        if observed!={k:rec[k] for k in keys}:raise ValueError('generation and observation journals differ')

def validate_decoding(natural,tokenizer):
    for row in natural:
        decoded=tokenizer.decode(row['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        if decoded!=row['answer']:raise ValueError('independent token decoding mismatch')

def validate_sources(frozen,launch,run):
    """Bind the actual full worker and launcher to their pre-run snapshots."""
    snapshot=Path(run).parent/(Path(run).name+'_sources')
    specs=(('runner_sha256',E1/'full_runner.py','full_runner.py'),
           ('contract_sha256',E1/'contract.py','contract.py'),
           ('position_adapter_sha256',E1/'position_adapter.py','position_adapter.py'),
           ('e0_adapter_sha256',HERE.parent/'e0_identity/adapter.py','adapter.py'),
           (None,E1/'full_launch.py','full_launch.py'))
    source_hashes=launch.get('source_sha256',{});paths={}
    for field,original,name in specs:
        path=snapshot/name
        if not path.is_file():raise ValueError('missing launch source snapshot: '+name)
        digest=sha(path)
        if source_hashes.get(str(original))!=digest or (field is not None and frozen.get(field)!=digest):
            raise ValueError('launch/frozen/snapshot source mismatch: '+name)
        paths[name]=path
    if launch.get('command',[]).count(str(E1/'full_runner.py'))!=1:
        raise ValueError('launch command did not invoke the full worker')
    return paths

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--terminal',type=Path,required=True);p.add_argument('--exit',type=Path,required=True)
    p.add_argument('--launch',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--input-proof',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    run=a.run_dir.resolve();panel_path=a.panel.resolve();terminal_path=a.terminal.resolve();exit_path=a.exit.resolve()
    terminal=json.loads(terminal_path.read_text());exited=json.loads(exit_path.read_text());launch=json.loads(a.launch.read_text())
    validate_terminal(terminal,exited,launch,run)
    frozen=json.loads((run/'frozen.json').read_text());done=json.loads((run/'worker_complete.json').read_text())
    if frozen.get('stage')!='full' or frozen.get('n_questions')!=384 or done.get('n')!=384:
        raise ValueError('not a complete full 384-question run')
    if frozen.get('panel_sha256')!=sha(panel_path):raise ValueError('frozen panel mismatch')
    manifest_summary=panel_path.parent/'manifest_summary.json';manifest_verification=panel_path.parent/'verification.json'
    manifest=json.loads(manifest_summary.read_text());audit=json.loads(manifest_verification.read_text())
    if (manifest.get('outputs_sha256',{}).get('panel_manifest.jsonl')!=sha(panel_path) or audit.get('status')!='pass' or
        audit.get('builder_sha256')!=sha(panel_path.parent/'build.py') or
        audit.get('verifier_sha256')!=sha(panel_path.parent/'verify.py')):
        raise ValueError('dataset manifest verification failed')
    if frozen.get('inputs_sha256')!=sha(a.inputs) or frozen.get('input_proof_sha256')!=sha(a.input_proof):
        raise ValueError('run input/proof identity mismatch')
    input_proof=json.loads(a.input_proof.read_text())
    if (input_proof.get('status')!='passed' or input_proof.get('n_verified')!=384 or
        input_proof.get('inputs_sha256')!=sha(a.inputs) or input_proof.get('panel_sha256')!=sha(panel_path) or
        input_proof.get('verifier_sha256')!=sha(E1/'verify_inputs.py')):
        raise ValueError('input proof is inconsistent')
    if frozen.get('gt_read') is not False or frozen.get('scoring') is not False:raise ValueError('run did not preserve score separation')
    source_snapshots=validate_sources(frozen,launch,run)
    panel=read_jsonl(panel_path);journal=run/'raw_predictions.jsonl';markers=run/'validated_quartets.jsonl'
    records=read_jsonl(journal); natural=validate_records(panel,records,read_jsonl(markers))
    generated=run/'generated_arms.jsonl';validate_generated(records,read_jsonl(generated))
    inputs=read_jsonl(a.inputs);by_q={(r['task'],str(r['qid'])):r for r in inputs}
    if len(inputs)!=384 or len(by_q)!=384:raise ValueError('input inventory mismatch')
    for r in natural:
        item=by_q[(r['task'],str(r['qid']))]
        if r['group_id']!=item['group_id'] or r['message_sha256']!=item['message_sha256']:
            raise ValueError('record does not match frozen question input')
    # Greedy decoder only: this imports no model weights and never touches GT.
    plan_path=HERE.parents[2]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
    if frozen.get('plan_sha256')!=sha(plan_path):raise ValueError('run plan identity mismatch')
    plan=json.loads(plan_path.read_text());recipe=plan['recipe']
    if recipe.get('post_process',False) is not False:raise ValueError('unsupported model post-processing')
    tokenizer_sources={p:d for p,d in plan['source_sha256'].items() if Path(p).parent==Path(recipe['model_path']) and Path(p).name in ('tokenizer.json','tokenizer_config.json','vocab.json')}
    if len(tokenizer_sources)!=3 or any(sha(path)!=digest for path,digest in tokenizer_sources.items()):
        raise ValueError('tokenizer files differ from run plan')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
    validate_decoding(natural,tokenizer)
    files={'frozen':run/'frozen.json','worker_complete':run/'worker_complete.json','journal':journal,
           'quartets':markers,'generated_arms':generated,'panel':panel_path,'terminal':terminal_path,'container_exit':exit_path,
           'launch':a.launch.resolve(),'inputs':a.inputs.resolve(),'input_proof':a.input_proof.resolve(),
           'contract':source_snapshots['contract.py'],'runner':source_snapshots['full_runner.py'],
           'launcher_source':source_snapshots['full_launch.py'],'position_adapter':source_snapshots['position_adapter.py'],
           'e0_adapter':source_snapshots['adapter.py'],'plan':plan_path,'dataset_summary':manifest_summary,
           'dataset_verification':manifest_verification,'dataset_builder':panel_path.parent/'build.py',
           'dataset_verifier':panel_path.parent/'verify.py'}
    for path in tokenizer_sources:files['tokenizer_'+Path(path).name]=Path(path)
    if set(files)!=PROOF_FILES:raise ValueError('internal acceptance file inventory mismatch')
    proof={'status':'passed','stage':'full','natural_questions':384,'natural_records':len(natural),
           'files':{k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()},
           'verifier_sha256':sha(Path(__file__)),'tokenizer_decode_checked':len(natural),'GT_opened':False,'scoring_performed':False}
    with a.out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'status':'passed','natural_records':len(natural)}))
if __name__=='__main__':main()
