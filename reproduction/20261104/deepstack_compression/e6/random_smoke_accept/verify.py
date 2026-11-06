"""Draft pre-run-frozen, no-GT random smoke result verifier.

Must be separately CPU-tested and independently reviewed before use. Invocation
fails closed unless actual model run and GPU admission have already passed.
"""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,types
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;B=A.parents[1];D=B/'e6/random_smoke';R=B/'e1_independent_review';O=D/'model_run001'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load(name,path):
 path=Path(path).resolve()
 if name in sys.modules:raise ValueError('preloaded verifier module '+name)
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
 try:spec.loader.exec_module(mod)
 except BaseException:del sys.modules[name];raise
 return mod
def terminal(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]):raise ValueError('actual supervised exit '+str(d))
 return s
def validate_docker(doc,command):
 def args(flag):return [command[i+1] for i,x in enumerate(command[:-1]) if x==flag]
 ent=args('--entrypoint');user=args('--user');work=args('--workdir');name=args('--name')
 if command[:2]!=['docker','run'] or any(len(x)!=1 for x in (ent,user,work,name)):raise ValueError('container command')
 idx=command.index('--entrypoint');cfg=doc['Config']
 if doc['Image']!=command[idx+2] or cfg['Entrypoint']!=ent or cfg['Cmd']!=command[idx+3:] or cfg['Cmd']!=[str(D/'model_worker.py'),'--out',str(O)] or cfg['User']!=user[0] or cfg['WorkingDir']!=work[0] or doc['Name'].lstrip('/')!=name[0]:raise ValueError('actual Docker identity')
 env=dict(x.split('=',1) for x in cfg['Env'] if '=' in x)
 for item in args('--env'):
  k,v=item.split('=',1)
  if env.get(k)!=v:raise ValueError('docker env')
 mounts={}
 for item in args('--mount'):
  parts=item.split(',');v=dict(x.split('=',1) for x in parts if '=' in x)
  if v.get('type')!='bind':raise ValueError('mount type')
  mounts[v['dst']]=(v['src'],'readonly' not in parts)
 actual={x['Destination']:(x['Source'],x['RW']) for x in doc['Mounts'] if x['Type']=='bind'}
 if mounts!=actual:raise ValueError('mount mapping')
 devices={x.split(':')[0] for x in args('--device')};real={x['PathOnHost'] for x in doc['HostConfig']['Devices'] or []}
 if devices!=real or not any('/dev/nvidia' in x for x in real):raise ValueError('actual GPU devices')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 if args.out.resolve()!=A/'verification001.json' or args.out.exists():raise ValueError('unplanned result path')
 verification=read(A/'verification_plan001.json')
 for path,digest in verification['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('acceptance source changed '+path)
 if verification['execution_plan_sha256']!=sha(D/'execution_plan009.json'):raise ValueError('execution plan mismatch')
 gate=load('_random_accept_gate',D/'smoke_gate.py');approval=gate.admit(False);started=terminal(D/'monitor_model_run001',D/'model_launch.py');launch=read(O/'launch.json');ex=read(O/'exit.json')
 if launch.get('approval_bundle')!=approval or launch.get('controller',{}).get('pid')!=started['worker_pid'] or launch.get('worker_sha256')!=sha(D/'model_worker.py') or launch.get('launcher_sha256')!=sha(D/'model_launch.py') or launch.get('run_out')!=str(O):raise ValueError('run launcher identity')
 cid=ex['container_identity']['container_id'];doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15,check=True).stdout)[0]
 if (ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[] or doc['Id']!=cid or doc['State']['Running'] is not False or doc['State']['Pid']!=0 or doc['State']['ExitCode']!=0 or doc['State']['OOMKilled'] is not False or doc['HostConfig']['NetworkMode']!='none'):raise ValueError('actual run or container exit')
 validate_docker(doc,launch['command'])
 runplan=read(D/'execution_plan009.json');ready=read(O/'model_ready.json');provenance=read(O/'completion_provenance.json')
 if provenance.get('status')!='passed' or provenance.get('source_sha256')!=runplan['source_sha256'] or ready.get('weight_stats_before')!=provenance.get('weight_stats_before') or provenance.get('weight_stats_before')!=provenance.get('weight_stats_after') or len(provenance.get('weight_stats_before',{}))!=4 or ready.get('timing_eligible') is not False:raise ValueError('model provenance')
 weight_manifest=read(B.parents[2]/'reproduction/20261028/qwen/qwen_weight_identity.json');expected={str(Path(x['path']).resolve()):x['bytes'] for x in weight_manifest['files']}
 actual={str(Path(x).resolve()):v[2] for x,v in provenance['weight_stats_before'].items()}
 if actual!=expected:raise ValueError('weight files/size')
 author_plan=read(B.parents[2]/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')
 if ready.get('recipe')!=author_plan['recipe'] or ready.get('versions')!=author_plan['expected_versions']:raise ValueError('recipe/version identity')
 e1=read(B/'e1/e1full001_completion.json');review=read(R/'e1full001_review.json');terminal(B/'e1/monitorverifye1full001',B/'e1_analysis/completion.py');terminal(R/'monitor_e1full001',R/'review_full.py')
 if e1.get('status')!='passed' or review.get('status')!='passed' or review.get('completion_proof_sha256')!=sha(B/'e1/e1full001_completion.json') or review.get('reviewer_sha256')!=sha(R/'review_full.py'):raise ValueError('E1 acceptance')
 random_plan=read(D/'plan001.json');anchors=random_plan['anchors'];refs=load('_random_accept_refs',D/'e1_refs.py').load(B/'e1/e1full001/raw_predictions.jsonl',types.SimpleNamespace(journal_sha=e1['files']['journal']['sha256']),anchors);choices=load('_random_accept_manifest',D/'manifest_load.py').load(D/'selection_manifest001.jsonl',random_plan)
 def lines(path):
  with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]
 journal=lines(O/'journal.jsonl');generated=lines(O/'generated.jsonl');records=lines(O/'records.jsonl');complete=read(O/'worker_complete.json');load('_random_accept_inventory',A/'inventory_contract.py').validate(anchors,journal,generated,records,complete)
 contract=load('_random_accept_contract',A/'contract.py');array_sha={}
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 for rec in records:
  key=(rec['task'],str(rec['qid']),rec['seed']);path=Path(rec['arrays']['path']).resolve()
  if not path.is_relative_to((O/'arrays').resolve()) or str(path) in array_sha or rec['arrays']['kind']!='full_arrays':raise ValueError('archive path')
  digest=sha(path)
  if digest!=rec['arrays']['sha256']:raise ValueError('archive file SHA')
  with np.load(path,allow_pickle=False) as arrays:contract.validate_record(rec,choices[key],refs[key[:2]][0],arrays)
  if tokenizer.decode(rec['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=rec['answer']:raise ValueError('tokenizer decode')
  array_sha[str(path)]=digest
 if len(array_sha)!=18:raise ValueError('array denominator')
 gate.plan()
 files={str(p):sha(p) for p in (O/'launch.json',O/'exit.json',O/'model_ready.json',O/'completion_provenance.json',O/'journal.jsonl',O/'generated.jsonl',O/'records.jsonl',O/'worker_complete.json')}
 proof={'status':'passed','questions':3,'arms':18,'natural':9,'forced33':9,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'timing_eligible':False,'verification_plan_sha256':sha(A/'verification_plan001.json'),'verifier_sha256':sha(__file__),'execution_plan_sha256':sha(D/'execution_plan009.json'),'run_started_sha256':sha(D/'monitor_model_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_model_run001/terminal.json'),'files_sha256':files,'array_sha256':array_sha,'errors':[],'limitations':['3 posthoc DEV anchors only','author greedy verification occurs online, its separate actual indices are not archived','forced KV contents are fingerprints, not archived KV tensors','weight stats cannot exclude transient replacement','model and archive hooks make timing incomparable']}
 with args.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('passed')
if __name__=='__main__':main()
