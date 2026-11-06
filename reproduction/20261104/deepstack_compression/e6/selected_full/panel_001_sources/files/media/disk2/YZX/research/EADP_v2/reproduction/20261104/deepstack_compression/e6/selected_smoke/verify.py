"""Independent no-GT acceptance for all Div/CD stock/adapter diagnostic arms."""
import argparse,collections,json,os,shutil,subprocess,sys
from pathlib import Path
from freeze_plan import sha
from local_imports import HERE,ROOT,E4,pinned,prepare
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def terminal(run,started,ended):
 s=json.loads(started.read_text());t=json.loads(ended.read_text());launch=json.loads((run/'launch.json').read_text());ex=json.loads((run/'exit.json').read_text())
 cmd=s['command'];cwd=Path(s['cwd']);rid=run.name.removeprefix('smoke_')
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
     s.get('worker_pid')!=t.get('worker_pid') or s.get('worker_pid')!=launch.get('controller',{}).get('pid') or
     [(cwd/x).resolve() for x in cmd if Path(x).name=='launcher.py']!=[(HERE/'launcher.py').resolve()] or
     cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!=rid or
     ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or
     ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or
     ex.get('client_exited_while_container_running') is not False or ex.get('errors')!=[] or ex.get('gpu_pids_after')!=[]):
  raise ValueError('supervisor or container failed')
 alive=pinned('completion',ROOT/'e1_analysis/completion.py').alive
 for ident in (launch['controller'],ex['container_identity']['host_init']):
  if alive(ident):raise ValueError('controller or container init still alive')
 q=subprocess.run(['docker','inspect',ex['container_identity']['container_id']],text=True,capture_output=True,timeout=20,check=True)
 inspector=pinned('eadp_selected_container_identity',HERE/'container_identity.py');inspection=inspector.validate(run,launch,ex,json.loads(q.stdout)[0])
 return launch,ex,inspection
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run id')
 run=HERE/('smoke_'+a.run_id);out=a.out.resolve();started=HERE/('monitor_smoke'+a.run_id)/'started.json';ended=started.with_name('terminal.json')
 stock,contract=prepare();gate=pinned('eadp_selected_gate',HERE/'gate.py');approval=gate.check()
 launch,ex,inspection=terminal(run,started,ended)
 if launch.get('approval')!=approval or launch.get('worker_sha256')!=sha(HERE/'worker.py') or launch.get('launcher_sha256')!=sha(HERE/'launcher.py'):
  raise ValueError('launched source/approval differs')
 frozen=json.loads((run/'frozen.json').read_text());complete=json.loads((run/'worker_complete.json').read_text())
 if (frozen.get('approval')!=approval or frozen.get('GT_opened') is not False or frozen.get('scoring') is not False or
     complete.get('questions')!=3 or complete.get('generations')!=16 or complete.get('paired')!=8 or complete.get('GT_opened') is not False or
     frozen.get('plan_sha256')!=sha(HERE/'plan.json')):raise ValueError('worker completion or frozen metadata mismatch')
 admission=pinned('eadp_selected_e1_admission',ROOT/'m3_oracle_exec/e1_admission.py');admission.validate(metadata_only=False)
 e1=json.loads((ROOT/'e1/e1full001_completion.json').read_text());refs=lines(e1['files']['journal']['path']);inputs=lines(e1['files']['inputs']['path'])
 plan=json.loads((HERE/'plan.json').read_text());anchors=plan['anchors'];records=lines(run/'records.jsonl');generated=lines(run/'generated.jsonl')
 def key(r):return (r.get('method'),r.get('task'),str(r.get('qid')),r.get('group_id'),r.get('path'),r.get('forced'))
 expected={(method,a['task'],str(a['qid']),a['group_id'],path,forced)
           for method in ('div','cd') for a in anchors for path in ('author_stock','adapted')
           for forced in ((False,True) if a['task']=='DocVQA_VAL' else (False,))}
 if (len(expected)!=16 or len(records)!=16 or len(generated)!=16 or any(type(r.get('forced')) is not bool for r in records+generated) or
     {key(r) for r in records}!=expected or {key(r) for r in generated}!=expected):
  raise ValueError('arm inventory incomplete, duplicate, or extra')
 events=lines(run/'events.jsonl')
 for kind in ('arm_started','arm_complete'):
  observed=[key(x) for x in events if x.get('event')==kind]
  if len(observed)!=16 or set(observed)!=expected:raise ValueError('exposure event inventory mismatch: '+kind)
 if not events or events[-1].get('event')!='worker_complete' or any(x.get('event')=='worker_failed' for x in events):
  raise ValueError('worker event termination mismatch')
 bykey={key(r):r for r in records};bygen={key(r):r for r in generated}
 from transformers import AutoTokenizer
 iw=pinned('identity_worker',E4/'identity_worker.py');author=json.loads(iw.PLAN.read_text())
 tokenizer=AutoTokenizer.from_pretrained(author['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
 import numpy as np
 for k,rec in bykey.items():
  marker=bygen[k]
  for field in ('output_ids','eos_ids','generation_scalars','generation_controls'):
   if marker.get(field)!=rec.get(field):raise ValueError('generation journal differs: '+field)
  row=[x for x in inputs if x['task']==rec['task'] and str(x['qid'])==str(rec['qid'])]
  ref=[x for x in refs if x['task']==rec['task'] and str(x['qid'])==str(rec['qid']) and x.get('arm')=='P3-D1' and x.get('forced') is False]
  if len(row)!=1 or len(ref)!=1 or any(rec.get(q)!=row[0].get(q) for q in ('group_id','image_sha256','message_sha256')):raise ValueError('source row/reference differs')
  descriptor=rec['arrays'];path=Path(descriptor['path']).resolve()
  if (descriptor.get('kind')!='full_arrays' or set(descriptor)!={'kind','path','sha256'} or path.parent!=(run/'arrays').resolve() or
      sha(path)!=descriptor['sha256']):raise ValueError('array artifact missing or changed')
  with np.load(path,allow_pickle=False) as array:
   data={name:array[name] for name in array.files}
   if rec['path']=='author_stock':contract.validate_stock(rec,data,tokenizer)
   else:contract.validate_adapted(rec,ref[0],data,tokenizer)
  contract.validate_cache(rec)
 for method in ('div','cd'):
  for anchor in anchors:
   for forced in ((False,True) if anchor['task']=='DocVQA_VAL' else (False,)):
    prefix=(method,anchor['task'],str(anchor['qid']),anchor['group_id'])
    one=bykey[prefix+('author_stock',forced)];two=bykey[prefix+('adapted',forced)];contract.compare_pair(one,two)
    if forced:
     contract.compare_stock_forced(bykey[prefix+('author_stock',False)],one)
     contract.compare_forced(bykey[prefix+('adapted',False)],two)
 src=json.loads(Path(json.loads(gate.APPROVAL.read_text())['cpu_proof']).read_text())['source_sha256_after'];snapshot=HERE/(run.name+'_sources')
 snapshot.mkdir(exist_ok=False)
 for source,digest in src.items():
  path=Path(source);dst=snapshot/'files'/path.relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
  if sha(dst)!=digest or sha(path)!=digest:raise ValueError('source snapshot mismatch')
 with (snapshot/'manifest.json').open('x') as f:json.dump({'source_sha256':src},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 proof={'status':'passed','scope':'16 DEV diagnostic generations, observed author Div/CD selection and separate P1/D0 vs P3/D1 paths; no output equality or accuracy claim',
        'questions':3,'generations':16,'pairs':8,'forced_decode_steps':32,'GT_opened':False,'scoring':False,
        'run':str(run),'inspection':inspection,'source_manifest_sha256':sha(snapshot/'manifest.json'),
        'files_sha256':{str(x):sha(x) for x in (started,ended,run/'frozen.json',run/'worker_complete.json',run/'launch.json',run/'exit.json',run/'records.jsonl',run/'generated.jsonl',run/'events.jsonl',run/'comparisons.jsonl')},
        'verifier_sha256':sha(__file__),'kv_scope':'actual per-layer KV content fingerprints; raw KV tensors not archived'}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0
if __name__=='__main__':sys.exit(main())
