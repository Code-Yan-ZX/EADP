"""Post-completion acceptance: 384 paired DEV questions, never scores GT."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from full_imports import HERE,SMOKE,load,prepare
ROOT=HERE.parents[1];E4=ROOT/'e4';sha=load('eadp_selected_full_plan',HERE/'freeze_plan.py').sha
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def require_completion(frozen,complete,approval,plan_sha):
 if (frozen.get('stage')!='full_dev' or frozen.get('approval')!=approval or frozen.get('GT_opened') is not False or
     frozen.get('scoring') is not False or frozen.get('full_tensors_archived') is not False or frozen.get('plan_sha256')!=plan_sha or
     complete.get('status')!='completed' or complete.get('questions')!=384 or complete.get('generations')!=1536 or complete.get('pairs')!=768 or
     complete.get('GT_opened') is not False or complete.get('scoring') is not False):raise ValueError('full worker completion mismatch')
def terminal(run,started,ended):
 s=json.loads(started.read_text());t=json.loads(ended.read_text());launch=json.loads((run/'launch.json').read_text());ex=json.loads((run/'exit.json').read_text())
 cmd=s['command'];cwd=Path(s['cwd']);rid=run.name.removeprefix('panel_')
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
     s.get('worker_pid')!=t.get('worker_pid') or s.get('worker_pid')!=launch.get('controller',{}).get('pid') or
     [(cwd/x).resolve() for x in cmd if Path(x).name=='launcher.py']!=[(HERE/'launcher.py').resolve()] or
     cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!=rid or
     ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or
     ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or
     ex.get('client_exited_while_container_running') is not False or ex.get('errors')!=[] or ex.get('gpu_pids_after')!=[]):
  raise ValueError('panel supervisor/container failed')
 alive=load('completion',ROOT/'e1_analysis/completion.py').alive
 for ident in (launch['controller'],ex['container_identity']['host_init']):
  if alive(ident):raise ValueError('panel controller or container still alive')
 q=subprocess.run(['docker','inspect',ex['container_identity']['container_id']],text=True,capture_output=True,timeout=20,check=True)
 inspection=load('eadp_selected_full_container_identity',HERE/'container_identity.py').validate(run,launch,ex,json.loads(q.stdout)[0])
 return launch,ex,inspection
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run id')
 run=HERE/('panel_'+a.run_id);out=a.out.resolve();started=HERE/('monitor_panel'+a.run_id)/'started.json';ended=started.with_name('terminal.json')
 _stock,smoke_contract=prepare();contract=load('eadp_selected_full_contract',HERE/'full_contract.py');gate=load('eadp_selected_full_gate',HERE/'gate.py')
 approval=gate.check(host=True);launch,ex,inspection=terminal(run,started,ended)
 if launch.get('approval')!=approval or launch.get('worker_sha256')!=sha(HERE/'worker.py') or launch.get('launcher_sha256')!=sha(HERE/'launcher.py'):
  raise ValueError('launched source or review differs')
 frozen=json.loads((run/'frozen.json').read_text());complete=json.loads((run/'worker_complete.json').read_text())
 require_completion(frozen,complete,approval,sha(HERE/'plan.json'))
 admission=load('eadp_full_e1_admission',ROOT/'m3_oracle_exec/e1_admission.py');e1=admission.validate(metadata_only=False)
 ep=json.loads((ROOT/'e1/e1full001_completion.json').read_text());refs=lines(ep['files']['journal']['path']);inputs=lines(ep['files']['inputs']['path'])
 if frozen.get('e1_completion_sha256')!=e1.proof_sha or frozen.get('weights_identity_sha256')!=e1.weights_sha:raise ValueError('E1 provenance differs')
 lookup={}
 for x in refs:
  if x.get('arm')!='P3-D1' or x.get('forced') is not False:continue
  key=(x['task'],str(x['qid']),x['group_id'])
  if key in lookup:raise ValueError('duplicate E1 baseline reference')
  lookup[key]=x
 if len(inputs)!=384 or len(lookup)!=384:raise ValueError('E1 full inventory')
 expected={(method,row['task'],str(row['qid']),row['group_id'],path,False)
           for row in inputs for method in ('div','cd') for path in ('author_stock','adapted')}
 def key(x):return (x.get('method'),x.get('task'),str(x.get('qid')),x.get('group_id'),x.get('path'),x.get('forced'))
 records=lines(run/'records.jsonl');markers=lines(run/'generated.jsonl');events=lines(run/'events.jsonl');pairs=lines(run/'pairs.jsonl')
 if (len(expected)!=1536 or len(records)!=1536 or len(markers)!=1536 or len(pairs)!=768 or
     any(type(x.get('forced')) is not bool for x in records+markers) or {key(x) for x in records}!=expected or {key(x) for x in markers}!=expected):
  raise ValueError('full arm inventory incomplete or extra')
 for event in ('arm_started','arm_completed'):
  found=[key(x) for x in events if x.get('event')==event]
  if len(found)!=1536 or set(found)!=expected:raise ValueError('incomplete exposure events: '+event)
 if not events or events[-1].get('event')!='worker_completed' or any(x.get('event')=='worker_failed' for x in events):raise ValueError('worker event termination')
 bykey={key(r):r for r in records};bygen={key(r):r for r in markers};rows={(r['task'],str(r['qid']),r['group_id']):r for r in inputs}
 from transformers import AutoTokenizer
 iw=load('identity_worker',E4/'identity_worker.py');author=json.loads(iw.PLAN.read_text())
 tokenizer=AutoTokenizer.from_pretrained(author['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
 expected_controls=load('eadp_selected_full_generation_reference',HERE/'generation_reference.py').controls()
 for k,rec in bykey.items():
  marker=bygen[k];row=rows[k[1:4]];reference=lookup[k[1:4]]
  if any(marker.get(x)!=rec.get(x) for x in ('output_ids','eos_ids','generation_scalars','generation_controls')):raise ValueError('raw output journal mismatch')
  if any(rec.get(x)!=row.get(x) for x in ('group_id','image_sha256','message_sha256')):raise ValueError('frozen input differs')
  if rec.get('generation_controls')!=expected_controls[k[0]][k[1]]:raise ValueError('actual generation controls differ from admitted smoke')
  if rec['path']=='author_stock':contract.validate_stock(rec,reference,tokenizer)
  else:contract.validate_adapted(rec,reference,tokenizer)
 for row in inputs:
  for method in ('div','cd'):
   prefix=(method,row['task'],str(row['qid']),row['group_id'])
   smoke_contract.compare_pair(bykey[prefix+('author_stock',False)],bykey[prefix+('adapted',False)])
 expected_pairs={(m,r['task'],str(r['qid']),r['group_id']) for r in inputs for m in ('div','cd')}
 if {(x.get('method'),x.get('task'),str(x.get('qid')),x.get('group_id')) for x in pairs}!=expected_pairs or any(x.get('status')!='passed' for x in pairs):
  raise ValueError('paired selector journal differs')
 src=json.loads(Path(json.loads(gate.APPROVAL.read_text())['cpu_proof']).read_text())['source_sha256_after'];snapshot=HERE/(run.name+'_sources');snapshot.mkdir(exist_ok=False)
 for source,digest in src.items():
  path=Path(source);dst=snapshot/'files'/path.relative_to('/');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
  if sha(dst)!=digest or sha(path)!=digest:raise ValueError('source snapshot mismatch')
 with (snapshot/'manifest.json').open('x') as f:json.dump({'source_sha256':src},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 proof={'status':'passed','stage':'full_dev','questions':384,'generations':1536,'pairs':768,'GT_opened':False,'scoring':False,
        'run':str(run),'inspection':inspection,'source_manifest_sha256':sha(snapshot/'manifest.json'),
        'files_sha256':{str(x):sha(x) for x in (started,ended,run/'frozen.json',run/'worker_complete.json',run/'launch.json',run/'exit.json',
                                                run/'records.jsonl',run/'generated.jsonl',run/'events.jsonl',run/'pairs.jsonl')},
        'verifier_sha256':sha(__file__),'scope':'runtime checks, tensor summaries, selected IDs, positions, KV lengths and output replay; full tensors are not archived or independently reconstructed'}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0
if __name__=='__main__':sys.exit(main())
