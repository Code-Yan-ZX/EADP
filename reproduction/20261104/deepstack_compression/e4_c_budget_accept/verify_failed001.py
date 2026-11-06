"""Post-run DEV-only, no-GT 32-generation budget identity verifier."""
import argparse,importlib.util,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;RUN=BASE/'e4_c_budget_execution';V2=BASE/'e4_c_budget_pilot_v2'
def load(name,path):
 p=Path(path).resolve();spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
gate=load('_budget_accept_gate',RUN/'gate.py')
def event(segment,job,side):
 if len(segment)!=3 or [x.get('event') for x in segment]!=['arm_started','raw_generation','arm_completed'] or any(x.get('job')!=job or x.get('side')!=side for x in segment):raise ValueError('missing or unordered arm journal')
 raw=segment[1]['raw'];rec=segment[2]['record']
 for key in ('task','qid','group_id','budget','ds_on','forced','output_ids','eos_ids','generation_scalars'):
  if raw.get(key)!=rec.get(key):raise ValueError('raw/result disagreement '+key)
 return rec
def verify(runid):
 if not runid.isalnum():raise ValueError('invalid run')
 out=RUN/('run_'+runid);mon=RUN/('monitor_run'+runid);gate.terminal(mon/'started.json',mon/'terminal.json')
 approval=gate.admit_gpu();launch=gate.read(out/'launch.json');exit=gate.read(out/'exit.json');st=exit.get('last_inspect') or {}
 if (launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=gate.sha(RUN/'worker.py') or launch.get('launcher_sha256')!=gate.sha(RUN/'launch.py') or
     launch.get('run_out')!=str(out) or exit.get('controller')!=launch.get('controller') or exit.get('returncode')!=0 or exit.get('container_exit_code')!=0 or
     exit.get('timed_out') is not False or exit.get('interrupted') is not None or exit.get('container_stopped_verified') is not True or
     exit.get('client_exited_while_container_running') is not False or exit.get('gpu_pids_after')!=[] or st.get('running') is not False or st.get('pid')!=0):raise ValueError('launcher/exit proof invalid')
 cid=(exit.get('container_identity') or {}).get('container_id')
 if not cid or st.get('id')!=cid:raise ValueError('container identity missing')
 inspected=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=10,check=True);doc=json.loads(inspected.stdout)[0]
 if doc['State']['Running'] is not False or doc['State']['Pid']!=0 or doc['State']['ExitCode']!=0 or doc['Id']!=cid or doc['Image']!=st['image']:raise ValueError('actual Docker state mismatch')
 frozen=gate.read(out/'frozen.json');complete=gate.read(out/'worker_complete.json');diag=out/'diagnostic';actual_plan=gate.plan()
 snapshot=RUN/('run_'+runid+'_sources');manifest=snapshot/'manifest.json'
 gate.terminal(HERE/'monitor_snapshot001/started.json',HERE/'monitor_snapshot001/terminal.json');snap=gate.read(manifest)
 if (snap.get('status')!='passed' or snap.get('execution_plan_sha256')!=gate.sha(RUN/'execution_plan001.json') or
     snap.get('snapshot_script_sha256')!=gate.sha(HERE/'snapshot.py') or set(snap.get('sources',{}))!=set(actual_plan['source_sha256'])):raise ValueError('source snapshot inventory mismatch')
 for path,digest in actual_plan['source_sha256'].items():
  item=snap['sources'][path];copy=Path(item['snapshot']).resolve()
  if item.get('source')!=path or item.get('sha256')!=digest or not copy.is_relative_to(snapshot.resolve()) or gate.sha(copy)!=digest:raise ValueError('source snapshot mismatch '+path)
 if (frozen.get('approval')!=approval or frozen.get('source_sha256')!=actual_plan['source_sha256'] or complete.get('status')!='passed' or
     complete.get('generations')!=32 or complete.get('natural_pairs')!=12 or complete.get('forced_pairs')!=4 or
     complete.get('source_unchanged') is not True or complete.get('weights_stat_unchanged') is not True or
     complete.get('diagnostic_complete_sha256')!=gate.sha(diag/'worker_complete.json')):raise ValueError('worker completion invalid')
 reference=load('_budget_accept_reference',V2/'reference.py');refs=reference.load_references();core=load('_budget_accept_core',V2/'diagnostic_core.py');compare=load('_budget_accept_compare',V2/'budget_compare.py')
 if any(frozen.get(key)!=refs[value] for key,value in [('E4_formal_sha256','formal_sha256'),('E4_review_sha256','review_sha256'),('E4_records_sha256','records_sha256')]):raise ValueError('reference mismatch')
 from transformers import AutoTokenizer
 model='/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
 tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False);fixed=tokenizer.encode(' the',add_special_tokens=False)
 if not fixed or complete.get('forced_token')!=fixed[0]:raise ValueError('forced tokenizer identity')
 journal=diag/'journal.jsonl';events=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()];jobs=core.schedule(refs['rows'])
 if len(events)!=len(jobs)*7:raise ValueError('journal denominator')
 arrays={}
 for i,job in enumerate(jobs):
  seg=events[i*7:(i+1)*7];fresh=event(seg[:3],job,'fresh');light=event(seg[3:6],job,'light');pair=seg[6]
  if pair.get('event')!='pair_verified' or pair.get('job')!=job or (pair.get('comparison') or {}).get('status')!='passed':raise ValueError('pair proof missing')
  if not job['forced']:core.compare_panel(fresh,refs[(job['task'],job['qid'],job['budget'])])
  for rec in (fresh,light):
   if rec.get('forced_token')!=(complete['forced_token'] if job['forced'] else None):raise ValueError('forced token mismatch')
   decoded=tokenizer.decode(rec['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
   if decoded!=rec['answer']:raise ValueError('token decode mismatch')
   path=Path(rec['arrays']['path']).resolve()
   if not path.is_relative_to((diag/'arrays').resolve()) or str(path) in arrays:raise ValueError('array path invalid/duplicate')
   arrays[str(path)]=gate.sha(path)
  compare.compare_metadata(light,fresh);compare.compare_archives(light,fresh)
 if len(arrays)!=32:raise ValueError('array denominator')
 return {'status':'passed','stage':'four-budget identity','pairs':16,'generations':32,'natural_pairs':12,'forced_pairs':4,
  'forced_decode_steps':32,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'predictions_opened':True,
  'plan_sha256':gate.sha(RUN/'execution_plan001.json'),'run_started_sha256':gate.sha(mon/'started.json'),'run_terminal_sha256':gate.sha(mon/'terminal.json'),
  'launch_sha256':gate.sha(out/'launch.json'),'exit_sha256':gate.sha(out/'exit.json'),'worker_complete_sha256':gate.sha(out/'worker_complete.json'),
  'snapshot_manifest_sha256':gate.sha(manifest),'snapshot_started_sha256':gate.sha(HERE/'monitor_snapshot001/started.json'),'snapshot_terminal_sha256':gate.sha(HERE/'monitor_snapshot001/terminal.json'),
  'journal_sha256':gate.sha(journal),'array_sha256':arrays,'verifier_sha256':gate.sha(__file__),'errors':[]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 try:
  if args.run_id!='001' or args.out.resolve()!=(HERE/'verification001.json'):raise ValueError('unplanned verification invocation')
  verification_plan=gate.read(HERE/'verification_plan002.json')
  for path,digest in verification_plan['source_sha256'].items():
   if gate.sha(path)!=digest:raise ValueError('verification source changed '+path)
  report=verify(args.run_id);report['verification_plan_sha256']=gate.sha(HERE/'verification_plan002.json')
 except Exception as exc:
  report={'status':'failed','errors':[type(exc).__name__+': '+str(exc)],'GT_opened':False,'scoring_performed':False,'verifier_sha256':gate.sha(__file__)}
 with args.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
