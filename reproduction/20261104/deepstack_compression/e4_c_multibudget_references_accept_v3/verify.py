"""Post-run formal no-GT verifier for all 248 fixed32 references."""
import argparse,importlib.util,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;RUN=BASE/'e4_c_multibudget_references_v3'
def load(name,path):
 p=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=p:raise ValueError('foreign verifier module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
gate=load('_fixed32_reference_accept_gate',RUN/'gate.py')
def arm(segment,job):
 if len(segment)!=4 or [x.get('event') for x in segment]!=['arm_started','raw_generation','arm_captured','arm_completed'] or any(x.get('job')!=job for x in segment):raise ValueError('missing or unordered reference journal')
 raw=segment[1]['raw'];rec=segment[2]['record']
 for key in ('task','qid','group_id','budget','ds_on','forced','protocol','output_ids','eos_ids','generation_scalars','min_new_tokens'):
  if raw.get(key)!=rec.get(key):raise ValueError('raw/result disagreement '+key)
 return rec

def verify(runid):
 if runid!='001':raise ValueError('unplanned run ID')
 out=RUN/'run_001';mon=RUN/'monitor_run001';started=gate.terminal(mon/'started.json',mon/'terminal.json')
 approval=gate.admit_gpu();launch=gate.read(out/'launch.json');exit=gate.read(out/'exit.json');st=exit.get('last_inspect') or {}
 if (started.get('worker_pid')!=(launch.get('controller') or {}).get('pid') or launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=gate.sha(RUN/'worker.py') or launch.get('launcher_sha256')!=gate.sha(RUN/'launch.py') or launch.get('run_out')!=str(out) or exit.get('controller')!=launch.get('controller') or exit.get('returncode')!=0 or exit.get('container_exit_code')!=0 or exit.get('timed_out') is not False or exit.get('interrupted') is not None or exit.get('container_stopped_verified') is not True or exit.get('client_exited_while_container_running') is not False or exit.get('gpu_pids_after')!=[] or st.get('running') is not False or st.get('pid')!=0):raise ValueError('launcher/exit evidence invalid')
 cid=(exit.get('container_identity') or {}).get('container_id')
 if not cid or st.get('id')!=cid:raise ValueError('container identity missing')
 doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=10,check=True).stdout)[0]
 if doc['State']['Running'] is not False or doc['State']['Pid']!=0 or doc['State']['ExitCode']!=0 or doc['Id']!=cid or doc['Image']!=st['image'] or doc['HostConfig']['NetworkMode']!='none':raise ValueError('actual Docker state mismatch')
 if not doc['Config']['Cmd'] or doc['Config']['Cmd'][0]!=str(RUN/'worker.py') or doc['Config']['Cmd'][1:]!=['--out',str(out)]:raise ValueError('actual Docker worker command mismatch')
 frozen=gate.read(out/'frozen.json');complete=gate.read(out/'worker_complete.json');diag=out/'diagnostic';plan=gate.plan()
 snapshot=RUN/'run_001_sources';manifest=snapshot/'manifest.json';gate.terminal(HERE/'monitor_snapshot001/started.json',HERE/'monitor_snapshot001/terminal.json');snap=gate.read(manifest)
 if snap.get('status')!='passed' or snap.get('execution_plan_sha256')!=gate.sha(RUN/'execution_plan001.json') or snap.get('snapshot_script_sha256')!=gate.sha(HERE/'snapshot.py') or set(snap.get('sources',{}))!=set(plan['source_sha256']):raise ValueError('source snapshot inventory mismatch')
 for path,digest in plan['source_sha256'].items():
  item=snap['sources'][path];copy=Path(item['snapshot']).resolve()
  if item.get('source')!=path or item.get('sha256')!=digest or not copy.is_relative_to(snapshot.resolve()) or gate.sha(copy)!=digest:raise ValueError('source snapshot mismatch '+path)
 if (frozen.get('approval')!=approval or frozen.get('source_sha256')!=plan['source_sha256'] or complete.get('status')!='passed' or complete.get('generations')!=248 or complete.get('source_unchanged') is not True or complete.get('weights_stat_unchanged') is not True or complete.get('diagnostic_complete_sha256')!=gate.sha(diag/'worker_complete.json')):raise ValueError('worker completion invalid')
 ref=load('_bulk_accept_references',BASE/'e4_c_multibudget_worker/references.py');refs=ref.load_references();bulk=load('_bulk_accept_core',RUN/'core.py');fixed=load('_bulk_accept_fixed_core',BASE/'e4_c_multibudget_fixed32/core.py')
 if any(frozen.get(key)!=refs[value] for key,value in [('E4_formal_sha256','formal_sha256'),('E4_review_sha256','review_sha256'),('E4_records_sha256','records_sha256')]):raise ValueError('accepted panel reference mismatch')
 from transformers import AutoTokenizer
 model='/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
 tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
 journal=diag/'journal.jsonl';events=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()];records_path=diag/'records.jsonl';records=[json.loads(line) for line in records_path.read_text().splitlines() if line.strip()];jobs=bulk.schedule(refs['rows'])
 if len(jobs)!=248 or len(events)!=992 or len(records)!=248:raise ValueError('reference journal denominator')
 for index,(row,k) in enumerate(jobs):
  job={'order':index,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':k,'protocol':'fixed32'};rec=arm(events[index*4:(index+1)*4],job)
  if rec!=records[index]:raise ValueError('journal/reference records mismatch')
  panel=refs['records'][(row['task'],str(row['qid']),k)];bulk.validate(rec,panel,fixed)
  if tokenizer.decode(rec['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=rec['answer']:raise ValueError('token decode mismatch')
 return {'status':'passed','stage':'four-budget fixed32 reference generation','inputs':62,'generations':248,'output_tokens_per_arm':32,'decode_forwards_after_prefill':31,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'predictions_opened':True,
         'plan_sha256':gate.sha(RUN/'execution_plan001.json'),'run_started_sha256':gate.sha(mon/'started.json'),'run_terminal_sha256':gate.sha(mon/'terminal.json'),'launch_sha256':gate.sha(out/'launch.json'),'exit_sha256':gate.sha(out/'exit.json'),'worker_complete_sha256':gate.sha(out/'worker_complete.json'),
         'snapshot_manifest_sha256':gate.sha(manifest),'snapshot_started_sha256':gate.sha(HERE/'monitor_snapshot001/started.json'),'snapshot_terminal_sha256':gate.sha(HERE/'monitor_snapshot001/terminal.json'),'journal_sha256':gate.sha(journal),'records_sha256':gate.sha(records_path),'verifier_sha256':gate.sha(__file__),'errors':[],
         'scope':'62 accepted DEV inputs; full prefill/DS tensor summaries; prefill KV fingerprints and subsequent decode position/length; no high-dimensional arrays archived'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:
  if a.run_id!='001' or a.out.resolve()!=(HERE/'verification001.json'):raise ValueError('unplanned verifier invocation')
  plan=gate.read(HERE/'verification_plan001.json')
  for path,digest in plan['source_sha256'].items():
   if gate.sha(path)!=digest:raise ValueError('verification source changed '+path)
  report=verify(a.run_id);report['verification_plan_sha256']=gate.sha(HERE/'verification_plan001.json')
 except Exception as exc:report={'status':'failed','errors':[type(exc).__name__+': '+str(exc)],'GT_opened':False,'scoring_performed':False,'verifier_sha256':gate.sha(__file__)}
 with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if report['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
