"""CPU acceptance of a completed 16-arm identity run; no GT or scoring."""
import argparse,json,os,subprocess,time
from pathlib import Path
from real_worker import PLAN,sha
from light_admission import admit,terminal
from light_verify_core import verify_records,lines

def validate_completed(out,started,terminal_path):
 out=Path(out).resolve();s=terminal(started,terminal_path)
 launch=json.loads((out/'launch.json').read_text());ex=json.loads((out/'exit.json').read_text());frozen=json.loads((out/'frozen.json').read_text());done=json.loads((out/'worker_complete.json').read_text())
 approval=admit();controller=launch.get('controller') or {};container=ex.get('container_identity') or {};host=container.get('host_init') or {}
 if (launch.get('approval_bundle')!=approval or frozen.get('source_sha256')!=approval['source_sha256'] or
     frozen.get('references_sha256')!=approval['reference_sha256'] or frozen.get('references_path')!=approval['reference_path'] or
     frozen.get('gt_read') is not False or frozen.get('scoring') is not False or frozen.get('timing_claims') is not False or
     done.get('status')!='completed' or done.get('jobs')!=16 or controller.get('pid')!=s.get('worker_pid') or ex.get('controller')!=controller):raise ValueError('run identity or completion mismatch')
 if (ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or
     ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or
     ex.get('failure_type') is not None or ex.get('gpu_pids_after')!=[] or not container.get('container_id') or
     type(host.get('pid')) is not int or type(host.get('start_ticks')) is not int or not host.get('boot_id')):raise ValueError('container exit proof incomplete')
 check=subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True);item=json.loads(check.stdout)[0];state=item['State'];plan=json.loads(PLAN.read_text())
 if (item['Id']!=container['container_id'] or item['Image']!=container['image'] or item['Image']!=plan['container_image'] or
     state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0):raise ValueError('actual container state mismatch')
 if Path('/proc/sys/kernel/random/boot_id').read_text().strip()==host['boot_id']:
  try:ticks=int(Path('/proc',str(host['pid']),'stat').read_text().rsplit(')',1)[1].split()[19])
  except FileNotFoundError:ticks=None
  if ticks==host['start_ticks']:raise ValueError('original container process still exists')
 return approval,done

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--started',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();begin=time.time()
 try:
  approval,done=validate_completed(args.run_dir,args.started,args.terminal)
  from transformers import AutoTokenizer
  plan=json.loads(PLAN.read_text());tok=AutoTokenizer.from_pretrained(plan['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
  forced_token=tok.encode(' the',add_special_tokens=False)[0]
  if done.get('forced_token')!=forced_token:raise ValueError('forced token differs from tokenizer')
  ref=json.loads(Path(approval['reference_path']).read_text());result=verify_records(args.run_dir,ref,forced_token)
  records=lines(args.run_dir/'records.jsonl')
  for item in records:
   record=item['record']
   if tok.decode(record['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=record['answer']:raise ValueError('answer/token decode mismatch')
  evidence=[args.run_dir/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','generated.jsonl','records.jsonl','comparisons.jsonl')]
  evidence.extend([args.started,args.terminal,Path(approval['reference_path'])])
  arrays={x['record']['arrays']['path']:x['record']['arrays']['sha256'] for x in records}
  result.update(errors=[],evidence_sha256={str(x.resolve()):sha(x) for x in evidence},array_sha256=arrays,source_sha256=approval['source_sha256'],
                verifier_sha256=sha(__file__),forced_token=forced_token,finished_unix=time.time(),
                limitations=['three DEV anchors; no general performance or quality claim','old ON reference lacks KV fingerprints; fresh ON to lightweight ON compares KV fingerprints',
                             'KV content represented by fingerprints, not archived full KV tensors'])
 except Exception as exc:
  result={'status':'failed','errors':[{'type':type(exc).__name__,'message':str(exc)}],'GT_opened':False,'scoring_performed':False,'timing_claims':False,'finished_unix':time.time()}
 result['started_unix']=begin
 with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if result['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
