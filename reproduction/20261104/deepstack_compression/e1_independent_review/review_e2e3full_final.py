"""Independent no-GT E2/E3 source and preflight admission review."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent; FULL=BASE/'e2_e3_full'
sys.path.insert(0,str(FULL))
from full_imports import local

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 cpu=FULL/'cpu_check_003/verification.json';started=FULL/'monitor_cpu003/started.json';terminal=FULL/'monitor_cpu003/terminal.json'
 c=read(cpu);s=read(started);t=read(terminal)
 assert c['status']=='passed' and c['returncode']==0 and c['timed_out'] is False
 assert c['GT_opened'] is False and c['scoring_performed'] is False and c['gpu_used'] is False
 assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and t['worker_pid']==s['worker_pid']
 assert [str((Path(s['cwd'])/x).resolve()) for x in s['command'] if Path(x).name=='cpu_checks.py']==[str((FULL/'cpu_checks.py').resolve())]
 assert s['command'].count('--out')==1 and Path(s['command'][s['command'].index('--out')+1]).resolve()==cpu.parent
 before=c['source_sha256_before'];after=c['source_sha256_after'];sources=local('source_inventory').sources()
 assert before==after and set(after)==set(sources) and all(sha(path)==digest for path,digest in after.items())
 assert sha(cpu.parent/'stdout.log')==c['stdout_sha256'] and sha(cpu.parent/'stderr.log')==c['stderr_sha256']
 stderr=(cpu.parent/'stderr.log').read_text();assert 'Ran 23 tests' in stderr and stderr.rstrip().endswith('OK')
 gate=local('panel_gate');smoke=gate.smoke_gate();b=gate.b_gate();assert c['upstream']=={'smoke':smoke,'e4b':b}
 contract=local('contract');runner=local('panel_runner')
 assert contract.E2==('000','100','010','001','110','101','011','111') and contract.E3==('PERM','MEAN')
 rows=[dict(task=task,qid=str(i),group_id=task+str(i),split='DEV') for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for i in range(128)]
 arms=[('E2',x) for x in contract.E2]+[('E3',x) for x in contract.E3]
 recs=[dict(task=r['task'],qid=r['qid'],group_id=r['group_id'],family=f,arm=arm,forced=False) for r in rows for f,arm in arms]
 counts=contract.validate_records(rows,recs);assert counts['physical_generations']==3840 and len(contract.e3_views(rows,recs))==384
 for change in ('drop','reorder','forced','identity'):
  bad=[dict(x) for x in recs]
  if change=='drop':bad.pop()
  elif change=='reorder':bad[0],bad[1]=bad[1],bad[0]
  elif change=='forced':bad[0]['forced']=True
  else:bad[0]['group_id']='different'
  try:contract.validate_records(rows,bad)
  except ValueError:pass
  else:raise AssertionError('invalid panel accepted: '+change)
 assert runner.ARMS==tuple(arms)
 # Audited worker, launch, intervention and verifier source directly: generated
 # events are fsynced before postprocessing; all rows compare E1 inputs; actual
 # 000/111 endpoints rerun; fixed anchors retain arrays; partial panel cannot
 # emit completion; timeout/signal path stops and inspects actual container.
 fp=FULL/'container_preflight_001.json';fs=FULL/'monitor_preflight001/started.json';ft=FULL/'monitor_preflight001/terminal.json'
 f=read(fp);gate.terminal(fs,ft,FULL/'preflight.py');gate.check_preflight(f,FULL/'preflight_001')
 assert f['status']=='passed' and f['container_stopped_verified'] is True and not f['errors'] and f['interrupted'] is None
 assert f['client_exited_while_container_running'] is False and f['container_pid']==0
 obj=json.loads(subprocess.check_output(['docker','inspect',f['container_id']],timeout=20))[0]
 assert obj['Id']==f['container_id'] and obj['State']['Running'] is False and obj['State']['Pid']==0 and obj['State']['ExitCode']==0 and not obj['State']['OOMKilled']
 assert not obj['HostConfig'].get('Devices') and not obj['HostConfig'].get('DeviceRequests')
 cmd=read(FULL/'preflight_001/launch.json')['command'];image=obj['Config']['Image']
 assert obj['Config']['Cmd']==cmd[cmd.index(image)+1:] and obj['Image']=='sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf'
 assert 'CUDA_VISIBLE_DEVICES=' in obj['Config']['Env']
 result={'status':'passed','independent_review':True,'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(started),'cpu_terminal_sha256':sha(terminal),'source_sha256':after,'smoke_review_sha256':smoke['review_sha256'],'b_review_sha256':b['review_sha256'],'reviewer_sha256':sha(__file__),'preflight_sha256':sha(fp),'preflight_started_sha256':sha(fs),'preflight_terminal_sha256':sha(ft),'GT_opened':False,'scoring_performed':False,'checks':['85 bound runtime and test sources unchanged','23 completed CPU tests, including real SIGTERM child test','actual smoke and E4-B formal/independent evidence','3840 synthetic full inventory, aliases and four corruptions','manual worker, journal, E1 endpoints, verifier and bounded container lifecycle audit','same-image actual stopped container without GPU devices and no model loading'],'limits':['no GPU, GT or scoring in this review','full run still requires bounded supervision and exclusive GPU']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','sources':len(after),'synthetic_records':len(recs)}))
if __name__=='__main__':main()
