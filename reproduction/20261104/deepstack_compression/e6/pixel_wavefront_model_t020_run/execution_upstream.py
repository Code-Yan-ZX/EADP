"""Bind the separately reviewed no-GPU execution import preflight only."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];E=B/'e6/pixel_wavefront_model_t020_execution';I=B/'e1_independent_review'
R=I/'pixel_wavefront_t020_execution_preflight_result_review.json';C=I/'review_pixel_wavefront_t020_execution_preflight_result.py';S=I/'monitor_pixel_wavefront_t020_execution_preflight_result/started.json';T=I/'monitor_pixel_wavefront_t020_execution_preflight_result/terminal.json'
EXPECTED={R:'7f8a9c6308a940fa0b9d0fbdc199dfe7fb087f87c2c5420a19ac50371c8e41f2',C:'c828de5b96034016a75544971e39f4e953ed075c45758b81b658c7333651a7bf',S:'c42689f25ade6ce8abf6bf84cde9fa5a3b4a19ff7e1b74ec7efa575ae5666f80',T:'ac8dc74d75d9b7ad871743a6a8795730c9e81f7c75705e510b05017dae6a59d1'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify():
 for p,v in EXPECTED.items():
  if sha(p)!=v:raise ValueError('execution preflight reviewer evidence changed')
 p=read(R);s=read(S);t=read(T);cmd=s['command'];cwd=Path(s['cwd'])
 if (p.get('status')!='passed' or p.get('independent_review') is not True or p.get('scope')!='execution import no-GPU preflight only' or p.get('gpu_authorized') is not False or p.get('GT_opened') is not False or p.get('existing_DEV_predictions_opened') is not True or p.get('errors')!=[] or p.get('reviewer_sha256')!=sha(C) or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid') or (cwd/cmd[1]).resolve()!=C or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=R):raise ValueError('execution preflight review scope or supervision')
 fields={'plan_sha256':E/'preflight_plan002.json','worker_preflight_sha256':E/'run_preflight001/worker_preflight.json','launch_sha256':E/'run_preflight001/launch.json','exit_sha256':E/'run_preflight001/exit.json','started_sha256':E/'monitor_preflight001/started.json','terminal_sha256':E/'monitor_preflight001/terminal.json'}
 for k,path in fields.items():
  if p.get(k)!=sha(path):raise ValueError('execution preflight file changed '+k)
 if len(p['source_sha256'])!=276:raise ValueError('execution preflight source count')
 for path,value in p['source_sha256'].items():
  if sha(path)!=value:raise ValueError('execution preflight source changed '+path)
 return {'status':'passed','review_sha256':sha(R),'reviewer_authored_wavefront_candidate':True,'scope':'no-GPU preflight only'}
