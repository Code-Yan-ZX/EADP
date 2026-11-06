"""Fail closed before any GPU query until a separate reviewer signs the preflight."""
import json
from pathlib import Path
from worker import sha,check_plan
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e1_independent_review'
def read(p):return json.loads(Path(p).read_text())
def terminal(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid'] or [(cwd/x).resolve() for x in c if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor mismatch')
 if out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('output mismatch')
 return s
def require_final():
 check_plan();test=D/'monitor_cpu_test001';terminal(test,D/'test_worker.py');proof=D/'independent_prelaunch_review.json';reviewer=R/'review_pixel_wavefront_t020_gpu32_prelaunch.py';m=R/'monitor_pixel_wavefront_t020_gpu32_prelaunch';terminal(m,reviewer,proof);r=read(proof);pre=D/'preflight001.json';term=D/'monitor_preflight001';terminal(term,D/'launch.py');ex=D/'run_preflight001/exit.json';x=read(ex);f=read(pre)
 fields={'status':'passed','independent_review':True,'plan_sha256':sha(D/'plan001.json'),'reviewer_sha256':sha(reviewer),'preflight_sha256':sha(pre),'preflight_started_sha256':sha(term/'started.json'),'preflight_terminal_sha256':sha(term/'terminal.json'),'preflight_exit_sha256':sha(ex),'cpu_started_sha256':sha(test/'started.json'),'cpu_terminal_sha256':sha(test/'terminal.json'),'cpu_stderr_sha256':sha(test/'stderr.log'),'GT_opened':False,'errors':[]}
 if any(r.get(k)!=v for k,v in fields.items()) or f.get('status')!='passed' or f.get('cuda_initialized') is not False or f.get('gpu_used') is not False or x.get('returncode')!=0 or x.get('container_exit_code')!=0 or x.get('container_stopped_verified') is not True or x.get('last_inspect',{}).get('host_devices')!=[]:raise ValueError('prelaunch review mismatch')
 return r
