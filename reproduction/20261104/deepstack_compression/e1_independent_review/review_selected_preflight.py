"""Independent CPU/source audit; authorizes only a no-GPU container preflight."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'e6/selected_smoke'
sys.path.insert(0,str(HERE))
import cpu_validation,freeze_plan,gate,container_mount
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 assert a.out.resolve()==HERE/'independent_preflight_review.json'
 cpu=HERE/'cpu_check_006/verification.json';start=HERE/'monitor_cpu006/started.json';end=HERE/'monitor_cpu006/terminal.json'
 gate.supervised(start,end,HERE/'cpu_validation.py',cpu.parent)
 c=read(cpu);src={str(p):sha(p) for p in cpu_validation.sources()}
 assert c['status']=='passed' and c['source_sha256_before']==c['source_sha256_after']==src
 assert c['GT_opened'] is False and c['real_model_loaded'] is False and c['generation_performed'] is False
 assert len(c['tests'])==2
 for i,(t,count) in enumerate(zip(c['tests'],(30,4))):
  assert t['returncode']==0 and t['timeout'] is False
  for stream in ('stdout','stderr'):assert t[stream+'_sha256']==sha(cpu.parent/f'{i}.{stream}')
  log=(cpu.parent/f'{i}.stderr').read_text();assert f'Ran {count} tests' in log and log.rstrip().endswith('OK')
 plan=read(HERE/'plan.json');assert plan==freeze_plan.plan();gate.budget(True)
 assert plan['methods']==['div','cd'] and plan['natural_generations']==12 and plan['forced_generations']==4 and plan['fixed_decode_steps']==32 and plan['gt_read'] is False
 old=read(ROOT/'m3_gpu_smoke/preflight_001/launch.json')['command'];worker=str(ROOT/'m3_gpu_smoke/worker.py');assert old.count(worker)==1
 cmd=container_mount.add(old[:old.index(worker)]+[str(HERE/'worker.py'),'--execute','--preflight-only','--out',str(HERE/'preflight_REVIEW')])
 assert cmd.count('--mount')>0 and cmd.count(container_mount.MOUNT)==1 and 'CUDA_VISIBLE_DEVICES=' in cmd and not any('/dev/nvidia' in str(x) for x in cmd)
 result={'status':'passed','independent_review':True,'preflight_only':True,'GT_opened':False,'gpu_used':False,
         'reviewer_sha256':sha(__file__),'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(start),'cpu_terminal_sha256':sha(end),'source_sha256':src,
         'limitations':['Only synthetic CPU tests have run; actual worker preflight must establish container mounts and source hashes.',
                        'No model, GPU, natural outputs, forced decode, or scoring are authorized by this proof.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','sources':len(src),'scope':'no-GPU preflight only'}))
if __name__=='__main__':main()
