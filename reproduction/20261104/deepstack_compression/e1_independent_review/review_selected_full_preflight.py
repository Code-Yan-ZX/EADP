"""Independent Div/CD full-panel source review; only CPU preflight permitted."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'e6/selected_full';sys.path.insert(0,str(HERE))
import cpu_validation,gate,freeze_plan
def sha(p):return freeze_plan.sha(p)
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==HERE/'independent_preflight_review.json'
 cpu=HERE/'cpu_check_005/verification.json';start=HERE/'monitor_cpu005/started.json';end=start.with_name('terminal.json');gate.supervised(start,end,HERE/'cpu_validation.py',cpu.parent)
 c=read(cpu);src={str(x):sha(x) for x in cpu_validation.sources()};assert c['status']=='passed' and c['source_sha256_before']==c['source_sha256_after']==src and len(src)==126 and all(c[k] is False for k in ('GT_opened','predictions_opened','real_model_loaded','gpu_used'))
 tests=c['tests'];assert len(tests)==6 and len({x['command'][1] for x in tests})==6
 for t in tests:
  name=Path(t['command'][1]).stem;assert t['returncode']==0 and t['timeout'] is False
  for stream in ('stdout','stderr'):assert t[stream+'_sha256']==sha(cpu.parent/(name+'.'+stream))
  log=(cpu.parent/(name+'.stderr')).read_text();assert 'Ran ' in log and log.rstrip().endswith('OK')
 p=read(HERE/'plan.json');assert p==freeze_plan.build() and p['questions']==384 and p['generation_count']==1536 and p['methods']==['div','cd'] and p['gpu_authorized'] is False and p['gt_opened'] is False and p['scoring'] is False
 assert p['inner_timeout_seconds']==10800 and p['outer_timeout_seconds']==10980 and p['snapshot_seconds']==3600 and p['progress_stale_seconds']==1800
 upstream=gate.upstream(host=True);assert upstream['plan_sha256']==sha(HERE/'plan.json')
 r={'status':'passed','independent_review':True,'preflight_only':True,'GT_opened':False,'gpu_used':False,'reviewer_sha256':sha(__file__),
    'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(start),'cpu_terminal_sha256':sha(end),'source_sha256':src,'upstream':upstream,
    'limitations':['Only actual no-GPU worker preflight is permitted by this proof.','Full panel stores summaries, not high-dimensional arrays; full tensor reconstruction is limited to the separate completed smoke.']}
 with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','sources':len(src),'scope':'CPU container preflight only'}))
if __name__=='__main__':main()
