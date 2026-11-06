"""Create the future reviewed resource/source plan without admitting a GPU."""
import hashlib,json,os,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;R=B/'e4_c_multibudget_worker192_v5'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def main():
 out=R/'gpu_plan001.json'
 if out.exists():raise ValueError('GPU resource plan already frozen')
 cpu=json.loads((R/'cpu_plan001.json').read_text());sources=dict(cpu['source_sha256'])
 for p in (R/'cpu_plan001.json',R/'cpu_proof001.json',R/'monitor_cpu001/started.json',R/'monitor_cpu001/terminal.json',D/'freeze_gpu_plan.py'):
  sources[str(p.resolve())]=sha(p)
 if any(sha(p)!=digest for p,digest in sources.items()):raise ValueError('source changed during GPU-plan freeze')
 p={'stage':'192-image four-budget timing','jobs':7920,'warmups':240,'measured':7680,'GT_opened':False,'target_predictions_opened':False,'predictions_opened':True,'prediction_scope':'previously accepted DEV prediction files opened as source bytes for hashes only','GPU_authorized_by_plan':False,'source_sha256':sources,'proposal_sha256':sha(B/'e4_c_multibudget_expansion192/proposal002.json'),'budget_source_sha256':cpu['budget_source_sha256'],'inner_timeout_seconds':28680,'outer_timeout_seconds':28800,'stale_seconds':1800,'acceptance_reserve_seconds':3600,'minimum_free_bytes':cpu['minimum_free_bytes'],'created_unix':time.time(),'scope':'resource/source proposal only; complete accepted 888 references, no-GPU container, analysis, environment, verification and GPU independent reviews are mandatory'}
 with out.open('x') as f:json.dump(p,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('resource proposal SHA256:',sha(out))
if __name__=='__main__':main()
