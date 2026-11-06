"""Independent CPU-only random preflight source and complete manifest review."""
import argparse,collections,hashlib,json,os,struct
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e6/random_smoke';I=B/'e1_independent_review';OUT=I/'random_smoke_preflight_review_002.json';MON=I/'monitor_random_smoke_preflight_002'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(folder):
 s=read(folder/'started.json');t=read(folder/'terminal.json')
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid'):raise ValueError('supervisor incomplete: '+str(folder))
 return s
def run():
 plan=D/'runtime_plan003.json';p=read(plan)
 if p.get('stage')!='preflight_only' or p.get('GPU_authorized') is not False or p.get('GT_opened') is not False or p.get('predictions_opened') is not True or p.get('prediction_scope')!='previously accepted DEV mixed prediction files opened only for source SHA; not parsed':raise ValueError('scope/disclosure')
 for src,h in p['source_sha256'].items():
  if sha(src)!=h:raise ValueError('source changed '+src)
 if sha(D/'selection_manifest001.jsonl')!=p['manifest_sha256'] or sha(D/'plan001.json')!=p['random_plan_sha256']:raise ValueError('manifest binding')
 term(D/'monitor_runtime_freeze003');term(D/'monitor_preflight_cpu003');term(D/'monitor_validate001');term(D/'monitor_freeze001');testlog=(D/'monitor_preflight_cpu003/stderr.log').read_text()
 if 'Ran 5 tests' not in testlog or 'OK' not in testlog:raise ValueError('CPU tests incomplete')
 old=read(D/'manifest_validation001.json')
 if old.get('status')!='passed' or old.get('questions')!=384 or old.get('selections')!=1152 or old.get('reviewer_sha256')!=sha(D/'validate_manifest.py') or old.get('selection_manifest_sha256')!=sha(D/'selection_manifest001.jsonl'):raise ValueError('manifest prior audit')
 inputs=[json.loads(x) for x in (B/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines() if x.strip()];inputs={(x['task'],str(x['qid']),x['group_id']):x for x in inputs};seeds=(20261104,20261105,20261106);seen=set();counts=collections.Counter()
 if len(inputs)!=384:raise ValueError('input count')
 for line in (D/'selection_manifest001.jsonl').read_text().splitlines():
  r=json.loads(line);k=(r['task'],str(r['qid']),r['group_id']);seed=r['seed'];full=(*k,seed)
  if full in seen or k not in inputs or seed not in seeds or r['algorithm']!='sha256-rejection-fisher-yates-v1' or any(r[x]!=inputs[k][x] for x in ('image_sha256','message_sha256')):raise ValueError('input identity')
  seedbytes=hashlib.sha256(('eadp-e6-random-v1|'+str(seed)+'|'+r['image_sha256']).encode()).digest()[:8];counter=0;queue=[];arr=list(range(1024))
  for end in range(1023,0,-1):
   n=end+1;limit=(1<<64)-((1<<64)%n)
   while True:
    if not queue:queue=list(struct.unpack('>QQQQ',hashlib.sha256(seedbytes+struct.pack('>Q',counter)).digest()));counter+=1
    n64=queue.pop(0)
    if n64<limit:break
   pos=n64%n;arr[end],arr[pos]=arr[pos],arr[end]
  if int.from_bytes(seedbytes,'big')!=r['derived_seed64'] or sorted(arr[:256])!=r['spatial_ids']:raise ValueError('random selection mismatch')
  seen.add(full);counts[r['task']]+=1
 if len(seen)!=1152 or counts!={'DocVQA_VAL':384,'TextVQA_VAL':384,'AI2D_TEST':384}:raise ValueError('missing selection')
 return {'status':'passed','independent_review':True,'preflight_only':True,'runtime_plan_sha256':sha(plan),'source_sha256':p['source_sha256'],'reviewer_sha256':sha(__file__),'review_started_sha256':sha(MON/'started.json'),'GT_opened':False,'GPU_used':False,'predictions_opened':True,'prediction_scope':'previously accepted DEV mixed prediction files opened only for source SHA; not parsed','questions':384,'selections':1152,'errors':[],'limitations':['Only author module imports, paths and four weight payloads will be preflighted; no model/GPU identity or performance follows.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=OUT:raise ValueError('unexpected output')
 result=run()
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
