"""Independently reconstruct every fixed random image/seed selection; CPU-only."""
import argparse,collections,hashlib,json,os,struct
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def reproduce(image,seed):
 seed_bytes=hashlib.sha256(('eadp-e6-random-v1|'+str(seed)+'|'+image).encode()).digest()[:8];seed64=int.from_bytes(seed_bytes,'big');counter=0;values=[];a=list(range(1024))
 for end in range(1023,0,-1):
  bound=end+1;ceiling=2**64-(2**64%bound)
  while True:
   if not values:
    block=hashlib.sha256(seed_bytes+struct.pack('>Q',counter)).digest();counter+=1;values=list(struct.unpack('>QQQQ',block))
   value=values.pop(0)
   if value<ceiling:break
  j=value%bound;a[end],a[j]=a[j],a[end]
 return seed64,sorted(a[:256])
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();plan=json.loads((D/'plan001.json').read_text());mp=D/'selection_manifest001.jsonl'
if plan['status']!='planning_only' or plan['GPU_authorized'] is not False or plan['GT_opened'] is not False or plan['predictions_opened'] is not False or plan['seeds']!=[20261104,20261105,20261106] or sha(mp)!=plan['selection_manifest_sha256']:raise ValueError('plan mismatch')
for p,h in plan['source_sha256'].items():
 if sha(p)!=h:raise ValueError('source changed '+p)
rows=[json.loads(s) for s in (B/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];inputs={(r['task'],str(r['qid']),r['group_id']):r for r in rows};count=collections.Counter();seen=set()
for line in mp.open():
 r=json.loads(line);key=(r['task'],str(r['qid']),r['group_id']);full=(*key,r['seed']);src=inputs[key]
 if full in seen or r['seed'] not in plan['seeds'] or r['image_sha256']!=src['image_sha256'] or r['message_sha256']!=src['message_sha256'] or r['algorithm']!='sha256-rejection-fisher-yates-v1':raise ValueError('identity mismatch')
 derived,ids=reproduce(r['image_sha256'],r['seed'])
 if r['derived_seed64']!=derived or r['spatial_ids']!=ids or len(ids)!=256 or len(set(ids))!=256:raise ValueError('selection differs')
 seen.add(full);count[r['task']]+=1
if len(seen)!=1152 or len(inputs)!=384 or count!={'DocVQA_VAL':384,'TextVQA_VAL':384,'AI2D_TEST':384}:raise ValueError('denominator mismatch')
out={'status':'passed','scope':'CPU selection/input audit only; no real model or GPU identity','questions':384,'selections':1152,'seeds':plan['seeds'],'plan_sha256':sha(D/'plan001.json'),'selection_manifest_sha256':sha(mp),'reviewer_sha256':sha(__file__),'GT_opened':False,'predictions_opened':False,'GPU_used':False,'errors':[]}
with args.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
