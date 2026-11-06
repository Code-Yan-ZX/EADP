"""Extract predeclared unlabeled features. Never opens E4 scores or GT."""
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np
from PIL import Image, __version__ as pillow_version

B=Path(__file__).resolve().parent.parent;D=B/'e4_adaptive_features'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def features(gray):
 arr=np.asarray(gray,dtype=np.uint8)
 if arr.shape!=(256,256):raise ValueError('unexpected grayscale shape')
 bins=np.bincount((arr.reshape(-1)//16).astype(np.int64),minlength=16)
 prob=bins[bins>0].astype(np.float64)/arr.size
 entropy=float(-(prob*np.log2(prob)).sum())
 val=arr.astype(np.int16);gradient=float((np.abs(np.diff(val,axis=1)).mean()+np.abs(np.diff(val,axis=0)).mean())/255.)
 if not math.isfinite(entropy) or not math.isfinite(gradient):raise ValueError('invalid feature')
 return entropy,gradient
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 plan=json.loads((D/'plan001.json').read_text());src=B/plan['inputs']
 if sha(src)!=plan['inputs_sha256']:raise ValueError('input source changed')
 rows=[json.loads(x) for x in src.read_text().splitlines()]
 if len(rows)!=384 or len({r['group_id'] for r in rows})!=384:raise ValueError('input inventory')
 for task in plan['tasks']:
  if sum(r['task']==task for r in rows)!=128:raise ValueError('task inventory')
 a.out.mkdir(exist_ok=False);results=[]
 for n,r in enumerate(rows):
  path=Path(r['image_path'])
  if sha(path)!=r['image_sha256']:raise ValueError('image identity mismatch')
  with Image.open(path) as image:
   gray=image.convert('L').resize((256,256),Image.Resampling.BILINEAR)
   entropy,gradient=features(gray)
  text=''.join(x['value'] for x in r['message'] if x.get('type')=='text')
  results.append({'task':r['task'],'qid':str(r['qid']),'group_id':r['group_id'],'image_sha256':r['image_sha256'],'gray_entropy16':entropy,'gray_gradient':gradient,'user_text_chars':len(text)})
  with (a.out/'progress.jsonl').open('a') as f:f.write(json.dumps({'completed':n+1,'task':r['task'],'qid':str(r['qid'])})+'\n');f.flush();os.fsync(f.fileno())
 out=a.out/'features.jsonl'
 with out.open('x') as f:
  for r in results:f.write(json.dumps(r,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 done={'status':'completed','n':len(results),'plan_sha256':sha(D/'plan001.json'),'input_sha256':sha(src),'feature_sha256':sha(out),'extractor_sha256':sha(__file__),'pillow_version':pillow_version,'numpy_version':np.__version__,'GT_opened':False,'score_or_prediction_opened':False}
 with (a.out/'extraction.json').open('x') as f:json.dump(done,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('completed')
if __name__=='__main__':main()
