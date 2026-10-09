"""Paired image-cluster bootstrap; fixed seed, official per-question values."""
from collections import defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
def compare(reports):
 by={r['score']['method']:r for r in reports}
 if set(by)!={'EADP','AnchorZip'}:raise ValueError('One complete EADP and one complete AnchorZip required')
 e,a=by['EADP'],by['AnchorZip'];er=e['score']['per_question'];ar=a['score']['per_question'];task=e['score']['task'];field='correct'if task=='gqa'else'acc'
 if e['score']['n']!=a['score']['n']or len(er)!=len(ar)or [str(r['question_id'])for r in er]!=[str(r['question_id'])for r in ar]:raise ValueError('Full ordered paired IDs required')
 ident=json.loads(Path(e['protocol']['identity_manifest']).read_text())['rows'];images={str(q['question_id']):q['image']for q in ident};clusters=defaultdict(list)
 differences=[]
 for x,y in zip(er,ar):
  delta=float(y[field])-float(x[field]);differences.append(delta);clusters[images[str(x['question_id'])]].append(delta)
 sums=np.array([sum(v)for v in clusters.values()],dtype=np.float64);counts=np.array([len(v)for v in clusters.values()]);rng=np.random.default_rng(20261009);boot=[]
 for _ in range(2000):
  ix=rng.integers(0,len(sums),len(sums));boot.append(100*sums[ix].sum()/counts[ix].sum())
 return dict(task=task,n=len(er),image_clusters=len(clusters),eadp=e['score']['value'],anchorzip=a['score']['value'],delta_pp=a['score']['value']-e['score']['value'],question_higher=sum(x>0 for x in differences),question_lower=sum(x<0 for x in differences),question_equal=sum(x==0 for x in differences),bootstrap=dict(unit='image cluster',repeats=2000,seed=20261009,ci95_pp=np.quantile(boot,[.025,.975]).tolist()),sources={m:dict(prediction_sha256=r['score']['prediction_sha256'],control_protocol_sha256=r['score']['control_protocol_sha256'])for m,r in by.items()},interpretation='Same protocol local method difference only; reported paper configuration identity remains unresolved.')
