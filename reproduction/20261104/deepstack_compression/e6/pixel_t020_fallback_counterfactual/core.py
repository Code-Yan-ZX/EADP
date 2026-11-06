"""Pure synthetic-compatible paired counterfactual; never reads files."""
import math
import numpy as np
THRESHOLDS=(128,256,512)
def calculate(rows):
 if len(rows)!=128 or len({r['group_id'] for r in rows})!=128:raise ValueError('task denominator or group duplication')
 rows=sorted(rows,key=lambda r:r['group_id'])
 for r in rows:
  if any(type(r[k]) is not int or not 1<=r[k]<=1024 for k in ('k020','k0')):raise ValueError('invalid selected K')
  if any(not math.isfinite(r[k]) or not 0<=r[k]<=1 for k in ('s020','s0','e4')):raise ValueError('invalid score')
 rng=np.random.default_rng(20261104);ix=rng.integers(0,128,(10000,128));answer={}
 for threshold in THRESHOLDS:
  fallback=np.array([r['k020']<=threshold for r in rows]);x=np.array([r['s0'] if f else r['s020'] for r,f in zip(rows,fallback)],dtype=float);k=np.array([r['k0'] if f else r['k020'] for r,f in zip(rows,fallback)]);ref0=np.array([r['s0'] for r in rows]);refe=np.array([r['e4'] for r in rows]);comp={}
  for label,y in (('versus_t0',ref0),('versus_eadp_k256',refe)):
   d=x-y;ci=np.quantile(100*d[ix].mean(axis=1),[.025,.975]);comp[label]={'difference_pp':float(100*d.mean()),'image_group_percentile_95_ci_pp':[float(v) for v in ci],'improved':int((d>1e-12).sum()),'declined':int((d< -1e-12).sum()),'unchanged':int((abs(d)<=1e-12).sum())}
  answer[str(threshold)]={'n_questions':128,'n_image_groups':128,'fallback_count':int(fallback.sum()),'fallback_percent':float(100*fallback.mean()),'mean_selected_k':float(k.mean()),'mean_k020':float(np.mean([r['k020'] for r in rows])),'mean_k0':float(np.mean([r['k0'] for r in rows])),'mean_score_pct':float(100*x.mean()),'contrasts':comp}
 return answer
