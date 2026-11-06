"""Three-anchor, three-repeat descriptive timing; no population inference."""
import json,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];RUN=B/'e6/pixel_wavefront_t020_timing_final_v4';A=B/'e6/pixel_wavefront_t020_timing_accept_final'
def analyze(jobs,rows,annotations):
 if len(jobs)!=24 or len(rows)!=24 or len(annotations)!=24 or [j['order'] for j in jobs]!=list(range(24)) or [r['order'] for r in rows]!=list(range(24)) or [a['order'] for a in annotations]!=list(range(24)):raise ValueError('full timing inventory')
 cells={};out=[]
 for j,r,a in zip(jobs,rows,annotations):
  if j['phase']!=r['phase'] or type(r.get('ttft_ns')) is not int or r['ttft_ns']<=0:raise ValueError('invalid timing record')
  row={'job':j,'ttft_ns':r['ttft_ns'],'background':a['background']};out.append(row)
  if j['phase']=='measurement':
   key=(j['task'],str(j['qid']),j['arm']);rep=j['repeat']
   if type(rep) is not int or rep not in (0,1,2) or rep in cells.setdefault(key,{}):raise ValueError('repeat')
   cells[key][rep]=r['ttft_ns']
 if len(cells)!=6 or any(set(v)!={0,1,2} for v in cells.values()):raise ValueError('incomplete cells')
 anchors=sorted({(k[0],k[1]) for k in cells});reports=[]
 if len(anchors)!=3:raise ValueError('three anchors')
 for task,qid in anchors:
  a=cells[(task,qid,'author')];w=cells[(task,qid,'wavefront')]
  reports.append({'task':task,'qid':qid,'author_ns':[a[i] for i in range(3)],'wavefront_ns':[w[i] for i in range(3)],'author_median_ns':statistics.median(a.values()),'wavefront_median_ns':statistics.median(w.values()),'wavefront_over_author_paired_ratios':[w[i]/a[i] for i in range(3)]})
 return {'status':'passed','all_jobs':out,'warmups':6,'measured':18,'per_anchor':reports,'scope':'post-hoc three anchors; three isolated processes per arm','limitations':['repeat labels are preassigned; the two arm processes run at different times','no cross-task population or inferential confidence interval','separate-process warmup does not establish per-process steady state','selector observation overhead included for both arms','unobserved host activity remains unknown','not equal-compute comparison to t0 or EADP']}
def main(approved):
 plan=json.loads((RUN/'gpu_plan001.json').read_text());formal=json.loads((A/'verification001.json').read_text());result=analyze(plan['jobs'],formal['rows'],formal['background_annotations']);result.update(approval=approved,GT_opened=False,scoring_performed=False)
 out=D/'analysis001.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();__import__('os').fsync(f.fileno())
if __name__=='__main__':raise SystemExit('analysis requires reviewed entry and result proofs')
