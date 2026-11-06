"""Exploratory group-level summary of fully accepted 32-image pilot expansion."""
import argparse,json,math,os,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE/'e4_c_latency'))
from identity_gate import sha,supervised
from latency_admission import terminal

COMPONENTS=('selector','ds_merger_three_streams','ds_injection')
def summarize(events,identities):
 if len(identities)!=32:raise ValueError('32 images required')
 buckets={(i,c):[] for i in range(32) for c in COMPONENTS}
 for e in events:
  if e.get('status')!='measured':continue
  i=e.get('image_index');j=e.get('job',{});c=j.get('component');v=e.get('gpu_event_ms')
  if (i,c) not in buckets or type(v) not in (int,float) or not math.isfinite(v) or v<0 or (j.get('task'),j.get('group_id'))!=(identities[i]['task'],identities[i]['group_id']):raise ValueError('measurement identity/value mismatch')
  buckets[(i,c)].append(v)
 if any(len(v)!=20 for v in buckets.values()):raise ValueError('missing or duplicate measurement')
 per_image=[{'task':r['task'],'qid':str(r['qid']),'group_id':r['group_id'],'median_gpu_ms':{c:statistics.median(buckets[(i,c)]) for c in COMPONENTS}} for i,r in enumerate(identities)]
 def group(rows):return {'n_images':len(rows),'components':{c:{'median_of_image_medians_ms':statistics.median(r['median_gpu_ms'][c] for r in rows),'min_image_median_ms':min(r['median_gpu_ms'][c] for r in rows),'max_image_median_ms':max(r['median_gpu_ms'][c] for r in rows)} for c in COMPONENTS}}
 tasks=sorted(set(r['task'] for r in per_image));return {'overall':group(per_image),'by_task':{t:group([r for r in per_image if r['task']==t]) for t in tasks},'per_image':per_image}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();full=BASE/'e4_c_microbench_full';rd=BASE/'e1_independent_review';f=full/'full_001_verification.json';fs=full/'monitor_verify_full001/started.json';ft=full/'monitor_verify_full001/terminal.json';r=rd/'component_full001_review.json';rs=rd/'monitor_component_full001/started.json';rt=rd/'monitor_component_full001/terminal.json';runs=full/'monitor_full001/started.json';runt=full/'monitor_full001/terminal.json';reviewer=rd/'review_component_full001.py'
 supervised(fs,ft,full/'full_verify.py',f);supervised(rs,rt,reviewer,r);terminal(runs,runt);formal=json.loads(f.read_text());review=json.loads(r.read_text())
 if formal.get('status')!='passed' or review.get('status')!='passed' or review.get('independent_review') is not True or any(formal.get(k)!=v or review.get(k)!=v for k,v in [('images',32),('jobs',1920),('warmups',288),('diagnostic_generations',32),('GT_opened',False),('scoring_performed',False)]) or review.get('reviewer_sha256')!=sha(reviewer) or review.get('array_sha256')!=formal.get('array_sha256') or any(review.get(k+'_sha256')!=sha(v) for k,v in [('formal',f),('formal_started',fs),('formal_terminal',ft),('run_started',runs),('run_terminal',runt)]):raise ValueError('full independent gate failed')
 j=full/'full_001/journal.jsonl'
 if formal.get('files_sha256',{}).get(str(j.resolve()))!=sha(j):raise ValueError('journal changed')
 schedule=BASE/'e4_c_microbench/schedule001.json'
 if formal.get('source_sha256',{}).get(str(schedule.resolve()))!=sha(schedule):raise ValueError('frozen schedule changed')
 identities=json.loads(schedule.read_text())['input_identities'];events=[json.loads(x) for x in j.read_text().splitlines() if x.strip()];result=summarize(events,identities)
 result.update(status='passed',formal_sha256=sha(f),independent_review_sha256=sha(r),journal_sha256=sha(j),schedule_sha256=sha(schedule),summary_source_sha256=sha(__file__),scope='post-pilot exploratory 32-image component timing; no GT or scoring', limitations=['median of 20 repeats per image then median across images; image is sampling unit','not end-to-end TTFT or an equal-compute comparison','ds_injection excludes decoder DS addition','one light Python file-stat window [13974385382685808,13974385382692250] monotonic ns; process startup/other light overhead unknown and journal lacks timestamp boundaries; all observations retained','individual measurement outputs verified online but not archived'])
 with a.out.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
