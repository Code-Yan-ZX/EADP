"""Independent table arithmetic against the nine accepted pilot records."""
import argparse,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixel_natural_timing_report_v4';P=B/'e6/pixel_natural_timing_pilot_v4';I=B/'e1_independent_review';read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');c=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==I/'pixel_natural_timing_v4_report_review.json'
term(I/'monitor_pixel_natural_timing_v4_result',I/'review_pixel_natural_timing_v4_result.py');f=read(I/'pixel_natural_timing_v4_result_review.json');assert f['status']=='passed' and f['independent_review'] is True and f['measured']==9
term(D/'monitor_report001',D/'report.py');r=read(D/'report001.json');md=(D/'RESULT.md').read_text();assert r['status']=='completed' and r['measured']==9 and r['all_samples_retained'] is True and r['independent_review_sha256']==sha(I/'pixel_natural_timing_v4_result_review.json') and r['report_sha256']==sha(D/'RESULT.md')
by={(x['task'],x['qid'],x['arm']):x for x in f['measurement_recomputation']};plan=read(P/'plan001.json');labels={'eadp_k256':'EADP K256','pixel_t0':'Pixel τ=0','pixel_t045':'Pixel τ=0.45'}
for job in plan['jobs']:
 key=(job['task'],job['qid'],job['arm']);x=by[key];base=by[(job['task'],job['qid'],'eadp_k256')];seq=[json.loads(z) for z in (P/'run_pilot001'/('job_%03d'%job['order'])/'observations/journal.jsonl').read_text().splitlines()];setup=seq[1]['setup_wall_ns']/1e9
 row=f"| {job['task']} / {job['qid']} | {labels[job['arm']]} | {x['K']} | {x['ttft_ns']/1e6:.2f} | {x['ttft_ns']/base['ttft_ns']:.3f}× | {setup:.2f} | {x['allocator_max_allocated_bytes']/1024**3:.2f} | {x['allocator_max_reserved_bytes']/1024**3:.2f} |"
 assert row in md,row
assert '各测次与查询直接重叠未知' in md and '每个方法、每图一次测量' in md and '正式验收器在GPU运行结束后' in md and '不是 NVML' in md
out={'status':'passed','independent_review':True,'rows_verified':9,'GT_opened':False,'scoring_performed':False,'report_sha256':sha(D/'RESULT.md'),'report_json_sha256':sha(D/'report001.json'),'result_review_sha256':sha(I/'pixel_natural_timing_v4_result_review.json'),'reviewer_sha256':sha(__file__),'limitations':['same three-anchor observational pilot; one measurement per cell','health-monitor overlap remains unknown; external host activity effect not determined'],'errors':[]};a.out.write_text(json.dumps(out,indent=2)+'\n');print('passed')
