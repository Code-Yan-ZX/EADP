"""Read-only aggregation of saved host process snapshots; no job timing values."""
import hashlib,json,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/pixel_natural_timing_pilot_v4/monitor_pilot001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rows=[json.loads(x) for x in (D/'process_samples001.jsonl').read_text().splitlines()];assert rows and all(x.get('returncode')==0 and x['query_start_ns']<=x['query_end_ns'] for x in rows) and all(a['query_end_ns']<=b['query_start_ns'] for a,b in zip(rows,rows[1:]))
java=[];counts=[];snapshot=[]
for i,x in enumerate(rows):
 cur=[]
 for line in x['stdout'].splitlines()[1:]:
  fields=line.split(None,6)
  if len(fields)==7 and fields[6].startswith('java '):
   q={'pid':int(fields[1]),'ps_percent_cpu_lifetime':float(fields[2]),'elapsed':fields[5]};java.append(q);cur.append(q)
 counts.append(len(cur));snapshot.append({'index':i,'query_start_ns':x['query_start_ns'],'query_end_ns':x['query_end_ns'],'java_processes':cur})
b=read(D/'before001.json');health=[json.loads(x) for x in (P/'health.jsonl').read_text().splitlines()];terminal=read(P/'terminal.json');assert terminal['reason']=='completed' and terminal['worker_returncode']==0 and terminal['remaining_process_group']==[]
result={'status':'completed','scope':'saved low-frequency host ps snapshots; no TTFT or outcome analysis','snapshot_count':len(rows),'all_queries_exit0':True,'first_query_start_ns':rows[0]['query_start_ns'],'last_query_end_ns':rows[-1]['query_end_ns'],'max_query_seconds':max((x['query_end_ns']-x['query_start_ns'])/1e9 for x in rows),'max_consecutive_start_gap_seconds':max((b['query_start_ns']-a['query_start_ns'])/1e9 for a,b in zip(rows,rows[1:])),'java_count_per_snapshot_min_max':[min(counts),max(counts)],'distinct_java_pids':len({x['pid'] for x in java}),'highest_ps_cpu_percent_lifetime':max(x['ps_percent_cpu_lifetime'] for x in java),'snapshots':snapshot,'pre_snapshot_utc':b['utc'],'pre_snapshot_monotonic_ns':b['monotonic_ns'],'pilot_started_utc':read(P/'started.json')['created_utc'],'pilot_terminal_utc':terminal['utc'],'health_snapshot_count':len(health),'clock_and_scope_limitations':['ps %CPU is process lifetime CPU ratio and may exceed 100% for multithreaded processes; it is not instantaneous or causal load.','Each query window identifies when ps ran; the 15-second sampling gaps cannot exclude unobserved activity.','The one preceding UTC/monotonic pair allows only approximate alignment to supervisor UTC if clock remains stable; measured jobs lack per-job monotonic boundaries, so do not assign individual samples or effects to an arm.','The snapshot monitor itself calls ps, fsync and sleeps; its potential impact is unquantified.','Initial GPU compute list was empty but graphical/device background is not ruled out; do not discard timings or claim controlled isolation.'],'source_sha256':{str(x):sha(x) for x in (D/'watch.py',D/'before001.json',D/'process_samples001.jsonl',P/'started.json',P/'terminal.json',P/'health.jsonl',Path(__file__))}}
(D/'background_summary001.json').write_text(json.dumps(result,indent=2)+'\n');print('snapshots',len(rows),'distinct_java_pids',result['distinct_java_pids'],'java_per_sample',result['java_count_per_snapshot_min_max'],'max_lifetime_cpu_pct',result['highest_ps_cpu_percent_lifetime'])
