"""Feed the actual timing supervisor health schema to the real formal parser."""
import importlib.util,json,os,tempfile
from pathlib import Path
E=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_warm192_formal_smoke_v3',E/'formal_v3.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prior=json.loads((E/'observer_actual_integration_v3_001.json').read_text());h=prior['actual_health_first'];ns=h['time_namespace'];end=h['health_snapshot_window_monotonic_ns'][1]
with tempfile.TemporaryDirectory() as t:
 d=Path(t)
 for n in ('monitor_run001','monitor_observer001'):
  (d/n).mkdir();(d/n/'health.jsonl').write_text(json.dumps(h)+'\n')
 (d/'manual_health_windows001.jsonl').write_text('')
 observation=[{'event':'snapshot','index':0,'time_namespace':ns,'ps':{'query_start_ns':end+1,'query_end_ns':end+2},'docker':{'query_start_ns':end+3,'query_end_ns':end+4}},{'event':'logging_complete','index':0,'time_namespace':ns,'first_fsync_end_ns':end+5}]
 observation.append({'event':'observer_end','reason':'stop','samples':1,'time_namespace':ns,'end_ns':end+6})
 (d/'observations001.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in observation));m.D=d
 health,windows,manual=m.observe_windows()
 assert len(health)==2 and len(windows)==3 and manual is None
with (E/'formal_observer_smoke_v3_001.json').open('x') as f:json.dump({'status':'passed','actual_supervisor_health_schema':True,'formal_parser':'formal_v3.observe_windows','health_windows':len(health),'observer_query_windows':len(windows),'empty_manual':'unknown','GPU_used':False,'GT_opened':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
