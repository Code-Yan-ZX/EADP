"""No-GPU check that the frozen external observer is already live."""
import json,os,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
def live(clock=time.monotonic_ns):
 m=D/'monitor_observer001';s=json.loads((m/'started.json').read_text());cmd=s['command'];cwd=Path(s['cwd']);pid=s.get('worker_pid')
 expected=[str((D/'observer.py').resolve()),'--output',str((D/'observations001.jsonl').resolve()),'--stop',str((D/'stop001').resolve()),'--interval','60','--duration','23000']
 if type(pid) is not int or pid<=0 or s.get('process_group')!=pid or len(cmd)<2 or (cwd/cmd[1]).resolve()!=D/'observer.py' or cmd[1:]!=expected or (m/'terminal.json').exists() or (D/'stop001').exists():raise ValueError('observer supervision/arguments')
 argv=[x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
 if argv!=cmd:raise ValueError('observer process identity')
 lines=(D/'observations001.jsonl').read_text().splitlines();events=[]
 for line in lines:
  try:events.append(json.loads(line))
  except json.JSONDecodeError:break
 complete=[x for x in events if x.get('event')=='logging_complete'];snap=[x for x in events if x.get('event')=='snapshot']
 if not complete or not snap or complete[-1]['index'] not in {x.get('index') for x in snap} or type(complete[-1].get('first_fsync_end_ns')) is not int or not 0<=clock()-complete[-1]['first_fsync_end_ns']<300_000_000_000:raise ValueError('observer has no recent complete snapshot')
 return {'worker_pid':pid,'first_fsync_end_ns':complete[-1]['first_fsync_end_ns'],'time_namespace':complete[-1]['time_namespace']}
