"""Independent CPU completion checks; caller supplies frozen identities."""
import hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check(start,terminal,expected_command,expected_cwd,expected_sources,before,after):
 if (start.get('command')!=expected_command or Path(start.get('cwd','/')).resolve()!=Path(expected_cwd).resolve()
     or type(start.get('worker_pid')) is not int or start['worker_pid']<=0 or terminal.get('worker_pid')!=start['worker_pid']
     or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[]):raise ValueError('run supervisor incomplete or wrong command')
 if not isinstance(expected_sources,dict) or not expected_sources or any(sha(k)!=v for k,v in expected_sources.items()):raise ValueError('source inventory changed')
 if (not isinstance(before,list) or len(before)!=4 or before!=after or len({v.get('path') for v in before})!=4
     or any(not isinstance(v.get('path'),str) or not isinstance(v.get('stat'),list) or len(v['stat'])!=5 or any(type(x) is not int or x<0 for x in v['stat']) for v in before)):
  raise ValueError('weight stat changed or inventory incomplete')
 return {'supervisor_completed':True,'weight_stat_identity_checked':True,'limitations':['weight stat does not rule out transient changes or replacement with restored metadata','no real Docker inspection or model replay by this check']}
