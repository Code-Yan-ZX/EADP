"""Actual parent and bounded supervisor checks before any GPU query."""
import json,os
from pathlib import Path
B=Path(__file__).resolve().parents[2]
def verify(start,expected_script,expected_args,timeout,grace,parent_argv,self_argv,pid,pgid,sid,cwd,monitor,poll):
 cmd=start.get('command',[]);wd=Path(start.get('cwd','/')).resolve()
 if (wd!=Path(cwd).resolve() or type(pid) is not int or pid<=0 or pid!=pgid or pid!=sid or start.get('worker_pid')!=pid or start.get('process_group')!=pid or cmd!=self_argv or len(cmd)<2 or (wd/cmd[1]).resolve()!=Path(expected_script).resolve() or cmd[2:]!=expected_args or start.get('timeout_seconds')!=timeout or start.get('grace_seconds')!=grace or start.get('snapshot_seconds')!=3600 or start.get('poll_seconds')!=poll):raise ValueError('supervised child identity')
 def flag(name):
  if parent_argv.count(name)!=1 or parent_argv.index(name)+1>=len(parent_argv):raise ValueError('supervisor flag')
  return parent_argv[parent_argv.index(name)+1]
 if (sum((wd/x).resolve()==B/'monitor/supervise.py' for x in parent_argv)!=1 or parent_argv.count('--')!=1 or parent_argv[parent_argv.index('--')+1:]!=self_argv or Path(flag('--run-dir')).resolve()!=Path(monitor).resolve() or float(flag('--timeout'))!=timeout or float(flag('--grace-seconds'))!=grace or float(flag('--snapshot-seconds'))!=3600 or float(flag('--poll-seconds'))!=poll):raise ValueError('missing or altered actual supervisor parent')
 return True
def current(directory,script,args,timeout,grace,poll):
 start=json.loads((Path(directory)/'started.json').read_text());parent=[x.decode() for x in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if x];own=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]
 return verify(start,script,args,timeout,grace,parent,own,os.getpid(),os.getpgrp(),os.getsid(0),Path.cwd(),directory,poll)
