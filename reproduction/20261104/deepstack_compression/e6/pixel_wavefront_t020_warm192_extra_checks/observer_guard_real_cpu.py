"""Exercise actual /proc parent, clocks and namespace with a CPU child."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import observer_guard_v2 as guard
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
try:
 value=guard.verify(child.pid,os.getpid(),Path(__file__),a.run_dir,21600,300)
 assert value['observer_pid']==child.pid and value['supervisor_pid']==os.getpid()
 try:guard.verify(child.pid,os.getppid(),Path(__file__),a.run_dir,21600,300)
 except ValueError:pass
 else:raise AssertionError('foreign parent accepted')
 value.update({'status':'passed','scope':'synthetic actual /proc CPU process, not real observer or GPU','GT_opened':False})
 with Path(__file__).with_name('observer_guard_real_cpu001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
finally:
 child.terminate();child.wait(timeout=5)
