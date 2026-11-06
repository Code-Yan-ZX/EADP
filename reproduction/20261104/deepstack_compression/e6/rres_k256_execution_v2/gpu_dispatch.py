"""Record empty-GPU start and pass the real end-of-run callback."""
import json,os
from pathlib import Path

def dispatch(runtime,timing,author,run,worker):
 before=runtime.gpu_pids()
 if before!=[]:raise ValueError('GPU compute processes present before launch')
 run=Path(run);run.mkdir(exist_ok=False);out=run/'observations';out.mkdir();name='eadp-rres-k256-'+str(os.getpid());command=runtime.container_command(author,[str(worker),'--out',str(out)],True,name)
 if command.count('--rm')!=1:raise ValueError('unexpected container command')
 command.remove('--rm')
 with (run/'launch.json').open('x') as f:json.dump({'approval':runtime.approval,'controller':runtime.identity(os.getpid()),'command':command,'name':name,'gpu_pids_before':before},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing.run_bounded(command,name,3600,runtime.gpu_pids,runtime.identity,run,require_gpu_idle=True)
