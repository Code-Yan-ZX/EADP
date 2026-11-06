"""Real-sampler controller building blocks; no CLI or GPU authorization."""
import sys
import subprocess
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(HERE))
from controller_adapter import ControllerAdapter
from controller_io import publish
from docker_guard import DockerGuard
from interruptible import run_control
from guard import parse_inventory,validate_running


def query_gpu(run=subprocess.run):
    inventory=run(['nvidia-smi','--query-gpu=index,name,uuid','--format=csv,noheader,nounits'],
                  capture_output=True,text=True,check=True,timeout=10).stdout
    compute=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],
                capture_output=True,text=True,check=True,timeout=10).stdout
    return inventory,compute


def validate_worker_proc(proc, cid, expected):
    proc=Path(proc)
    argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if argv!=expected or cid not in (proc/'cgroup').read_text():
        raise ValueError('worker host argv/cgroup mismatch')


class ProfiledAdapter(ControllerAdapter):
    def __init__(self,*args,image,create_argv,worker_argv,**kwargs):
        super().__init__(*args,**kwargs)
        self.image=image;self.create_argv=list(create_argv);self.worker_argv=list(worker_argv)

    def observe_running(self,cid):
        item=self.guard.inspect(cid)
        if item.get('container_id')!=cid:raise ValueError('container identity changed')
        actual=validate_running(item['raw'],self.create_argv,self.image,self.guard.name,self.worker_argv)
        pid=actual['host_pid']
        proc=Path('/proc')/str(pid)
        expected=[self.create_argv[self.create_argv.index('--entrypoint')+1]]+self.worker_argv
        validate_worker_proc(proc,cid,expected)
        publish(self.directory/'running_inspect.json',[item['raw']])
        identity=super().observe_running(cid)
        if identity.get('host_pid')!=pid:raise ValueError('worker PID changed during identity binding')
        return identity


def construct(*,directory,mode,nonce,gpu_uuid,create_argv,name,image,worker_argv,
              sampler_command,sampler_worker_command,sampler_monitor,
              worker_timeout=1800,ready_timeout=300,max_gap_ns=5_000_000_000):
    """Build candidate; caller must establish independent admission first."""
    if mode not in ('t0_author','t020_author','t020_wavefront'):raise ValueError('invalid Pixel arm')
    guard=DockerGuard(name,nonce,create_argv)
    return ProfiledAdapter(guard,directory,mode,nonce,gpu_uuid,max_gap_ns,worker_timeout,
                           sampler_command,sampler_worker_command,sampler_monitor,
                           ready_timeout=ready_timeout,image=image,create_argv=create_argv,
                           worker_argv=worker_argv)


def lifecycle(ops):
    """Bounded real sampler handshake, stop/kill on error; not an admission."""
    return run_control(ops)


def guarded_lifecycle(ops,query):
    """Query is injected; caller must persist and supervise this whole process."""
    def observed(phase):
        start=time.monotonic_ns()
        evidence={'phase':phase,'query_start_ns':start,'query_end_ns':None,
                  'inventory':None,'compute_processes':None,'query_error':None}
        try: inventory,compute=query();evidence.update(inventory=inventory,compute_processes=compute)
        except BaseException as exc:
            evidence['query_error']={'type':type(exc).__name__,'message':str(exc)}
            evidence['query_end_ns']=time.monotonic_ns();publish(ops.directory/(phase+'_gpu_query.json'),evidence)
            raise
        evidence['query_end_ns']=time.monotonic_ns()
        publish(ops.directory/(phase+'_gpu_query.json'),evidence)
        uuid=parse_inventory(inventory,compute)
        if uuid!=ops.gpu_uuid:raise ValueError('host GPU UUID changed')
    observed('before') # blocks creation and release when occupied/unknown
    result=None
    try:
        result=lifecycle(ops)
        return result
    finally:
        # A failed/unknown final GPU observation cannot become a successful run.
        observed('after')


if __name__=='__main__':raise SystemExit('locked: no real GPU controller CLI exists')
