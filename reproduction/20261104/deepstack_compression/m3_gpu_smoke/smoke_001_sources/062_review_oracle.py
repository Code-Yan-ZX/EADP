"""Independent CPU audit of the frozen M3 synthetic teacher primitives."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ORACLE=HERE.parent/'m3_oracle'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    proof=ORACLE/'cpu_validation001.json';old=json.loads(proof.read_text());start=json.loads((ORACLE/'monitor_cpu_validation001/started.json').read_text())
    terminal=json.loads((ORACLE/'monitor_cpu_validation001/terminal.json').read_text())
    if (old.get('status')!='passed' or old.get('returncode')!=0 or old['source_sha256_before']!=old['source_sha256_after'] or
        terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or
        terminal.get('worker_pid')!=start.get('worker_pid') or 'Ran 12 tests' not in old.get('stderr','')):
        raise ValueError('upstream frozen CPU audit incomplete')
    cwd=Path(start.get('cwd','/'));cmd=start.get('command',[])
    if [(cwd/x).resolve() for x in cmd if Path(x).name=='cpu_validation.py']!=[(ORACLE/'cpu_validation.py').resolve()]:
        raise ValueError('upstream CPU supervisor ran a different script')
    for name,digest in old['source_sha256_after'].items():
        if sha(name)!=digest:raise ValueError('upstream code changed: '+name)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    cmd=[sys.executable,'-m','unittest','discover','-s',str(ORACLE),'-p','test_*.py','-v']
    proc=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=180)
    if proc.returncode or 'Ran 12 tests' not in proc.stderr or not proc.stderr.rstrip().endswith('OK'):
        raise ValueError('independent CPU audit failed: '+proc.stderr[-3000:])
    for name,digest in old['source_sha256_after'].items():
        if sha(name)!=digest:raise ValueError('upstream code changed during review: '+name)
    result={'status':'passed','independent_review':True,'source_sha256':old['source_sha256_after'],
        'upstream_proof_sha256':sha(proof),'upstream_started_sha256':sha(ORACLE/'monitor_cpu_validation001/started.json'),
        'upstream_terminal_sha256':sha(ORACLE/'monitor_cpu_validation001/terminal.json'),
        'reviewer_sha256':sha(__file__),'command':cmd,'returncode':proc.returncode,'stderr':proc.stderr,
        'findings':['P3 prefix positions use prompt length plus native delta; loss predicts each fixed target from its previous position.',
            'All parameters must be eval/frozen and four visual tensors are distinct gradient leaves; input identity is the caller responsibility.',
            'CPU tests do not establish real 8B backward feasibility, GPU numerical identity, memory capacity or label-based quality.',
            'Caller must reject inference_mode, check actual image-token segment and native inputs, and supply no ground-truth targets.'],
        'GT_opened':False,'real_model_loaded':False,'gpu_used':False}
    with (HERE/'oracle_review_001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':'passed','test_count':12}))

if __name__=='__main__':main()
