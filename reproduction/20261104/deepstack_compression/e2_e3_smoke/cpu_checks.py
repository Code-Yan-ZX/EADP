"""Run isolated bounded CPU tests and bind all diagnostic source hashes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from local_imports import load_local
gate=load_local('gate');HERE=gate.HERE;CONTRACT=gate.CONTRACT;SOURCES=gate.SOURCES;sha=gate.sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
    before={str(p):sha(p) for p in SOURCES};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    results=[];start=time.time()
    for i,cwd in enumerate((CONTRACT,HERE)):
        cmd=[sys.executable,'-m','unittest','discover','-p','test_*.py','-v']
        try:
            proc=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=180)
            code=proc.returncode;stdout=proc.stdout;stderr=proc.stderr;timeout=False
        except subprocess.TimeoutExpired as e:
            code=None;stdout=str(e.stdout or '');stderr=str(e.stderr or '');timeout=True
        (out/f'stdout_{i}.log').write_text(stdout);(out/f'stderr_{i}.log').write_text(stderr)
        results.append({'command':cmd,'cwd':str(cwd),'returncode':code,'timeout':timeout,'stdout_sha256':sha(out/f'stdout_{i}.log'),'stderr_sha256':sha(out/f'stderr_{i}.log'),'ok':code==0 and not timeout and stderr.rstrip().endswith('OK')})
    after={str(p):sha(p) for p in SOURCES};ok=all(x['ok'] for x in results) and before==after
    proof={'status':'passed' if ok else 'failed','returncode':0 if ok else 1,'source_sha256_before':before,'source_sha256_after':after,'checks':results,'GT_opened':False,'scoring_performed':False,'start_unix':start,'end_unix':time.time()}
    with (out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
