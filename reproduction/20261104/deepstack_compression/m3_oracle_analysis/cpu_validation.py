"""Source-bound synthetic-only CPU scoring tests; no predictions or GT."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
    files=list(HERE.glob('*.py'))+[HERE/'plan.json',ROOT/'m3_oracle/protocol_001.json',ROOT/'m3_oracle_exec/protocol_amendment_001.json']
    files += [ROOT/'e1_analysis'/x for x in ('score.py','metrics.py','completion.py')]
    files += [ROOT/'m3_oracle_exec'/x for x in ('runtime_gate.py','verify.py','verify_contract.py','container_identity.py','e1_admission.py','e1_baseline.py')]
    before={str(p.resolve()):sha(p) for p in files};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');results=[];start=time.time()
    for name in ('test_paired.py','test_gate.py','test_native_predictions.py','test_score_pipeline.py','test_analysis_gate.py'):
        cmd=[sys.executable,str(HERE/name),'-v']
        try:r=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=120);code=r.returncode;stdout=r.stdout;stderr=r.stderr;timed=False
        except subprocess.TimeoutExpired as e:code=None;stdout=str(e.stdout or '');stderr=str(e.stderr or '');timed=True
        (out/(name+'.stdout')).write_text(stdout);(out/(name+'.stderr')).write_text(stderr)
        results.append({'test':name,'returncode':code,'timeout':timed,'stdout_sha256':sha(out/(name+'.stdout')),'stderr_sha256':sha(out/(name+'.stderr'))})
    after={str(p.resolve()):sha(p) for p in files};ok=before==after and all(x['returncode']==0 and not x['timeout'] for x in results)
    proof={'status':'passed' if ok else 'failed','source_sha256_before':before,'source_sha256_after':after,'tests':results,
           'GT_opened':False,'predictions_opened':False,'scoring_performed':False,'gpu_used':False,'started_unix':start,'finished_unix':time.time()}
    with (out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
