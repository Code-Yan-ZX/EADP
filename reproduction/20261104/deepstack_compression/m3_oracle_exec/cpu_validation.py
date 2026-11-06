"""Bounded CPU-only synthetic tests and source-bound evidence; no labels/GPU."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def sources():
    protocol=json.loads((ROOT/'m3_oracle/protocol_001.json').read_text());run=json.loads((ROOT/'m3_gpu_smoke/cpu_check_006/verification.json').read_text())
    files=list(HERE.glob('*.py'))+[HERE/'protocol_amendment_001.json',HERE/'runtime_plan.json',ROOT/'m3_oracle/protocol_001.json']
    for collection in (protocol['source_sha256'],protocol['cpu_source_sha256'],run['source_sha256_after']):
        for name,digest in collection.items():
            if sha(name)!=digest:raise ValueError('upstream source changed: '+name)
            files.append(Path(name))
    return sorted(set(x.resolve() for x in files))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
    files=sources();before={str(p):sha(p) for p in files};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    reports=[];begin=time.time()
    for name in ('test_search.py','test_teacher_backend.py','test_selection_adapter.py','test_generate_backend.py','test_preflight.py','test_extract.py','test_baseline.py','test_e1_baseline.py','test_one_question.py','test_e1_admission.py','test_control.py','test_panel.py','test_lazy_truth.py','test_runtime_gate.py','test_verify_contract.py','test_verify_exit.py','test_container_identity.py'):
        command=[sys.executable,str(HERE/name),'-v']
        try:
            r=subprocess.run(command,capture_output=True,text=True,env=env,timeout=120);rc=r.returncode;stdout=r.stdout;stderr=r.stderr;timed=False
        except subprocess.TimeoutExpired as e:rc=None;stdout=str(e.stdout or '');stderr=str(e.stderr or '');timed=True
        (out/(name+'.stdout')).write_text(stdout);(out/(name+'.stderr')).write_text(stderr)
        reports.append({'test':name,'command':command,'returncode':rc,'timeout':timed,'stdout_sha256':sha(out/(name+'.stdout')),'stderr_sha256':sha(out/(name+'.stderr'))})
    after={str(p):sha(p) for p in files};ok=before==after and all(x['returncode']==0 and not x['timeout'] for x in reports)
    proof={'status':'passed' if ok else 'failed','source_sha256_before':before,'source_sha256_after':after,'tests':reports,
           'started_unix':begin,'finished_unix':time.time(),'cuda_visible_devices':'','GT_opened':False,
           'real_model_loaded':False,'generation_performed':False,'scope':'synthetic CPU-only contract tests; real execution remains locked'}
    with (out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
