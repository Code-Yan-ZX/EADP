"""CPU proof for minimal independently located M3 verifier repair."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def expected(original):
 s=original.replace('import argparse,hashlib,json,subprocess,sys','import argparse,hashlib,importlib.util,json,subprocess,sys')
 s=s.replace("HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;SMOKE=ROOT/'m3_gpu_smoke'", "V2=Path(__file__).resolve().parent;ROOT=V2.parent;HERE=ROOT/'m3_oracle_exec';SMOKE=ROOT/'m3_gpu_smoke'")
 return s.replace("    sys.path.insert(0,str(PLAN.parent));from run_panel import alive\n", "    completion_path=ROOT/'e1_analysis/completion.py'\n    spec=importlib.util.spec_from_file_location('m3_verify_v2_completion',completion_path)\n    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);alive=module.alive\n")
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False)
 original=ROOT/'m3_oracle_exec/verify.py';fixed=HERE/'verify.py';completion=ROOT/'e1_analysis/completion.py';snap=ROOT/'m3_oracle_exec/runtime_001_sources/manifest.json'
 manifest=json.loads(snap.read_text());src=manifest['source_sha256']
 if (fixed.read_text()!=expected(original.read_text()) or src.get(str(original.resolve()))!=sha(original) or
     src.get(str(completion.resolve()))!=sha(completion)):raise ValueError('source mismatch or more than intended repair')
 files=[HERE/'verify.py',HERE/'test_verify.py',HERE/'cpu_validation.py',original,completion,snap];before={str(p.resolve()):sha(p) for p in files}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');cmd=[sys.executable,'-m','unittest','-v','test_verify']
 r=subprocess.run(cmd,cwd=HERE,capture_output=True,text=True,env=env,timeout=120)
 (out/'stdout.log').write_text(r.stdout);(out/'stderr.log').write_text(r.stderr);after={str(p.resolve()):sha(p) for p in files}
 proof={'status':'passed' if r.returncode==0 and before==after else 'failed','source_sha256_before':before,'source_sha256_after':after,
        'command':cmd,'returncode':r.returncode,'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),
        'GT_opened':False,'prediction_files_opened':False,'scope':'actual frozen completion.alive import, live/dead process, minimal source diff'}
 with (out/'verification.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if proof['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
