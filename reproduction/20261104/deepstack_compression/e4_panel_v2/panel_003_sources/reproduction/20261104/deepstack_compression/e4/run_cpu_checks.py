"""Run isolated CPU checks and bind the observed exit to exact source hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/media/disk2/YZX/research/EADP_v2')
AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'
PRUNER=ROOT/'_reproduction/author_eadp/Qwen_vl/model/pruner.py'
VISION=ROOT/'_reproduction/envs/qwen_author/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py'
CONFIG=VISION.parent/'configuration_qwen3_vl.py'
OLD=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02'
LAUNCH_DEPENDENCIES=[OLD/'run_plan.json',OLD/'run_panel.py',OLD/'diagnose.py',OLD.parent/'container_runtime.py']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[HERE/'skip_deepstack.py',HERE/'off_adapter.py',HERE/'identity_records.py',HERE/'identity_worker.py',
             HERE/'budget_adapter.py',HERE/'budget_runner.py',HERE/'budget_reuse.py',HERE/'verify_k256.py',HERE/'smoke_contract.py',HERE/'smoke_launch.py',HERE/'smoke_worker.py',HERE/'test_budget_adapter.py',
             HERE/'panel_worker.py',HERE/'panel_launch.py',HERE/'panel_gate.py',HERE/'panel_verify.py',HERE/'decode_observer.py',HERE/'test_panel.py',
             HERE/'test_skip_deepstack.py',HERE/'test_off_adapter.py',HERE/'run_cpu_checks.py',
             HERE.parent/'e1_analysis/completion.py',HERE.parent/'e1_analysis/score.py',HERE.parent/'e1_analysis/metrics.py',
             HERE.parent/'e1_analysis/paired_stats.py',HERE.parent/'e1/contract.py',AUTHOR,PRUNER,VISION,CONFIG]+LAUNCH_DEPENDENCIES
    before={str(x):sha(x) for x in sources}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','unittest','-v','test_skip_deepstack','test_off_adapter','test_budget_adapter','test_panel']
    start=time.time()
    try:
        run=subprocess.run(command,cwd=HERE,env=env,capture_output=True,text=True,timeout=120)
        code=run.returncode;stdout=run.stdout;stderr=run.stderr;timeout=False
    except subprocess.TimeoutExpired as exc:
        code=None;stdout=(exc.stdout or b'').decode(errors='replace') if isinstance(exc.stdout,bytes) else (exc.stdout or '')
        stderr=(exc.stderr or b'').decode(errors='replace') if isinstance(exc.stderr,bytes) else (exc.stderr or '')
        timeout=True
    (out/'stdout.log').write_text(stdout)
    (out/'stderr.log').write_text(stderr)
    after={str(x):sha(x) for x in sources}
    passed=code==0 and not timeout and before==after and 'Ran 27 tests' in stderr and stderr.rstrip().endswith('OK')
    result={'status':'passed' if passed else 'failed','command':command,'returncode':code,'timeout':timeout,
            'started_unix':start,'finished_unix':time.time(),'python':sys.version,
            'cuda_visible_devices':env['CUDA_VISIBLE_DEVICES'],'source_sha256_before':before,'source_sha256_after':after,
            'stdout_sha256':sha(out/'stdout.log'),'stderr_sha256':sha(out/'stderr.log'),
            'scope':'Synthetic and randomly initialized tiny Qwen3 vision CPU only; no benchmark, trained weights, GPU, generation or downstream identity.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':result['status'],'returncode':code,'seconds':result['finished_unix']-start}))
    return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
