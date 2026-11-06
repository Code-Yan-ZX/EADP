"""Run isolated CPU contracts and exercise the real launcher refusal path."""
import json
import os
import subprocess
import sys
from pathlib import Path
import admission

D=Path(__file__).resolve().parent
def main():
    p=admission.check_plan()
    if (D/'independent_preflight_review.json').exists() or (D/'run_preflight001').exists():
        raise ValueError('CPU test requires unapproved unused run directory')
    logs={}
    for name in ('test_compare','test_verify','test_selector','test_worker'):
        result=subprocess.run([sys.executable,'-m','unittest','-v',name],cwd=D,capture_output=True,text=True,timeout=120)
        (D/(name+'_cpu002.log')).write_text(result.stdout+result.stderr)
        if result.returncode!=0:raise ValueError('CPU test failed: '+name)
        logs[name]=admission.sha(D/(name+'_cpu002.log'))
    smoke=subprocess.run([sys.executable,str(D/'launch.py'),'--dependency-smoke'],cwd=D,capture_output=True,text=True,timeout=120)
    (D/'dependency_smoke_cpu002.log').write_text(smoke.stdout+smoke.stderr)
    if smoke.returncode!=0 or 'actual cwd launcher dependencies loaded' not in smoke.stdout:
        raise ValueError('actual launcher dependency smoke failed')
    result=subprocess.run([sys.executable,str(D/'launch.py'),'--stage','preflight'],cwd=D,capture_output=True,text=True,timeout=120)
    (D/'launch_rejection_cpu002.log').write_text(result.stdout+result.stderr)
    if result.returncode==0 or 'monitor_pixel_wavefront_full_preflight' not in result.stderr or (D/'run_preflight001').exists():
        raise ValueError('real launcher did not fail closed before Docker')
    proof={'status':'passed','plan_sha256':admission.sha(D/'plan002.json'),'source_sha256':p['source_sha256'],
           'test_logs':logs,'dependency_smoke_sha256':admission.sha(D/'dependency_smoke_cpu002.log'),'launch_rejection_sha256':admission.sha(D/'launch_rejection_cpu002.log'),
           'GT_opened':False,'predictions_opened':True,'scoring_performed':False,
           'prediction_scope':'accepted DEV files only opened for frozen SHA; no labels'}
    with (D/'cpu_proof002.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print('CPU contracts and actual unapproved launch rejection passed')
if __name__=='__main__':main()
