"""Test fixed entry on real retained container without opening candidate outputs."""
import json
import os
import subprocess
import sys
from pathlib import Path
import acceptance
import finish_guard
V=Path(__file__).resolve().parent
def main():
    p=acceptance.check()
    test=subprocess.run([sys.executable,'-m','unittest','-v','test_guard'],cwd=V,capture_output=True,text=True,timeout=120)
    (V/'test_guard002.log').write_text(test.stdout+test.stderr)
    if test.returncode:raise ValueError('guard synthetic test failed')
    actual=finish_guard.validate()
    with (V/'guard_smoke002.json').open('x') as f:
        json.dump({'status':'passed','acceptance_plan_sha256':acceptance.sha(V/'acceptance_plan002.json'),
                   'actual':actual,'candidate_outputs_opened':False,'GT_opened':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    proof={'status':'passed','acceptance_plan_sha256':acceptance.sha(V/'acceptance_plan002.json'),
           'source_sha256':p['source_sha256'],'guard_smoke_sha256':acceptance.sha(V/'guard_smoke002.json'),
           'test_log_sha256':acceptance.sha(V/'test_guard002.log'),'candidate_outputs_opened':False,'GT_opened':False}
    with (V/'cpu_proof002.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
