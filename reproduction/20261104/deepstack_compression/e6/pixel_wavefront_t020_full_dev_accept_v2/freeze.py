import json
import os
from pathlib import Path
import acceptance
V=Path(__file__).resolve().parent;D=acceptance.D
def main():
    source=[V/x for x in ('acceptance.py','finish_guard.py','snapshot.py','verify.py','test_guard.py','cpu_validation.py','freeze.py')]
    prior=[D/'monitor_snapshot001'/x for x in ('started.json','terminal.json','stderr.log')]
    prior += [D/'monitor_run001'/x for x in ('started.json','terminal.json')]
    prior += [D/'run_run001'/x for x in ('launch.json','exit.json')]
    p={'run_plan_sha256':acceptance.sha(D/'plan002.json'),'source_sha256':{str(x):acceptance.sha(x) for x in source},
       'prior_evidence_sha256':{str(x):acceptance.sha(x) for x in prior},
       'change':'post-run fix: retain run launch record instead of shadowing it with imported module; preserve original sources and failed snapshot',
       'GT_opened':False,'scoring_performed':False}
    with (V/'acceptance_plan002.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
