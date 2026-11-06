"""Review the bounded host observer before any timing run starts."""
import argparse
import hashlib
import json
import os
from pathlib import Path

R=Path(__file__).resolve().parent
B=R.parent
D=B/'e4_c_multibudget_environment'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def terminal(folder):
    s,t=read(folder/'started.json'),read(folder/'terminal.json')
    if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or
        t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid')):
        raise ValueError('supervision mismatch')
    return s


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve()
    if out!=R/'e4c_multibudget_environment_prelaunch_review.json':raise ValueError('unplanned proof')
    p=read(D/'plan001.json');sources={path:sha(path) for path in p['source_sha256']}
    if (sources!=p['source_sha256'] or p.get('GT_opened') is not False or p.get('scoring_performed') is not False or
        p.get('GPU_authorized') is not False):raise ValueError('plan/source mismatch')
    s=terminal(D/'monitor_cpu001');f=terminal(D/'monitor_freeze001')
    if (s['command'][1:4]!=['-m','unittest','discover'] or 'test_validate.py' not in s['command'] or
        'Ran 2 tests' not in (D/'monitor_cpu001/stderr.log').read_text() or
        (Path(f['cwd'])/f['command'][1]).resolve()!=D/'freeze_plan.py'):
        raise ValueError('test or freeze command mismatch')
    c=p['command']
    def value(flag):
        if c.count(flag)!=1:raise ValueError('observer argument '+flag)
        return c[c.index(flag)+1]
    if ((Path(p['cwd'])/c[1]).resolve()!=B/'e4_c_latency/timing_supervise.py' or
        (Path(p['cwd'])/value('--run-dir')).resolve()!=D/'monitor_watch001' or
        (Path(p['cwd'])/value('--output')).resolve()!=D/'samples001.jsonl' or
        (Path(p['cwd'])/value('--stop')).resolve()!=D/'stop001' or
        value('--timeout')!='7620' or value('--duration')!='7500' or value('--interval')!='60' or
        value('--snapshot-seconds')!='3600' or value('--poll-seconds')!='5' or value('--grace-seconds')!='30' or
        c.count('--')!=1 or (Path(p['cwd'])/c[c.index('--')+2]).resolve()!=D/'watch.py'):
        raise ValueError('observer launch contract')
    if (D/'samples001.jsonl').exists() or (D/'stop001').exists() or (D/'monitor_watch001').exists():
        raise ValueError('observer already started')
    proof={'status':'passed','independent_review':True,'scope':'multibudget_environment_prelaunch',
           'plan_sha256':sha(D/'plan001.json'),'source_sha256':sources,
           'cpu_started_sha256':sha(D/'monitor_cpu001/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu001/terminal.json'),
           'reviewer_sha256':sha(__file__),'GT_opened':False,'timing_results_opened':False,'errors':[],
           'limitations':['ps CPU usage is lifetime average, not instantaneous; sampling is sparse.',
              'Second log fsync and residual host perturbation remain unknown; retain all timing samples.',
              'The actual observer must start before model work and stop after it, with matching time namespace.',
              'Observer and model supervisor health windows and user-required manual checks need separate coverage.']}
    with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


if __name__=='__main__':main()
