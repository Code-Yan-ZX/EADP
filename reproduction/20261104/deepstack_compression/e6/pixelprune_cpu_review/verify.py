"""Read-only source/evidence audit, does not rerun a model or selector."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
EXTERNAL = BASE/'e6/external_sources/pixelprune_8d347eb'
OUT = BASE/'e6/pixelprune_cpu_review/review001.json'
COMMIT = '8d347eb1bdec92c5991057e75d424ecc1ecfe53e'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    head = subprocess.run(['git','-C',str(EXTERNAL),'rev-parse','HEAD'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    dirty = subprocess.run(['git','-C',str(EXTERNAL),'status','--short'], capture_output=True, text=True, check=True, timeout=10).stdout
    if head != COMMIT or dirty: raise ValueError('external source checkout mismatch')
    evidence={}; findings={}
    for name, count in [('smoke',19), ('dedup',21)]:
        d=BASE/f'e6/pixelprune_cpu_{name}'
        plan=read(d/'plan001.json'); result=read(d/'result001.json')
        started=read(d/'monitor001/started.json'); terminal=read(d/'monitor001/terminal.json')
        if plan['source_commit'] != COMMIT or len(plan['sources']) != count: raise ValueError(name+' source inventory')
        if any(sha(path) != digest for path,digest in plan['sources'].items()): raise ValueError(name+' source sha')
        if result['plan_sha256'] != sha(d/'plan001.json'): raise ValueError(name+' plan sha')
        command=started['command']; cwd=Path(started['cwd'])
        if ((cwd/command[-1]).resolve() != (d/'probe.py').resolve() or len(command)!=2
            or started['worker_pid'] != started['process_group'] or started['worker_pid'] != terminal['worker_pid']
            or terminal['reason'] != 'completed' or terminal['worker_returncode'] != 0
            or terminal['remaining_process_group'] != []): raise ValueError(name+' supervisor')
        records=result['records']
        if [x['arm'] for x in records] != ['unpatched','patched_explicit_disabled','patched_enabled_threshold_zero'] or any(x['status'] != 'completed' for x in records): raise ValueError(name+' records')
        if result['cuda_initialized'] is not False or result['disabled_output_equals_unpatched'] is not True or records[0]['output_ids'] != records[1]['output_ids']: raise ValueError(name+' disabled identity')
        lengths=[x['cache_length'] for x in records]
        if name=='smoke':
            if lengths != [9,9,9] or any(x['output_ids'] != records[0]['output_ids'] for x in records): raise ValueError('smoke outputs')
        elif (lengths != [12,12,9] or result.get('original_merged_tokens') != 4
              or result.get('selector_kept_indices') != [[0]]): raise ValueError('dedup outputs')
        paths=[d/'plan001.json', d/'probe.py', d/'result001.json', d/'monitor001/started.json', d/'monitor001/terminal.json']
        evidence[name]={'source_files_verified':count, 'sha256':{str(p):sha(p) for p in paths}}
        findings[name]={'cache_lengths':lengths,'disabled_tokens_equal':True,
                        'output_ids': [x['output_ids'] for x in records],
                        'selector_kept_indices':result.get('selector_kept_indices')}
    report={'status':'passed','scope':'independent read-only evidence review; no model or selector rerun',
            'source_commit':head,'external_worktree_clean':True,'evidence':evidence,'findings':findings,
            'limitations':['smoke has one merged visual token and does not establish pruning',
              'dedup uses four identical synthetic visual tokens; selector [[0]] was a separate direct call, while the actual generation hook was not captured with full arrays',
              'random tiny model, synthetic inputs, CPU eager and four generated tokens; no inference about real-model compatibility, FA2, quality or speed',
              'only final cache length was recorded, not per-layer cache content'],
            'reviewer_sha256':sha(__file__)}
    with OUT.open('x') as f: json.dump(report,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


if __name__=='__main__': main()
