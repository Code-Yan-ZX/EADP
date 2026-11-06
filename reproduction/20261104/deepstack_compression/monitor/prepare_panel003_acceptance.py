"""After a clean panel_003 exit, preserve approved sources and start CPU acceptance."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE=Path(__file__).resolve().parent.parent;PROJECT=BASE.parents[2];PANEL=BASE/'e4_panel_v2'
RUN=PANEL/'panel_003';SUP=PANEL/'monitor_panel003'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    ended=json.loads((SUP/'terminal.json').read_text());started=json.loads((SUP/'started.json').read_text())
    exited=json.loads((RUN/'exit.json').read_text());launch=json.loads((RUN/'launch.json').read_text())
    if (ended.get('reason')!='completed' or ended.get('worker_returncode')!=0 or ended.get('remaining_process_group')!=[] or
        ended.get('worker_pid')!=started.get('worker_pid') or exited.get('returncode')!=0 or
        exited.get('container_exit_code')!=0 or exited.get('container_stopped_verified') is not True or
        exited.get('timed_out') is not False or exited.get('interrupted') is not None or
        exited.get('gpu_pids_after')!=[] or not (RUN/'worker_complete.json').is_file()):
        raise ValueError('panel_003 is not completely and cleanly terminated')
    container=exited['container_identity']['container_id']
    probe=subprocess.run(['docker','inspect',container],capture_output=True,text=True,timeout=10,check=True)
    obj=json.loads(probe.stdout)[0]
    if obj['Id']!=container or obj['State']['Running'] or obj['State']['ExitCode']!=0:
        raise ValueError('panel container not independently confirmed stopped')
    frozen=json.loads((RUN/'frozen.json').read_text());cpu=json.loads((PANEL/'cpu_check_005/verification.json').read_text())
    if cpu.get('status')!='passed' or cpu['source_sha256_before']!=cpu['source_sha256_after']:
        raise ValueError('approved CPU source evidence invalid')
    sources=dict(cpu['source_sha256_after'])
    for name,digest in frozen['code_sha256'].items():
        if name in sources and sources[name]!=digest:raise ValueError('runtime source differs from approved source')
        sources[name]=digest
    sources[str((PANEL/'panel_verify.py').resolve())]=sha(PANEL/'panel_verify.py')
    snapshot=PANEL/'panel_003_sources';snapshot.mkdir(exist_ok=False);copied={}
    for name,digest in sorted(sources.items()):
        path=Path(name)
        try:relative=path.relative_to(PROJECT)
        except ValueError:raise ValueError('source path outside project root: '+name)
        if sha(path)!=digest:raise ValueError('runtime source changed: '+name)
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        if sha(target)!=digest or sha(path)!=digest:raise ValueError('source snapshot mismatch: '+name)
        copied[name]={'snapshot':str(target),'sha256':digest}
    manifest={'panel_run':str(RUN),'launch_sha256':sha(RUN/'launch.json'),'exit_sha256':sha(RUN/'exit.json'),
              'frozen_sha256':sha(RUN/'frozen.json'),'cpu_proof_sha256':sha(PANEL/'cpu_check_005/verification.json'),
              'independent_review_sha256':sha(PANEL/'independent_review_005.json'),'sources':copied}
    with (snapshot/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    original=started['command'];worker=[sys.executable,str(PANEL/'panel_verify.py')]
    for name in ('e1-acceptance','e1-started','e1-terminal','smoke-dir','smoke-proof','smoke-started','smoke-terminal'):
        flag='--'+name
        if original.count(flag)!=1:raise ValueError('missing panel launcher input: '+flag)
        worker += [flag,original[original.index(flag)+1]]
    worker += ['--run-dir',str(RUN),'--run-started',str(SUP/'started.json'),'--run-terminal',str(SUP/'terminal.json'),
               '--out',str(PANEL/'panel_003_verification.json')]
    command=[sys.executable,str(BASE/'monitor/supervise.py'),'--run-dir',str(PANEL/'monitor_verify_panel003'),
             '--timeout','3600','--snapshot-seconds','3600','--poll-seconds','15','--grace-seconds','30',
             '--disk',str(PANEL),'--']+worker
    with (PANEL/'verify_panel003_console.log').open('x') as log:
        proc=subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'supervisor_pid':proc.pid,'snapshot_manifest':str(snapshot/'manifest.json'),'command':command}))

if __name__=='__main__':main()
