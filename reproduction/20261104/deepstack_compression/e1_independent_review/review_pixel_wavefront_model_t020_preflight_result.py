"""Independent no-GPU container-result review; no GPU/model identity claim."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

R=Path(__file__).resolve().parent
B=R.parent
D=B/'e6/pixel_wavefront_model_t020'


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def main():
    a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=Path);args=a.parse_args()
    plan=read(D/'preflight_plan002.json');run=D/'run_preflight001';w=read(run/'worker_preflight.json')
    s=read(D/'monitor_preflight001/started.json');t=read(D/'monitor_preflight001/terminal.json')
    launch=read(run/'launch.json');exit_=read(run/'exit.json');ident=read(run/'container_identity.json')
    if (t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or
        s['worker_pid']!=t['worker_pid'] or s['worker_pid']!=launch['controller']['pid'] or
        s['command'][-3:]!=['e6/pixel_wavefront_model_t020/preflight_launch.py','--out',str(run)]):
        raise ValueError('supervisor identity or exit mismatch')
    if (exit_['returncode']!=0 or exit_['container_exit_code']!=0 or
        exit_['container_stopped_verified'] is not True or exit_['timed_out'] is not False or
        exit_['controller']!=launch['controller'] or exit_['container_identity']!=ident or
        exit_['last_inspect']['host_devices']!=[] or exit_['last_inspect']['device_requests'] is not None):
        raise ValueError('container exit/device mismatch')
    actual=subprocess.run(['docker','inspect',ident['container_id']],capture_output=True,text=True,timeout=15,check=True)
    obj=json.loads(actual.stdout)[0];state=obj['State'];host=obj['HostConfig']
    if (obj['Id']!=ident['container_id'] or obj['Image']!=ident['image'] or state['Running'] or
        state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] or host['NetworkMode']!='none' or
        host['Devices'] not in (None,[]) or host['DeviceRequests'] not in (None,[])):
        raise ValueError('actual container state/device mismatch')
    if ('CUDA_VISIBLE_DEVICES=' not in obj['Config']['Env'] or
        obj['Config']['Cmd']!=[str(D/'preflight_entry.py'),'--out',str(run/'worker_preflight.json')]):
        raise ValueError('actual container command/environment mismatch')
    if (w['status']!='passed' or w['weights_verified']!=4 or w['cuda_initialized'] is not False or
        w['model_loaded'] is not False or w['GT_opened'] is not False or w['gpu_authorized'] is not False or
        w['author_patch_before_model'] is not True or w['source_sha256']!=plan['source_sha256'] or
        w['approval']!=launch['approval'] or w['approval']['plan_sha256']!=sha(D/'preflight_plan002.json') or
        w['approval']['review_sha256']!=sha(D/'independent_preflight_review.json')):
        raise ValueError('worker preflight or approval mismatch')
    expected={(x['anchor']['task'],str(x['anchor']['qid'])):(x['anchor']['group_id'],x['expected']['processor']) for x in plan['images']}
    observed={(x['task'],str(x['qid'])):(x['group_id'],x['processor']) for x in w['images']}
    if len(expected)!=3 or len(observed)!=3 or expected!=observed:raise ValueError('processor identity mismatch')
    if any(sha(p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('source changed after preflight')
    aliases=plan['model_metadata_aliases_sha256']
    if len(aliases)!=7 or any(plan['source_sha256'].get(p)!=h or sha(p)!=h for p,h in aliases.items()):
        raise ValueError('metadata alias content mismatch')
    proof={'status':'passed','independent_review':True,'scope':'three-anchor t020 real no-GPU preflight result only',
           'GT_opened':False,'gpu_authorized':False,'images':3,'weights_verified':4,'metadata_aliases_verified':7,
           'plan_sha256':sha(D/'preflight_plan002.json'),'worker_preflight_sha256':sha(run/'worker_preflight.json'),
           'launch_sha256':sha(run/'launch.json'),'exit_sha256':sha(run/'exit.json'),
           'started_sha256':sha(D/'monitor_preflight001/started.json'),'terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),
           'source_sha256':plan['source_sha256'],'reviewer_sha256':sha(__file__),'errors':[],
           'limitations':['No model loaded, no GPU model identity or timing measured.',
                          'Seven model paths are mount aliases with verified content, not identical host and container resolved targets.',
                          'Existing DEV predictions were used only for identity.']}
    with args.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


if __name__=='__main__':main()
