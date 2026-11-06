"""Independent real no-device preflight review; no model or new predictions."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

B=Path(__file__).resolve().parents[1];D=B/'e6/pixel_wavefront_t020_warm192';I=B/'e1_independent_review';ROOT=B.parents[2]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(mon,script,out=None):
    s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
    if (len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0
        or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0
        or t.get('remaining_process_group')!=[] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()))):
        raise ValueError('supervision mismatch')
    return s
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.resolve()!=(I/'pixel_t020_warm192_preflight_result_review.json').resolve():raise ValueError('output path')
    plan=read(D/'preflight_plan001.json');source=plan['source_sha256'];design=read(D/'design001.json')
    if any(sha(k)!=v for k,v in source.items()):raise ValueError('source changed')
    prior=I/'pixel_t020_warm192_cpu_preflight_review.json';reviewer=I/'review_pixel_t020_warm192_cpu_preflight.py'
    terminal(I/'monitor_pixel_t020_warm192_cpu_preflight',reviewer,prior);q=read(prior)
    if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('plan_sha256')!=sha(D/'preflight_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('prior review')
    s=terminal(D/'monitor_preflight001',D/'preflight_launch.py',D/'run_preflight001');launch=read(D/'run_preflight001/launch.json');ident=read(D/'run_preflight001/container_identity.json');ex=read(D/'run_preflight001/exit.json');worker=read(D/'run_preflight001/worker_preflight.json')
    if (launch.get('approval')!={'plan_sha256':sha(D/'preflight_plan001.json'),'review_sha256':sha(prior)}
        or launch.get('controller',{}).get('pid')!=s['worker_pid'] or ex.get('controller')!=launch.get('controller')
        or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False
        or ex.get('failure_type') is not None or ex.get('container_stopped_verified') is not True):raise ValueError('launcher exit')
    cid=ex['container_identity']['container_id']
    if ident!=ex['container_identity'] or ex['last_inspect']['id']!=cid or not cid in ex['container_identity']['host_cgroup']:raise ValueError('container host identity')
    # Reconstruct command using frozen author runtime, then independently inspect actual image/config.
    author_path=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';author=read(author_path)
    path=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';spec=importlib.util.spec_from_file_location('_independent_warm_author_loader',path);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader);runtime=loader.load_author(source)
    cmd=runtime.container_command(author,[str(D/'preflight_worker.py'),'--out',str(D/'run_preflight001/worker_preflight.json')],False,launch['name'])
    if cmd.count('--rm')!=1:raise ValueError('bad fixed command')
    cmd.remove('--rm')
    if cmd!=launch['command']:raise ValueError('unexpected container command')
    obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0]
    cfg=obj['Config'];host=obj['HostConfig'];state=obj['State'];image=author['container_image']
    if (obj['Id']!=cid or obj['Name']!='/'+launch['name'] or obj['Image']!=image or cfg['Image']!=image
        or state['Running'] is not False or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] is not False
        or cfg['Entrypoint']!=[cmd[cmd.index('--entrypoint')+1]] or cfg['User']!=cmd[cmd.index('--user')+1]
        or cfg['WorkingDir']!=cmd[cmd.index('--workdir')+1] or cfg['Cmd']!=cmd[cmd.index(image)+1:]
        or host['NetworkMode']!=cmd[cmd.index('--network')+1] or host.get('PidMode') not in ('',None)
        or host.get('Devices') not in (None,[]) or host.get('DeviceRequests') not in (None,[])):
        raise ValueError('actual container image/process/device/state mismatch')
    for env in (cmd[i+1] for i,x in enumerate(cmd[:-1]) if x=='--env'):
        if [x for x in cfg['Env'] if x.startswith(env.split('=',1)[0]+'=')]!=[env]:raise ValueError('container env')
    mounts=[]
    for i,x in enumerate(cmd[:-1]):
        if x=='--mount':
            parts=cmd[i+1].split(',');attrs=dict(z.split('=',1) for z in parts if '=' in z)
            if attrs.get('type')!='bind':raise ValueError('mount type')
            mounts.append((str(Path(attrs['src']).resolve()),attrs['dst'],'readonly' not in parts))
    actual=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in obj['Mounts'] if m.get('Type')=='bind']
    if len(actual)!=len(obj['Mounts']) or sorted(actual)!=sorted(mounts):raise ValueError('mounts')
    expected={(x['task'],str(x['qid'])):x for block in design['blocks'] for x in block};warm=design['processes'][0]['warmup'];expected[(warm['task'],str(warm['qid']))]=warm
    checked=worker.get('processor_inputs',[])
    if (worker.get('status')!='passed' or worker.get('inputs_checked')!=193 or len(expected)!=193 or len(checked)!=193
        or worker.get('cuda_initialized') is not False or worker.get('model_loaded') is not False
        or worker.get('GPU_authorized') is not False or worker.get('GT_opened') is not False or worker.get('scoring_performed') is not False):raise ValueError('worker result')
    seen=set()
    for item in checked:
        key=(item['task'],str(item['qid']));ref=expected[key]
        if key in seen or any(item.get(k)!=ref.get(k) for k in ('group_id','image_sha256','processor','image_start')):raise ValueError('processor reference')
        seen.add(key)
    if seen!=set(expected):raise ValueError('processor denominator')
    weights=worker['weights'];manifest=read(ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json')['files'];by={x['path']:x for x in manifest}
    if len(weights)!=4 or len({w['path'] for w in weights})!=4 or set(by)!={w['path'] for w in weights}:raise ValueError('weight inventory')
    for w in weights:
        ref=by[w['path']];st=Path(w['path']).stat()
        if w['sha256']!=ref['expected_lfs_sha256'] or w['stat']!=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]:raise ValueError('weight hash/stat')
    files={'plan_sha256':D/'preflight_plan001.json','worker_preflight_sha256':D/'run_preflight001/worker_preflight.json',
           'launch_sha256':D/'run_preflight001/launch.json','identity_sha256':D/'run_preflight001/container_identity.json',
           'exit_sha256':D/'run_preflight001/exit.json','started_sha256':D/'monitor_preflight001/started.json',
           'terminal_sha256':D/'monitor_preflight001/terminal.json','reviewer_sha256':Path(__file__)}
    result={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,
            'existing_DEV_predictions_opened':True,'scoring_performed':False,'errors':[],**{k:sha(v) for k,v in files.items()},
            'input_count':193,'weights':4,'container_id':cid,'bind_mounts':len(mounts),
            'limitations':['actual processor tensors checked online; recorded hashes/metadata are reference values',
                           'four weight payload hashes performed by preflight; this review checks recorded hashes and actual endpoint stats',
                           'four prior t020 generated files opened only for SHA; accepted E1/older six-arm mixed records used for identity',
                           'host launcher extra smoke remains to be explicitly bound in GPU plan','no model, CUDA, or GPU run authorized']}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
