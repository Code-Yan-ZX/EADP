"""Independent four-image selector GPU preflight review; no GPU execution."""
import argparse, hashlib, json, subprocess
from pathlib import Path

B=Path(__file__).resolve().parent.parent
D=B/'e6/pixel_wavefront_gpu'
R=Path(__file__).resolve()
IMAGE='sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def monitor(name,script):
    d=D/name;s,t=read(d/'started.json'),read(d/'terminal.json');cwd=Path(s['cwd'])
    assert [(cwd/x).resolve() for x in s['command'] if Path(x).name==script]==[(D/script).resolve()]
    assert s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group']
    return s
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=read(D/'plan001.json');h=sha(D/'plan001.json')
    assert plan['threshold']==.45 and len(plan['anchors'])==4 and len(plan['source_sha256'])==65
    assert all(sha(k)==v for k,v in plan['source_sha256'].items())
    assert all(plan[k] is False for k in ('model_loaded','GT_opened','new_predictions_opened'))
    for name in ('worker.py','launch.py','gate.py','test_worker.py'):
        assert str((D/name).resolve()) in plan['source_sha256']
    cs=monitor('monitor_cpu_test001','test_worker.py');ps=monitor('monitor_preflight001','launch.py')
    log=(D/'monitor_cpu_test001/stderr.log').read_text()
    assert 'Ran 3 tests' in log and '\nOK\n' in log and 'test_boundaries_and_grids' in log
    pre=read(D/'preflight001.json');launch=read(D/'run_preflight001/launch.json');ex=read(D/'run_preflight001/exit.json')
    assert pre['status']=='passed' and pre['plan_sha256']==launch['plan_sha256']==h
    assert all(pre[k] is False for k in ('cuda_initialized','model_loaded','GT_opened','new_predictions_opened','gpu_used'))
    assert len(pre['cases'])==4 and all((c['task'],c['qid'])==(x['task'],x['qid']) and c['k']==len(x['expected_indices']) and c['capture_sha256']==sha(x['capture_path']) for c,x in zip(pre['cases'],plan['anchors']))
    assert ps['worker_pid']==launch['controller']['pid']==ex['controller']['pid']
    cmd=launch['command'];assert cmd[:2]==['docker','run'] and cmd[cmd.index('--network')+1]=='none' and '--device' not in cmd and '--gpus' not in cmd
    assert 'CUDA_VISIBLE_DEVICES=' in cmd and IMAGE in cmd and str(D/'worker.py') in cmd and '--preflight-only' in cmd
    assert ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and not ex['timed_out'] and ex['interrupted'] is None
    assert ex['failure_type'] is None and not ex['client_exited_while_container_running']
    cid=ex['container_identity']['container_id'];assert cid in ex['container_identity']['host_cgroup']
    obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15,check=True).stdout)[0]
    assert obj['Id']==cid and obj['Image']==IMAGE and not obj['State']['Running'] and obj['State']['Pid']==0 and obj['State']['ExitCode']==0
    assert not obj['State']['OOMKilled'] and not obj['HostConfig']['Devices'] and not obj['HostConfig']['DeviceRequests'] and obj['HostConfig']['NetworkMode']=='none'
    old=read(B/'e1_independent_review/pixel_t045_result_review.json');assert old['status']=='passed' and old['independent_review']
    observed={(x['task'],str(x['qid'])):x for x in old['observed']}
    assert all(observed[(x['task'],x['qid'])]['cpu_gpu_indices_equal'] is True and observed[(x['task'],x['qid'])]['visual_tokens']==len(x['expected_indices']) for x in plan['anchors'])
    paths={'preflight':D/'preflight001.json','preflight_started':D/'monitor_preflight001/started.json','preflight_terminal':D/'monitor_preflight001/terminal.json',
           'preflight_exit':D/'run_preflight001/exit.json','cpu_started':D/'monitor_cpu_test001/started.json','cpu_terminal':D/'monitor_cpu_test001/terminal.json','cpu_stderr':D/'monitor_cpu_test001/stderr.log'}
    result={'status':'passed','independent_review':True,'plan_sha256':h,'reviewer_sha256':sha(R),'GT_opened':False,'errors':[],
            'limitations':['No GPU work or model inference by this review.', 'Only four post-hoc images; per-arm timing is a single synchronized component measurement.',
                           'Allocator peak and device-wide memory are not measured.']}
    result.update({k+'_sha256':sha(v) for k,v in paths.items()})
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
