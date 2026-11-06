"""Independent review of four-image GPU selector records; no GPU/model work."""
import argparse, hashlib, json, subprocess
from pathlib import Path

B=Path(__file__).resolve().parent.parent
D=B/'e6/pixel_wavefront_gpu'; RUN=D/'run_smoke001'; REVIEWER=Path(__file__).resolve()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=read(D/'plan001.json');pre=read(D/'independent_prelaunch_review.json')
    assert pre['status']=='passed' and pre['plan_sha256']==sha(D/'plan001.json')
    assert all(sha(path)==h for path,h in plan['source_sha256'].items())
    s=read(D/'monitor_smoke001/started.json');t=read(D/'monitor_smoke001/terminal.json')
    assert s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group']
    assert s['timeout_seconds']==1920 and s['stale_seconds']==600 and s['grace_seconds']==90
    launch=read(RUN/'launch.json');end=read(RUN/'exit.json');assert launch['plan_sha256']==sha(D/'plan001.json') and launch['stage']=='smoke'
    assert end['returncode']==end['container_exit_code']==0 and end['container_stopped_verified'] and not end['timed_out'] and end['interrupted'] is None
    assert end['failure_type'] is None and not end['client_exited_while_container_running'] and end['gpu_pids_after']==[]
    cid=end['container_identity']['container_id'];assert cid in end['container_identity']['host_cgroup']
    obj=json.loads(subprocess.run(['docker','inspect',cid],text=True,capture_output=True,check=True,timeout=15).stdout)[0]
    assert obj['Id']==cid and not obj['State']['Running'] and obj['State']['Pid']==0 and obj['State']['ExitCode']==0 and not obj['State']['OOMKilled']
    data=read(RUN/'observations/complete.json');events=[json.loads(x) for x in (RUN/'observations/progress.jsonl').read_text().splitlines()]
    assert data['status']=='attempts_completed' and len(data['cases'])==len(plan['anchors'])==4
    cases=[]
    for i,(case,anchor) in enumerate(zip(data['cases'],plan['anchors'])):
        key=anchor['task']+'/'+anchor['qid'];assert case['key']==key and case['status']=='completed' and case['ordered_equal'] is True
        assert len(case['arms'])==2 and [x['arm'] for x in case['arms']]==(['author','wavefront'] if i%2==0 else ['wavefront','author'])
        logged=[x for x in events if x.get('event')=='arm_completed' and x.get('key')==key]
        assert len(logged)==2
        summary={}
        for arm,record in zip(case['arms'],logged):
            assert all(arm[k]==record[k] for k in ('arm','indices','matches_cpu','wall_ns_synchronized'))
            assert arm['indices']==anchor['expected_indices'] and arm['matches_cpu'] is True
            assert type(arm['wall_ns_synchronized']) is int and arm['wall_ns_synchronized']>0
            summary[arm['arm']]=arm['wall_ns_synchronized']
        cases.append({'key':key,'selected_count':len(anchor['expected_indices']),'ordered_equal':True,'component_wall_ns':summary})
    assert sum(x.get('event')=='case_started' for x in events)==4 and sum(x.get('event')=='case_recorded' for x in events)==4
    result={'status':'passed','independent_review':True,'scope':'four-image GPU anchored selector component','cases':cases,
            'GT_opened':False,'model_loaded':False,'scoring_performed':False,'container_id':cid,'reviewer_sha256':sha(REVIEWER),
            'plan_sha256':sha(D/'plan001.json'),'source_review_sha256':sha(D/'independent_prelaunch_review.json'),
            'worker_sha256':sha(D/'worker.py'),'progress_sha256':sha(RUN/'observations/progress.jsonl'),
            'complete_sha256':sha(RUN/'observations/complete.json'),'launch_sha256':sha(RUN/'launch.json'),'exit_sha256':sha(RUN/'exit.json'),
            'started_sha256':sha(D/'monitor_smoke001/started.json'),'terminal_sha256':sha(D/'monitor_smoke001/terminal.json'),
            'host_before_sha256':sha(D/'host_before001.txt'),'host_after_sha256':sha(D/'host_after001.txt'),
            'limitations':['One measured invocation per arm/image; four post-hoc images, no general speed inference.',
                           'Worker saved elapsed values, not raw start/end timestamps; elapsed values are not independently recomputed.',
                           'Includes selector and synchronization, excludes input preprocessing/transfer and model.',
                           'Host snapshots only before/after; background activity during measurement is unknown.',
                           'Does not establish identity or performance of full-model inference.']}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
