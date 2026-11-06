"""Independent admission review of the four-example post-hoc PixelPrune probe."""
import argparse, hashlib, json, subprocess
from pathlib import Path

B = Path(__file__).resolve().parent.parent
D = B / 'e6/pixelprune_t045_smoke'
M = B / 'e1_independent_review/monitor_pixel_t045_prelaunch'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def terminal(directory):
    s, t = read(directory/'started.json'), read(directory/'terminal.json')
    assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid']
    return s

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True); a=parser.parse_args()
    assert a.out.resolve()==D/'independent_prelaunch_review.json'
    plan=read(D/'plan001.json')
    assert sha(D/'plan001.json')=='3b483d26530f237b10fe1e812b9ab525f868afba11f6ad89f9d168171722329c'
    assert len(plan['sha256'])==134
    for path,digest in plan['sha256'].items(): assert sha(path)==digest,path
    assert plan['threshold']==0.45 and plan['method']=='pred_2d' and plan['metric']=='max' and plan['anchored'] is True and plan['vit_layer']==0
    assert plan['generation']=={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
    anchors=plan['anchors']
    assert [(x['task'],x['qid'],x['cpu_K']) for x in anchors]==[('DocVQA_VAL','16539',568),('TextVQA_VAL','39421',499),('AI2D_TEST','368100',500),('DocVQA_VAL','31078',1)]
    assert len({x['group_id'] for x in anchors})==4
    review=read(B/'e1_independent_review/pixel_threshold32_review.json')
    assert review['status']=='passed' and review['independent_review'] is True and sha(B/'e1_independent_review/pixel_threshold32_review.json')==plan['calibration_review_sha256']
    for x in anchors:
        events=[json.loads(line) for line in Path(x['cpu_journal']).read_text().splitlines()]
        match=[e for e in events if e.get('event')=='completed' and e.get('threshold')==0.45]
        assert len(match)==1 and match[0]['selected_indices']==x['cpu_selected_indices'] and match[0]['selected_count']==x['cpu_K']
        assert review['image_files_sha256'][x['cpu_journal']]==sha(x['cpu_journal'])
    for test in (D/'monitor_test001', B/'e1_independent_review/monitor_pixel_t045_cpu_review'):
        s=terminal(test); log=(test/'stderr.log').read_text()
        assert 'Ran 10 tests' in log and '\nOK' in log
        c=s['command']; cwd=Path(s['cwd'])
        direct=(cwd/c[-1]).resolve()==D/'test_worker.py'
        discovery=(c[-8:]==['-m','unittest','discover','-s','e6/pixelprune_t045_smoke','-p','test_worker.py','-v'] and cwd==B)
        assert direct or discovery
    s=terminal(D/'monitor_preflight001'); pre=read(D/'preflight001.json'); launch=read(D/'run_preflight001/launch.json'); ex=read(D/'run_preflight001/exit.json')
    assert pre['status']=='passed' and pre['source_plan_sha256']==sha(D/'plan001.json') and pre['cuda_initialized'] is False and pre['model_loaded'] is False and pre['GT_opened'] is False
    assert pre['weight_sha256_verified']==4 and pre['patch_before_model'] is True and len(pre['processor_verified'])==4
    assert [(x['task'],x['qid'],x['processor']) for x in pre['processor_verified']]==[(x['task'],x['qid'],x['processor']) for x in anchors]
    for x in pre['weight_stats']:
        st=Path(x['path']).stat(); assert (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino,st.st_dev)==(x['size'],x['mtime_ns'],x['ctime_ns'],x['inode'],x['device'])
    assert (Path(s['cwd'])/s['command'][-3]).resolve()==D/'launch.py' and s['command'][-2:]==['--stage','preflight']
    assert launch['plan_sha256']==sha(D/'plan001.json') and launch['controller']['pid']==s['worker_pid']==ex['controller']['pid']
    assert ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['failure_type'] is None and ex['timed_out'] is False and ex['client_exited_while_container_running'] is False
    obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,timeout=15,check=True).stdout)[0]
    st=obj['State'];host=obj['HostConfig'];cfg=obj['Config'];cmd=launch['command']
    assert obj['Id']==ex['last_inspect']['id'] and obj['Image']==ex['container_identity']['image'] and not st['Running'] and st['Pid']==0 and st['ExitCode']==0 and not st['OOMKilled']
    assert not host['Devices'] and not host['DeviceRequests'] and host['NetworkMode']=='none'
    assert cfg['Cmd']==[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['User']==cmd[cmd.index('--user')+1]
    mounts=[]
    for i,v in enumerate(cmd[:-1]):
        if v=='--mount':
            parts=cmd[i+1].split(','); d=dict(p.split('=',1) for p in parts if '=' in p); mounts.append((d['src'],d['dst'],'readonly' not in parts))
    assert sorted(mounts)==sorted((x['Source'],x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind') and len(obj['Mounts'])==len(mounts)
    started=read(M/'started.json')
    result={'status':'passed','independent_review':True,'plan_sha256':sha(D/'plan001.json'),'preflight_sha256':sha(D/'preflight001.json'),'preflight_started_sha256':sha(D/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(D/'run_preflight001/exit.json'),'reviewer_path':str(Path(__file__).resolve()),'reviewer_sha256':sha(__file__),'review_started_sha256':sha(M/'started.json'),'review_terminal_path':str(M/'terminal.json'),'GT_opened':False,'scoring_performed':False,'cpu_tests':10,'gpu_authorized':'subject to root scheduling','scope':'four deliberately chosen post-hoc DEV examples, CPU/GPU selection and model diagnostic only','limitations':['no GPU selection or model result yet','E1 DEV mixed prediction log parsed only for input identity','CPU K=1 is an expectation, GPU identity is not assumed','online feature/DS/position checks do not archive full tensors or KV contents','no accuracy, equal-compute or performance claim'],'errors':[]}
    with a.out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print('passed')

if __name__=='__main__': main()
