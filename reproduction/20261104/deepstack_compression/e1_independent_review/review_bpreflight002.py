import datetime, hashlib, json, pathlib, subprocess
B=pathlib.Path(__file__).resolve().parents[1]; D=B/'e4_b_identity'
def load(p): return json.loads(pathlib.Path(p).read_text())
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
r=load(D/'independent_review_009_preflight.json'); p=load(D/'b_container_preflight_002.json'); st=load(D/'monitor_bpreflight002/started.json'); te=load(D/'monitor_bpreflight002/terminal.json'); launch=load(D/'preflight_002/launch.json'); receipt=load(D/'preflight_002/host_panel_gate.json'); cmd=launch['command']; res=p['result']
assert r['status']=='passed' and r['independent_review'] is True
assert all(sha(k)==v for k,v in r['source_sha256'].items())
assert p['status']=='passed' and p['returncode']==0 and p['timed_out'] is False and p['container_running'] is False and p['container_exit_code']==0
assert p['worker_sha256']==sha(D/'b_worker.py') and p['host_script_sha256']==sha(D/'b_preflight.py')
for field,file in [('launch_sha256','launch.json'),('receipt_sha256','host_panel_gate.json'),('stdout_sha256','stdout.log'),('stderr_sha256','stderr.log')]: assert p[field]==sha(D/'preflight_002'/file)
assert receipt['host_live_docker_inspection'] is True and receipt['panel_proof_sha256']==r['panel_proof_sha256'] and receipt['panel_review_sha256']==r['panel_review_sha256']
assert res['status']=='preflight_passed' and not res['model_loaded'] and not res['predictions'] and not res['cuda_initialized']
assert load(D/'preflight_002/stdout.log')==res
assert st['command'][-1]==str(D/'b_preflight.py') and te['reason']=='completed' and te['worker_returncode']==0 and not te['remaining_process_group'] and st['worker_pid']==te['worker_pid']
i=json.loads(subprocess.check_output(['docker','inspect',p['container_id']]))[0];s=i['State'];assert not s['Running'] and s['Pid']==0 and s['ExitCode']==0 and not s['OOMKilled'] and not s['Dead']
assert not i['HostConfig']['DeviceRequests'] and not i['HostConfig']['Devices']
assert i['Config']['Image']=='sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf' and i['Config']['WorkingDir'].endswith('/author_eadp/Qwen_vl')
assert i['Config']['Cmd']==cmd[cmd.index(i['Config']['Image'])+1:] and i['Config']['Cmd'][-1]=='--preflight-only'
assert 'CUDA_VISIBLE_DEVICES=' in i['Config']['Env']
old=load(B/'e4_panel_v2/panel_003/launch.json')['command']; assert launch['upstream_launch_sha256']==sha(B/'e4_panel_v2/panel_003/launch.json')
def normalize(a):
 a=a[:]; idx=a.index('--name');a[idx+1]='<name>'
 while '--device' in a:
  idx=a.index('--device');del a[idx:idx+2]
 a=[('CUDA_VISIBLE_DEVICES=' if x=='CUDA_VISIBLE_DEVICES=0' else x) for x in a]
 idx=a.index('sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf');return a[:idx+1]
assert normalize(old)==normalize(cmd)
for key,file in [('preflight_sha256','b_container_preflight_002.json'),('preflight_started_sha256','monitor_bpreflight002/started.json'),('preflight_terminal_sha256','monitor_bpreflight002/terminal.json')]:r[key]=sha(D/file)
r.update(scope='Final independent CPU/source and actual no-GPU model-free container preflight review; permits B identity smoke only under its separate budget and GPU schedule',preflight_container_inspection={'id':p['container_id'],'state':s,'image':i['Config']['Image'],'device_requests':i['HostConfig']['DeviceRequests'],'devices':i['HostConfig']['Devices']},reviewer_sha256=sha(__file__),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=D/'independent_review_009.json';assert not out.exists();out.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n');print('passed',out,sha(out),'sources',len(r['source_sha256']))
