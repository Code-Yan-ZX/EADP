"""Independent M3 no-GPU container preflight review; no model run."""
import argparse,hashlib,json,pathlib,subprocess
B=pathlib.Path(__file__).resolve().parents[1];D=B/'m3_gpu_smoke';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(pathlib.Path(p).read_text())
p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();assert not a.out.exists()
q=read(D/'independent_review_002_preflight.json');cpu=read(D/'cpu_check_006/verification.json');temp_start=read(B/'e1_independent_review/monitor_m3gpu002_preflight/started.json');temp_end=read(B/'e1_independent_review/monitor_m3gpu002_preflight/terminal.json')
assert q['status']=='passed' and q['independent_review'] and q['cpu_proof_sha256']==sha(D/'cpu_check_006/verification.json') and q['reviewer_sha256']==sha(B/'e1_independent_review/review_m3gpu002_preflight.py')
assert temp_end['reason']=='completed' and temp_end['worker_returncode']==0 and not temp_end['remaining_process_group'] and temp_start['worker_pid']==temp_end['worker_pid'] and all(sha(p)==h for p,h in cpu['source_sha256_after'].items())
f=D/'container_preflight_001.json';s=D/'monitor_preflight001/started.json';t=D/'monitor_preflight001/terminal.json';v=read(f);st=read(s);te=read(t);out=D/'preflight_001';launch=read(out/'launch.json');cmd=launch['command'];receipt=read(out/'host_upstream.json')
assert te['reason']=='completed' and te['worker_returncode']==0 and not te['remaining_process_group'] and te['worker_pid']==st['worker_pid'] and any(pathlib.Path(x).name=='preflight.py' for x in st['command'])
for k,path in [('worker_sha256',D/'worker.py'),('preflight_sha256',D/'preflight.py'),('launch_sha256',out/'launch.json'),('receipt_sha256',out/'host_upstream.json'),('stdout_sha256',out/'stdout.log'),('stderr_sha256',out/'stderr.log')]:assert v[k]==sha(path)
assert v['status']=='passed' and v['returncode']==0 and not v['timed_out'] and v['container_running'] is False and v['container_exit_code']==0
assert read(out/'stdout.log')==v['result'] and v['result']['cuda_initialized'] is False and v['result']['model_loaded'] is False and v['result']['GT_opened'] is False and v['result']['predictions'] is False and v['result']['target_ids']==[100,200,300]
assert receipt['sample_review_sha256']==q['sample_review_sha256'] and receipt['oracle_review_sha256']==q['oracle_review_sha256'] and receipt['native_review_sha256']==q['native_review_sha256']
assert launch['host_upstream']=={'path':str((out/'host_upstream.json').resolve()),'sha256':sha(out/'host_upstream.json')} and launch['approval_bundle']['review_sha256']==sha(D/'independent_review_002_preflight.json')
i=json.loads(subprocess.check_output(['docker','inspect',v['container_id']]))[0];state=i['State'];image='sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf'
assert not state['Running'] and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'] and i['Image']==image and i['Config']['Image']==image
assert not i['HostConfig']['Devices'] and not i['HostConfig']['DeviceRequests'] and 'CUDA_VISIBLE_DEVICES=' in i['Config']['Env'] and i['Config']['WorkingDir'].endswith('/author_eadp/Qwen_vl')
assert i['Config']['Entrypoint']==['/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'] and i['Config']['Cmd']==cmd[cmd.index(image)+1:] and i['Config']['Cmd'][-1]==str(out)
old=read(B/'e4_panel_v2/panel_003/launch.json')['command'];assert launch['upstream_launch_sha256']==sha(B/'e4_panel_v2/panel_003/launch.json')
def prefix(c):
 c=c[:];c[c.index('--name')+1]='<name>'
 while '--device' in c:
  j=c.index('--device');del c[j:j+2]
 c=['CUDA_VISIBLE_DEVICES=' if x=='CUDA_VISIBLE_DEVICES=0' else x for x in c];return c[:c.index(image)+1]
assert prefix(old)==prefix(cmd) and not (out/'events.jsonl').exists() and not (out/'result.json').exists()
q.update(scope='Final independent review of M3 synthetic-target single-image preflight; no GPU model result implied',reviewer_path=str(pathlib.Path(__file__).resolve()),reviewer_sha256=sha(__file__),preflight_sha256=sha(f),preflight_started_sha256=sha(s),preflight_terminal_sha256=sha(t),actual_container_id=v['container_id'],actual_container_state=state)
a.out.write_text(json.dumps(q,indent=2)+'\n');print('passed',sha(a.out))
