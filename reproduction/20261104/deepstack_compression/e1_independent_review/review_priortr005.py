"""Independent PriorTR frozen CPU and preserved-container preflight review."""
import argparse,hashlib,json,pathlib,subprocess,sys,time
P=pathlib.Path(__file__).resolve().parents[1]/'e6/priortr/preparation'; I=P.parent/'install001'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(pathlib.Path(p).read_text())
a=argparse.ArgumentParser();a.add_argument('--out',type=pathlib.Path,required=True);args=a.parse_args();assert not args.out.exists()
cpu=load(P/'cpu_validation007.json');cs=load(P/'monitor_cpu_validation007/started.json');ct=load(P/'monitor_cpu_validation007/terminal.json');previous=load(P/'independent_review004.json')
assert cpu['status']=='passed' and cpu['returncode']==0 and cpu['source_sha256_before']==cpu['source_sha256_after']
assert all(sha(P/name)==digest for name,digest in cpu['source_sha256_after'].items())
assert ct['reason']=='completed' and ct['worker_returncode']==0 and not ct['remaining_process_group'] and cs['worker_pid']==ct['worker_pid']
assert sha(P/'cpu_validation007.stdout')==cpu['stdout_sha256'] and sha(P/'cpu_validation007.stderr')==cpu['stderr_sha256']
assert sha(P/'smoke_single_plan002.json')==cpu['plan_sha256'] and sha(P/'processor_check002.json')==cpu['processor_proof_sha256']
assert previous['status']=='passed' and previous['env_fingerprint_sha256']==sha(I/'env_fingerprint.json')==cpu['env_fingerprint_sha256'] and sha(I/'provenance.json')==cpu['environment_provenance_sha256']
files={'launch':P/'container_preflight003_launch.json','identity':P/'container_preflight003_identity.json','exit':P/'container_preflight003_exit.json','stdout':P/'container_preflight003.stdout','started':P/'monitor_container_preflight003/started.json','terminal':P/'monitor_container_preflight003/terminal.json','script':P/'container_preflight003.py','host':P/'container_preflight003_host.py'}
launch,identity,exited,output,st,te=[load(files[k]) for k in ('launch','identity','exit','stdout','started','terminal')]; c=launch['command'];s=identity['initial'];f=exited['final_inspect']
assert exited['status']=='passed' and exited['returncode']==0 and exited['failure'] is None and exited['identity']==identity and s['id']==f['id'] and s['image_id']==f['image_id']==c[-2] and c[-1]==str(files['script'])
assert s['state']['Running'] and s['state']['Pid']==identity['host_init']['pid']>0 and not f['state']['Running'] and f['state']['Pid']==0 and f['state']['ExitCode']==0
assert te['reason']=='completed' and te['worker_returncode']==0 and not te['remaining_process_group'] and te['worker_pid']==st['worker_pid']
assert output['status']=='passed' and not output['cuda_initialized'] and output['username']=='eadp' and output['n_frozen_samples']==8 and output['launcher_sha256']==sha(P/'smoke_launch.py')
assert output['torch']=='2.10.0+cu128' and output['transformers']=='5.2.0.dev0' and output['fingerprint_sha256']==cpu['env_fingerprint_sha256']
assert {'USER=eadp','LOGNAME=eadp','CUDA_VISIBLE_DEVICES='}.issubset(c) and '--gpus' not in c and '--device' not in c and '--rm' not in c
actual=json.loads(subprocess.check_output(['docker','inspect',s['id']]))[0]; state=actual['State'];assert not state['Running'] and state['ExitCode']==0 and state['Pid']==0 and not state['OOMKilled'] and not state['Dead']
assert actual['Image']==c[-2] and actual['Config']['Image']==c[-2] and not actual['HostConfig']['Devices'] and not actual['HostConfig']['DeviceRequests']
assert actual['Config']['Entrypoint']==[str(I/'env/bin/python')] and actual['Config']['Cmd']==[str(files['script'])] and actual['Config']['WorkingDir'].endswith('/author_eadp/Qwen_vl')
assert 'USER=eadp' in actual['Config']['Env'] and 'LOGNAME=eadp' in actual['Config']['Env'] and 'CUDA_VISIBLE_DEVICES=' in actual['Config']['Env']
run=subprocess.run([str(I/'env/bin/python'),'-m','unittest','-v','test_trace','test_env_contract','test_launch'],cwd=P,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90);assert run.returncode==0 and 'Ran 5 tests' in run.stderr and '\nOK' in run.stderr
assert all(sha(P/name)==digest for name,digest in cpu['source_sha256_after'].items())
proof={'status':'passed','independent_review':True,'scope':'single DEV PriorTR CPU source/environment and retained no-GPU preflight audit; GPU results remain untested','cpu_proof_sha256':sha(P/'cpu_validation007.json'),'cpu_terminal_sha256':sha(P/'monitor_cpu_validation007/terminal.json'),'source_sha256':cpu['source_sha256_after'],'plan_sha256':cpu['plan_sha256'],'processor_sha256':cpu['processor_proof_sha256'],'env_fingerprint_sha256':cpu['env_fingerprint_sha256'],'environment_provenance_sha256':cpu['environment_provenance_sha256'],'reviewer_sha256':sha(__file__),'prior_review_sha256':sha(P/'independent_review004.json'),'actual_container_id':s['id'],'actual_container_state':state,'checks':['CPU005-to-007 source changes and five tests independently rerun','preflight003 launch, initial/final identity, controller, stdout and supervisor','live docker inspect image, entrypoint, arguments, stopped/exit0 and no GPU devices','runtime inventory fingerprint tied to prior independent full rehash','same frozen single DEV sample, 7 diagnostic cases and stock/patched-disabled distinct'], 'limitations':['No model loaded or GPU inference in preflight','SDPA / torch and transformers differ from E1 FA2; no speed equivalence claim','KV content identity not established by CPU gate'], 'GT_opened':False,'scoring_performed':False,'review_unix':time.time()}
proof.update({'container_preflight_'+k+'_sha256':sha(v) for k,v in files.items()});args.out.write_text(json.dumps(proof,indent=2)+'\n');print('passed',args.out,sha(args.out))
