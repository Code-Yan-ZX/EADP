"""Compare a synthetic GPU inspect against the actual retained no-device baseline."""
import copy,json,subprocess,unittest
from pathlib import Path
import docker_contract
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
class Tests(unittest.TestCase):
 def test_retained_baseline_and_independent_gpu_identity_mutations(self):
  old=json.loads((ROOT/'e6/rres_k256_dev384/run001/launch.json').read_text())['command'];cid=json.loads((ROOT/'e6/rres_eadp_identity3_v2/preflight001/container_identity.json').read_text())['container_id']
  p=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=30);baseline=json.loads(p.stdout)[0]
  actual=copy.deepcopy(baseline);actual['Id']='f'*64;actual['Config']['Env']=[('CUDA_VISIBLE_DEVICES=0' if x.startswith('CUDA_VISIBLE_DEVICES=') else x) for x in actual['Config']['Env']];actual['HostConfig']['Devices']=[{'PathOnHost':x,'PathInContainer':x,'CgroupPermissions':'rwm'} for x in sorted(docker_contract.DEVICES)]
  worker=HERE/'gpu_worker.py';out=HERE/'run001/rres/observations';actual['Config']['Cmd']=[str(worker),'--arm','rres','--out',str(out)];cmd=list(old);cmd[cmd.index('--name')+1]='synthetic-name';cmd=cmd[:cmd.index(baseline['Image'])+1]+actual['Config']['Cmd']
  launch={'command':cmd,'name':'synthetic-name','arm':'rres','controller':{'pid':123,'start_ticks':456,'boot_id':'b'}};identity={'container_id':'f'*64,'host_init':{'pid':456,'start_ticks':789},'host_cgroup':'/docker/'+'f'*64};ex={'controller':launch['controller'],'container_identity':identity}
  args={'actual':actual,'baseline':baseline,'old_command':old,'launch':launch,'identity':identity,'exit':ex,'arm':'rres','worker':worker,'out':out,'supervisor_pid':123}
  self.assertTrue(docker_contract.check(**args))
  for mutation in ('image','command','env','mount','device','controller','pid'):
   x=copy.deepcopy(args)
   if mutation=='image':x['actual']['Image']='sha256:bad'
   if mutation=='command':x['actual']['Config']['Cmd'][1]='--wrong'
   if mutation=='env':x['actual']['Config']['Env'].append('EVIL=1')
   if mutation=='mount':x['actual']['Mounts'][0]['Destination']='/tampered'
   if mutation=='device':x['actual']['HostConfig']['Devices'].pop()
   if mutation=='controller':x['exit']['controller']['pid']=999
   if mutation=='pid':x['identity']['host_init']['start_ticks']=0
   with self.assertRaises(ValueError,msg=mutation):docker_contract.check(**x)
