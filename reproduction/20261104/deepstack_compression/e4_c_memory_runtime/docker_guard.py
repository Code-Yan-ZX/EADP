"""Bounded Docker ownership primitives, not an admission or launch CLI.

Container ownership is proved using an unguessable, predeclared run label and
exact name/ID. The caller must independently review the entire docker create
command and Docker environment and provide a real sampler/controller; this
module by itself does not authorize or start a GPU job.
"""
import json,re,subprocess
LABEL='eadp.memory.nonce'
class DockerGuard:
 def __init__(self,name,nonce,create_argv,run=subprocess.run):
  if not re.fullmatch(r'eadp-memory-[a-z0-9-]+',name) or not re.fullmatch(r'[0-9a-f]{32}',nonce):raise ValueError('invalid private container identity')
  if (not isinstance(create_argv,list) or create_argv[:2]!=['docker','create'] or '--rm' in create_argv or create_argv.count('--name')!=1 or create_argv[create_argv.index('--name')+1]!=name or create_argv.count('--label')!=1 or create_argv[create_argv.index('--label')+1]!=LABEL+'='+nonce):raise ValueError('unbound container creation command')
  self.name=name;self.nonce=nonce;self.create_argv=list(create_argv);self.run=run;self.owned=None;self.create_submitted=False
 def command(self,args,timeout=30):
  p=self.run(args,text=True,capture_output=True,timeout=timeout,check=False)
  if p.returncode!=0:raise RuntimeError('docker command failed: '+p.stderr.strip()[:500])
  return p.stdout.strip()
 def inspect_owned(self,target,allow_absent=False):
  if target not in (self.name,self.owned) or target is None:raise ValueError('only this controller container can be inspected')
  p=self.run(['docker','inspect','--type','container',target],text=True,capture_output=True,timeout=20,check=False)
  if p.returncode:
   if allow_absent and 'No such container:' in p.stderr:return None
   raise RuntimeError('docker inspect failed: '+p.stderr.strip()[:500])
  data=json.loads(p.stdout)
  if not isinstance(data,list) or len(data)!=1:raise ValueError('ambiguous docker inspect')
  item=data[0];cid=item.get('Id','');label=(item.get('Config') or {}).get('Labels') or {}
  if not re.fullmatch(r'[0-9a-f]{64}',cid) or item.get('Name')!='/'+self.name or label.get(LABEL)!=self.nonce or (self.owned is not None and cid!=self.owned):raise ValueError('container ownership mismatch')
  state=item.get('State') or {}
  if type(state.get('Running')) is not bool or type(state.get('Pid')) is not int or type(state.get('ExitCode')) is not int:raise ValueError('container state unavailable')
  return {'container_id':cid,'running':state['Running'],'pid':state['Pid'],'exit_code':state['ExitCode'],'raw':item}
 def recover_owned_container(self):
  if not self.create_submitted:return None
  value=self.inspect_owned(self.name,allow_absent=True)
  if value is None:return None
  self.owned=value['container_id'];return self.owned
 def create_container(self):
  if self.inspect_owned(self.name,allow_absent=True) is not None:raise ValueError('container name already exists')
  self.create_submitted=True
  cid=self.command(self.create_argv,timeout=30)
  if not re.fullmatch(r'[0-9a-f]{64}',cid):raise ValueError('invalid create container id')
  self.owned=cid
  inspected=self.inspect_owned(cid)
  if inspected['running'] or inspected['pid']!=0:raise ValueError('new container unexpectedly running')
  return cid
 def start_container(self,cid):
  self.inspect_owned(cid);self.command(['docker','start',cid],timeout=30)
 def inspect(self,cid):return self.inspect_owned(cid)
 def stop_container(self,cid):
  self.inspect_owned(cid);self.command(['docker','stop','--time','20',cid],timeout=30)
 def kill_container(self,cid):
  self.inspect_owned(cid);self.command(['docker','kill',cid],timeout=15)
