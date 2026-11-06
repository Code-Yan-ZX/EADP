import json,subprocess,unittest
from types import SimpleNamespace
from docker_guard import DockerGuard,LABEL
CID='a'*64;NONCE='b'*32;NAME='eadp-memory-test';CMD=['docker','create','--name',NAME,'--label',LABEL+'='+NONCE,'image']
class MockDocker:
 def __init__(self):self.exists=False;self.running=False;self.label=NONCE;self.after_create_timeout=False;self.calls=[]
 def __call__(self,args,**kw):
  self.calls.append(args)
  if args[1]=='inspect':
   if not self.exists:return SimpleNamespace(returncode=1,stderr='Error: No such container: '+NAME,stdout='')
   data=[{'Id':CID,'Name':'/'+NAME,'Config':{'Labels':{LABEL:self.label}},'State':{'Running':self.running,'Pid':123 if self.running else 0,'ExitCode':0}}]
   return SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(data))
  if args[1]=='create':
   self.exists=True
   if self.after_create_timeout:raise subprocess.TimeoutExpired(args,30)
   return SimpleNamespace(returncode=0,stderr='',stdout=CID+'\n')
  if args[1]=='start':self.running=True
  if args[1] in ('stop','kill'):self.running=False
  return SimpleNamespace(returncode=0,stderr='',stdout=CID+'\n')
class GuardTest(unittest.TestCase):
 def test_create_start_stop_only_own_id(self):
  m=MockDocker();g=DockerGuard(NAME,NONCE,CMD,run=m);self.assertEqual(g.create_container(),CID);g.start_container(CID);self.assertTrue(g.inspect(CID)['running']);g.stop_container(CID);self.assertFalse(g.inspect(CID)['running'])
  with self.assertRaises(ValueError):g.stop_container('c'*64)
 def test_timeout_after_create_recovers_labelled_container(self):
  m=MockDocker();m.after_create_timeout=True;g=DockerGuard(NAME,NONCE,CMD,run=m)
  with self.assertRaises(subprocess.TimeoutExpired):g.create_container()
  self.assertEqual(g.recover_owned_container(),CID)
 def test_wrong_label_cannot_be_recovered_or_stopped(self):
  m=MockDocker();m.exists=True;m.label='c'*32;g=DockerGuard(NAME,NONCE,CMD,run=m);g.create_submitted=True
  with self.assertRaisesRegex(ValueError,'ownership'):g.recover_owned_container()
  self.assertNotIn('stop',[x[1] for x in m.calls])
 def test_existing_name_and_unowned_label_rejected(self):
  with self.assertRaises(ValueError):DockerGuard(NAME,NONCE,CMD+['--rm'])
  m=MockDocker();m.exists=True;g=DockerGuard(NAME,NONCE,CMD,run=m)
  with self.assertRaisesRegex(ValueError,'already exists'):g.create_container()
  self.assertIsNone(g.recover_owned_container())
  self.assertNotIn('stop',[x[1] for x in m.calls])
if __name__=='__main__':unittest.main()
