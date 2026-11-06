import copy,hashlib,importlib.util,sys,tempfile,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('identity_accept',Path(__file__).with_name('verify.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def meta(a,shape,dtype):return {'shape':shape,'dtype':dtype,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
class IdentityTest(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();m.D=Path(self.t.name);d=m.D/'run_001';d.mkdir();self.dir=d
  self.anchor={'task':'x','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m'}
  arrays={'prefill_embeddings':np.zeros((1,4,4),np.uint8),'prefill_mask':np.zeros((1,32),np.uint8),'prefill_positions':np.zeros((3,1,32),np.uint8),'visual_mask':np.zeros((1,4),np.uint8),'prefill_logits':np.zeros((1,8),np.uint8),**{f'injected_ds{i}':np.zeros((2,4),np.uint8) for i in range(3)}}
  desc={k:meta(v, list(v.shape[:-1])+[v.shape[-1]//(1 if k=='visual_mask' else 8 if k in ('prefill_mask','prefill_positions') else 2)],'torch.bool' if k=='visual_mask' else 'torch.int64' if k in ('prefill_mask','prefill_positions') else 'torch.bfloat16') for k,v in arrays.items()}
  kv=lambda n:{'shape':[1,8,n,128],'dtype':'torch.bfloat16','sha256':'a'*64}
  self.rec=dict(self.anchor,status='completed',path='author',selected_indices=[1,3],output_ids=[4,5],eos_ids=[5],generation_scalars={'x':1},processor={'p':1},original_sequence_length=8,original_max_position=7,prefill_mask=[[1]*4],prefill_position_ids=[[[0,1,2,3]]]*3,visual_mask=[[False,True,True,False]],main_visual=desc['injected_ds0'],injected_deepstack=[desc[f'injected_ds{i}'] for i in range(3)],prefill_embeddings=desc['prefill_embeddings'],prefill_logits=desc['prefill_logits'],prefill_tensors=desc,rope_delta=[[0]],decode_trace=[{'position_ids':[[[0,1,2,3]]]*3,'rope_delta':[[0]]},{'position_ids':[[[4]]]*3,'rope_delta':[[0]]}],cache_fingerprints=[[{'key':kv(4+i),'value':kv(4+i)} for _ in range(36)] for i in range(2)],GT_opened=False,scoring_performed=False)
  self.expected={'anchor':self.anchor,'expected':{'selected_indices':[1,3],'tokens':[4,5],'eos_ids':[5],'processor':{'p':1},'generation_kwargs':{'x':1}}};self.records=[]
  for i,arm in enumerate(('author','wavefront')):
   p=d/f'{i}.npz';np.savez(p,**arrays);r=copy.deepcopy(self.rec);r.update(path=arm,archive_path=str(p),archive_sha256=m.sha(p));self.records.append(r)
 def tearDown(self):self.t.cleanup()
 def test_positive_and_mismatch(self):
  m.validate_pair(*self.records,self.expected,{})
  self.records[1]['selected_indices']=[1,4]
  with self.assertRaises(ValueError):m.validate_pair(*self.records,self.expected,{})
 def test_docker_profile(self):
  job={'task':'x','qid':'1','arm':'author'};out=self.dir/'arm_000';worker=str(m.D/'worker_entry.py');args=[worker,'--task','x','--qid','1','--arm','author','--out',str(out/'observations')]
  cmd=['docker','run','--name','n','--user','1000:1000','--workdir','/w','--device','/dev/nvidia0:/dev/nvidia0','--env','K=V','--mount','type=bind,src=/src,dst=/dst,readonly','--entrypoint','/python','sha256:abc']+args
  doc={'Image':'sha256:abc','Name':'/n','Config':{'Entrypoint':['/python'],'Cmd':args,'User':'1000:1000','WorkingDir':'/w','Env':['K=V']},'Mounts':[{'Type':'bind','Destination':'/dst','Source':'/src','RW':False}],'HostConfig':{'Devices':[{'PathOnHost':'/dev/nvidia0'}]}}
  m.verify_docker(doc,cmd,job,out)
  doc['Mounts'][0]['RW']=True
  with self.assertRaises(ValueError):m.verify_docker(doc,cmd,job,out)
 def test_cache_and_array_mutation(self):
  self.records[0]['cache_fingerprints'][1][0]['key']['shape'][2]=6
  with self.assertRaises(ValueError):m.validate_pair(*self.records,self.expected,{})
  self.records[0]['cache_fingerprints'][1][0]['key']['shape'][2]=5
  with open(self.records[0]['archive_path'],'ab') as f:f.write(b'x')
  with self.assertRaises(ValueError):m.validate_pair(*self.records,self.expected,{})
if __name__=='__main__':unittest.main()
