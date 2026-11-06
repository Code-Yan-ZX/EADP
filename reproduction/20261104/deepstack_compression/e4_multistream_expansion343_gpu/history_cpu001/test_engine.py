import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pilot_engine as p
class Cuda:
 def synchronize(self):pass
class T:cuda=Cuda();float64='float64'
class Tensor:
 def __init__(self,x):self.x=np.asarray(x,dtype=np.float64)
 def detach(self):return self
 def to(self,**kw):return self
 def cpu(self):return self
 def reshape(self,*a):return self
 def tolist(self):return self.x.tolist()
 def numpy(self):return self.x
class Norm:
 @staticmethod
 def norms(streams):return np.zeros((4,1024),dtype=np.float64)
 @staticmethod
 def features_from_norms(n,feature):return {'fixed':0.0}
class Tests(unittest.TestCase):
 def fixtures(self):
  x=Tensor(np.ones((1024,2)));v=x.tolist();proc={k:x for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')};meta=lambda t:{'hash':'fixed'};ref={'processor':{k:meta(x) for k in proc},'visual_sha256':meta(x),'deepstack_full_sha256':[meta(x)]*3,'ordered_ids':v,'spatial_ids':sorted(v),'prefill_embed_sha256':meta(x),'attention_mask_sha256':meta(x),'prefill_position_ids':v,'rope_delta':v,'injected_mask':v,'injected_deepstack':[meta(x)]*3};actual={'processor':proc,'visual':(x,[x,x,x]),'ordered':x,'prefill':{'inputs_embeds':x,'attention_mask':x,'position_ids':x,'visual_pos_masks':x,'deepstack_visual_embeds':[x,x,x]},'rope_delta':x};rows=[dict(task='T',qid=str(i),group_id='g'+str(i),image_sha256='i',message_sha256='m',split='DEV') for i in range(384)];selected=[{k:r[k] for k in ('task','qid','group_id','image_sha256','message_sha256')} for r in rows[:343]];refs={(r['task'],r['qid']):ref for r in selected};e4={(r['task'],r['qid']):{} for r in selected};return rows,selected,refs,e4,actual,meta
 def run_engine(self,out,callback,events):
  rows,selected,refs,e4,actual,meta=self.fixtures();return p.run(selected,rows,refs,e4,out=out,torch=T(),wrapper=None,native=None,author_generate=None,Adapter=None,Capture=None,clone=None,capture_one=callback(actual),preflight_one=lambda *args:None,validate_e1=lambda *args:{'status':'passed'},tensor_meta=meta,identity_check=lambda *args:None,norm_mod=Norm(),feature_mod=None,journal=events.append)
 def test_all_343(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'out';events=[];status=self.run_engine(out,lambda a:lambda *x:({},{},a,lambda:True),events);self.assertEqual(len(status),343);self.assertEqual(json.loads((out/'status.json').read_text())['images_passed'],343);self.assertEqual(len(list(out.glob('*_norms.npy'))),343)
 def test_failure_marks_all_remaining(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'out';calls=[0];events=[]
   def maker(a):
    def capture(*args):
     calls[0]+=1
     if calls[0]==3:raise ValueError('synthetic failure')
     return {},{},a,lambda:True
    return capture
   with self.assertRaises(ValueError):self.run_engine(out,maker,events)
   s=json.loads((out/'status.json').read_text());self.assertEqual(s['images_passed'],2);self.assertEqual(len(s['rows']),343);self.assertEqual(sum(x['status']=='not_evaluated' for x in s['rows']),340)
if __name__=='__main__':unittest.main()
