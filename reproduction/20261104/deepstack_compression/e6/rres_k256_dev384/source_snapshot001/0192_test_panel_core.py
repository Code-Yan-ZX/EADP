import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from design import TASKS
import panel_core
def inputs():
 anchors=[dict(task=t,qid=str(i),group_id=t+str(i),message_sha256='a'*64,image_sha256='b'*64,processor={k:{} for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},image_start=2) for t in TASKS for i in range(128)]
 rows=[dict(a,image_path='/synthetic') for a in anchors];refs=[dict(a,low_visual_tokens=256,low_processor={'synthetic':True}) for a in anchors];controls={(a['task'],a['qid']):{'generation_scalars':dict(panel_core.RECIPE),'eos_ids':[151645,151643]} for a in anchors}
 return rows,anchors,refs,controls
class Model:
 def __init__(self):self.model=SimpleNamespace(rope_deltas='stale');self.generation_config=SimpleNamespace(eos_token_id=[151645,151643])
 def generate(self,**kw):return [[1,2]]
class Arm:
 def __init__(self,fail_at=None):self.calls=0;self.fail_at=fail_at
 def run_arm(self,torch,wrapper,module,row,ref,raw,decode):
  self.calls+=1
  if wrapper.model.model.rope_deltas is not None:raise AssertionError('stale rope')
  wrapper.model.generate(**panel_core.RECIPE)
  panel_core.write(raw,{'full_sequence':[[1,2]],'input_length':1})
  if self.calls==self.fail_at:raise RuntimeError('synthetic failure after raw')
  return {**{k:row[k] for k in ('task','qid','group_id','image_sha256','message_sha256')},'visual_tokens':256,'processor':ref['low_processor'],'hooks_and_methods_restored':True}
class Tests(unittest.TestCase):
 def run_panel(self,d,arm):
  args=inputs();return panel_core.run(*args,None,SimpleNamespace(model=Model()),SimpleNamespace(QWEN3_FIXED_RESOLUTION=1024),arm,None,d,verify_image=lambda _:'b'*64)
 def test_384_attempts_complete(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.run_panel(d,Arm());self.assertEqual(len(r['recorded']),384);self.assertEqual(len(list(Path(d).glob('*_generated.json'))),384);self.assertFalse((Path(d)/'worker_failure.json').exists())
 def test_failure_preserves_raw_and_full_denominator(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError):self.run_panel(d,Arm(fail_at=3))
   f=json.loads((Path(d)/'worker_failure.json').read_text());self.assertEqual(f['expected'],384);self.assertEqual(len(f['started']),3);self.assertEqual(len(f['recorded']),2);self.assertEqual(len(list(Path(d).glob('*_generated.json'))),3)
 def test_controls_reject_recipe_before_generation(self):
  model=Model();call=panel_core.checked_generation(model,{'generation_scalars':dict(panel_core.RECIPE),'eos_ids':[151645,151643]})
  with self.assertRaises(ValueError):call(**dict(panel_core.RECIPE,do_sample=True))
  with self.assertRaises(ValueError):call(**dict(panel_core.RECIPE,eos_token_id=3))
  for bad in ({k:v for k,v in panel_core.RECIPE.items() if k!='top_k'},dict(panel_core.RECIPE,extra=0)):
   with self.assertRaises(ValueError):panel_core.checked_generation(model,{'generation_scalars':bad,'eos_ids':[151645,151643]})
if __name__=='__main__':unittest.main()
