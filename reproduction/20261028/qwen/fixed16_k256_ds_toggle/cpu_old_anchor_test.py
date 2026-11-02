"""Real saved tensors/predictions only; no author model, GPU or scoring."""
import json,os,unittest
import numpy as np
import torch
from historical_anchor import Q,FORMULA,FULL16,AUDIT,validate
class Tests(unittest.TestCase):
 def setUp(self):
  assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
  self.item=json.loads((FORMULA/'run_plan.json').read_text())['panel'][0]
  row=[json.loads(s) for s in (AUDIT/'public_fa2_full_run02_DocVQA_VAL.per_question.jsonl').open()][self.item['position']]
  def tensor(x):
   value=torch.from_numpy(x.copy());return value.view(torch.bfloat16) if x.dtype==np.uint16 else value
  with np.load(FORMULA/'execution/question_00.npz',allow_pickle=False) as a,np.load(FULL16/'execution/00_eadp1024.npz',allow_pickle=False) as b:
   self.rec=dict(native_failure=None,response=row['raw_prediction'],selection_calls=1,selection_errors=[],
    selection_results=[dict(ordered_ids=tensor(a['native_ordered_ids']),quotas=torch.tensor([256]))],
    pruner_input=tensor(a['visual_features_BF16_bits']),pruner_output=tensor(a['native_pruned_BF16_bits']),
    processor_inputs={k:tensor(b['processor_'+k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},
    vision=(tensor(b['visual_features']),[tensor(b['visual_deepstack_'+str(j)]) for j in range(3)]))
 def test_real_archives_and_literal_prediction_match(self):
  result=validate(torch,0,self.item,self.rec);self.assertTrue(result['historical_literal_answer_equal']);self.assertTrue(result['old_generation_token_ids_unavailable'])
 def test_wrong_native_response_rejected(self):
  self.rec['response']+='x'
  with self.assertRaises(AssertionError):validate(torch,0,self.item,self.rec)
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 assert not torch.cuda.is_initialized();raise SystemExit(0 if result.wasSuccessful() else 1)
