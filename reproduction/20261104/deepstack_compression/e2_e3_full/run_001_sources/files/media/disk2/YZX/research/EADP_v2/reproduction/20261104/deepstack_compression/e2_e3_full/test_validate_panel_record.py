import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e2_e3_smoke'))
import test_verify_interventions as fixture
from validate_panel_record import check,digest
class ValidateRecords(unittest.TestCase):
 def test_all_e2_masks_and_corruption(self):
  maker=fixture.VerifyInterventions()
  for arm in ('000','100','010','001','110','101','011','111'):
   row,arrays=maker.make('E2',arm);row['arrays']={'kind':'full_arrays'};self.assertTrue(check(row,arrays))
   bad={k:v.copy() for k,v in arrays.items()};bad['injected0'][0,0]^=1
   with self.assertRaises(ValueError):check(row,bad)
 def test_summary_zeros_and_kept_streams(self):
  row,arrays=fixture.VerifyInterventions().make('E2','101')
  meta=lambda a,shape:{'dtype':'torch.float32','shape':shape,'sha256':digest(a)}
  summaries={'visual':row['main_visual'],'prefill':row['prefill_embeds']}
  for i in range(3):
   summaries['ds'+str(i)]=row['full_deepstack'][i];summaries['selected_before'+str(i)]=meta(arrays['selected_before'+str(i)],[256,1]);summaries['injected'+str(i)]=meta(arrays['injected'+str(i)],[256,1]);row['selected_deepstack'][i]=row['injected_deepstack'][i]=summaries['injected'+str(i)]
  row['arrays']={'kind':'summary_only','tensors':summaries};self.assertTrue(check(row))
  bad=copy.deepcopy(row);bad['arrays']['tensors']['injected1']['sha256']='a'*64;bad['injected_deepstack'][1]['sha256']='a'*64;bad['selected_deepstack'][1]['sha256']='a'*64
  with self.assertRaises(ValueError):check(bad)
 def test_permutation_and_position_fail_closed(self):
  row,arrays=fixture.VerifyInterventions().make('E3','PERM');row['arrays']={'kind':'full_arrays'};self.assertTrue(check(row,arrays))
  bad=copy.deepcopy(row);bad['decode_trace'][4]['layer_cache_lengths'][2]+=1
  with self.assertRaises(ValueError):check(bad,arrays)
  bad=copy.deepcopy(row);bad['intervention_metadata']['permutation'][0]=0
  with self.assertRaises(ValueError):check(bad,arrays)
if __name__=='__main__':unittest.main()
