import types,unittest
import t020_timing_boundaries as timer,worker_candidate
class T:
 def __init__(self):self._version=0
 def detach(self):return self
 def clone(self):return T()
 def equal(self,other):return True
 def cpu(self):return self
 def tolist(self):return [2]
class S:
 method='max';threshold=.2;anchored=True
 def _select_2d_loco_anchored(self,*args):return T()
class Tests(unittest.TestCase):
 def test_first_token_and_sync(self):
  n=iter((10,20,30));w=iter((100,110,120));calls=[]
  def run(b):b(types.SimpleNamespace(shape=[1,33]),None);b(types.SimpleNamespace(shape=[1,34]),None);return 'ok'
  result,r=timer.measure(run,lambda:calls.append(1),lambda:next(n),lambda:next(w));self.assertEqual((result,r['first_token_selected_ns'],r['first_token_unix_ns'],r['criteria_calls']),('ok',20,110,2));self.assertEqual(len(calls),3)
 def test_hook_cleanup_both_paths_and_failure(self):
  old=S._select_2d_loco_anchored;base=types.SimpleNamespace(measure='old')
  def execute_pixel(*args,**kwargs):S()._select_2d_loco_anchored(None,1,1,None);return {'ok':True}
  base.execute_pixel=execute_pixel;ref={'selected_indices':[2]}
  for arm in ('author','wavefront'):
   r=worker_candidate.execute(base,S,lambda *args:T(),arm,None,None,None,{},lambda:None,ref,lambda v:None,lambda v:None);self.assertEqual(r['arm'],arm);self.assertEqual(r['selected_indices'],[2]);self.assertEqual(base.measure,'old');self.assertIs(S._select_2d_loco_anchored,old)
  with self.assertRaises(ValueError):worker_candidate.execute(base,S,lambda *args:T(),'author',None,None,None,{},lambda:None,{'selected_indices':[3]},lambda v:None,lambda v:None)
  self.assertEqual(base.measure,'old');self.assertIs(S._select_2d_loco_anchored,old)
if __name__=='__main__':unittest.main()
