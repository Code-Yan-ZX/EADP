import types,unittest
import observe
class Tests(unittest.TestCase):
 def test_bracket_and_failure_not_silenced(self):
  values=iter([10,20]);seen=[]
  def run(argv,**kwargs):seen.append(argv);return types.SimpleNamespace(returncode=0,stdout='x',stderr='')
  v=observe.query(observe.PS,run=run,clock=lambda:next(values));self.assertEqual((v['query_start_ns'],v['query_end_ns']),(10,20));self.assertEqual(seen,[observe.PS])
  values=iter([30,40]);v=observe.query(observe.DOCKER,run=lambda *a,**k:(_ for _ in ()).throw(TimeoutError()),clock=lambda:next(values));self.assertIsNone(v['returncode']);self.assertEqual(v['error_type'],'TimeoutError')
 def test_clock_regression_rejected(self):
  values=iter([20,10])
  with self.assertRaises(ValueError):observe.query(observe.PS,run=lambda *a,**k:types.SimpleNamespace(returncode=0,stdout='',stderr=''),clock=lambda:next(values))
if __name__=='__main__':unittest.main()
