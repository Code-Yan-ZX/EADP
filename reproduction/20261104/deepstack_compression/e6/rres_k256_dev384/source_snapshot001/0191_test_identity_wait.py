import json,unittest
from identity_wait import wait
class Tests(unittest.TestCase):
 def test_transient_missing_and_partial_then_valid(self):
  calls=iter([FileNotFoundError(),json.JSONDecodeError('partial','{',0),{'id':'valid'}]);clock=iter([0,1,2])
  def read(_):
   v=next(calls)
   if isinstance(v,Exception):raise v
   return v
  self.assertEqual(wait(read,['x'],clock=lambda:next(clock),sleep=lambda _:None),({'id':'valid'},))
 def test_timeout_and_invalid_errors(self):
  clock=iter([0,61])
  with self.assertRaises(TimeoutError):wait(lambda _:(_ for _ in ()).throw(FileNotFoundError()),['x'],clock=lambda:next(clock),sleep=lambda _:None)
  with self.assertRaises(ValueError):wait(lambda _:(_ for _ in ()).throw(ValueError('bad identity')),['x'],clock=lambda:0,sleep=lambda _:None)
if __name__=='__main__':unittest.main()
