import unittest
import numpy as np
import processor_preflight as p
class CUDA:
 def is_initialized(self):return False
class Torch:
 cuda=CUDA()
 @staticmethod
 def all(x):return np.all(x)
class Wrapper:
 def _build_messages(self,message,dataset=None):return message
 def _processor_inputs(self,messages):return {'attention_mask':np.ones((1,2))}
class Config:image_token_id=7
class Tests(unittest.TestCase):
 def data(self):
  rows=[dict(task='t',qid=str(i),group_id='g'+str(i),image_sha256='i',message_sha256='m',message='ignore') for i in range(384)];selected=[{k:r[k] for k in ('task','qid','group_id','image_sha256','message_sha256')} for r in rows[:9]];refs={(r['task'],r['qid']):{'processor':{'digest':'ok'},'image_start':1} for r in selected};e4={k:{'processor':v['processor']} for k,v in refs.items()};return rows,selected,refs,e4
 def test_nine(self):
  rows,sel,refs,e4=self.data();calls=[];out=p.check(Torch(),Wrapper(),Config(),sel,rows,refs,e4,lambda *args:calls.append(1));self.assertEqual(len(out),9);self.assertEqual(len(calls),9)
 def test_mismatched_e4_and_processor_rejected(self):
  rows,sel,refs,e4=self.data();e4[('t','0')]['processor']={'digest':'bad'}
  with self.assertRaises(ValueError):p.check(Torch(),Wrapper(),Config(),sel,rows,refs,e4,lambda *args:None)
  rows,sel,refs,e4=self.data()
  def bad(*args):raise ValueError('tensor mismatch')
  with self.assertRaises(ValueError):p.check(Torch(),Wrapper(),Config(),sel,rows,refs,e4,bad)
if __name__=='__main__':unittest.main()
