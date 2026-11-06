import importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('zip_natural_worker',HERE/'worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class TestPreflight(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name);self.old=w.BASE;w.BASE=self.base
  d=self.base/'e1/input_preparation_002';d.mkdir(parents=True)
  image=self.base/'test.img';image.write_bytes(b'image')
  self.row={'task':'DocVQA_VAL','qid':'1','message_sha256':'abc','group_id':'group','image_path':str(image),
            'message':[{'type':'image'},{'type':'text'}]}
  (d/'inputs.jsonl').write_text(json.dumps(self.row)+'\n')
  self.anchor={'task':'DocVQA_VAL','qid':'1','message_sha256':'abc','group_id':'group','image_sha256':w.sha(image)}
 def tearDown(self):w.BASE=self.old;self.tmp.cleanup()
 def test_accept(self):self.assertEqual(w.preflight({'sha256':{},'anchors':[self.anchor]})[0],self.row)
 def test_reject_wrong_image(self):
  self.anchor['image_sha256']='0'*64
  with self.assertRaises(ValueError):w.preflight({'sha256':{},'anchors':[self.anchor]})
 def test_reject_missing_question(self):
  self.anchor['qid']='2'
  with self.assertRaises(ValueError):w.preflight({'sha256':{},'anchors':[self.anchor]})
 def test_reject_group_mismatch(self):
  self.anchor['group_id']='wrong'
  with self.assertRaises(ValueError):w.preflight({'sha256':{},'anchors':[self.anchor]})
 def test_persist_before_decode_failure(self):
  class Seq:
   def detach(self):return self
   def cpu(self):return self
   def tolist(self):return [[10,11,12,13]]
  class Output:sequences=Seq()
  out=self.base/'generated.json'
  def fail(_):raise RuntimeError('decode failed')
  with self.assertRaises(RuntimeError):w.persist_then_decode(Output(),2,out,fail,[10,11])
  self.assertEqual(json.loads(out.read_text()),{'input_sequence_length':2,'sequences':[[10,11,12,13]]})
 def test_persist_before_shape_and_prefix_errors(self):
  class Seq:
   def __init__(self,raw):self.raw=raw
   def detach(self):return self
   def cpu(self):return self
   def tolist(self):return self.raw
  class Output:
   def __init__(self,raw):self.sequences=Seq(raw)
  for i,(raw,prefix) in enumerate([([[10]], [10,11]),([[99,11,12]], [10,11])]):
   out=self.base/('bad%d.json'%i)
   with self.assertRaises(ValueError):w.persist_then_decode(Output(raw),2,out,lambda x:x,prefix)
   self.assertEqual(json.loads(out.read_text())['sequences'],raw)
 def test_reject_source_drift(self):
  with self.assertRaises(ValueError):w.preflight({'sha256':{self.row['image_path']:'bad'},'anchors':[self.anchor]})
if __name__=='__main__':unittest.main(verbosity=2)
