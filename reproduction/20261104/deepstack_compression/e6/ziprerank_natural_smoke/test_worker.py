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
  self.row={'task':'DocVQA_VAL','qid':'1','message_sha256':'abc','image_path':str(image),
            'message':[{'type':'image'},{'type':'text'}]}
  (d/'inputs.jsonl').write_text(json.dumps(self.row)+'\n')
  self.anchor={'task':'DocVQA_VAL','qid':'1','message_sha256':'abc','image_sha256':w.sha(image)}
 def tearDown(self):w.BASE=self.old;self.tmp.cleanup()
 def test_accept(self):self.assertEqual(w.preflight({'sha256':{},'anchors':[self.anchor]})[0],self.row)
 def test_reject_wrong_image(self):
  self.anchor['image_sha256']='0'*64
  with self.assertRaises(ValueError):w.preflight({'sha256':{},'anchors':[self.anchor]})
 def test_reject_missing_question(self):
  self.anchor['qid']='2'
  with self.assertRaises(ValueError):w.preflight({'sha256':{},'anchors':[self.anchor]})
 def test_reject_source_drift(self):
  with self.assertRaises(ValueError):w.preflight({'sha256':{self.row['image_path']:'bad'},'anchors':[self.anchor]})
if __name__=='__main__':unittest.main(verbosity=2)
