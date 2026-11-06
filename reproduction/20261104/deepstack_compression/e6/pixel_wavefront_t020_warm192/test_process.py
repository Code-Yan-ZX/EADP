import tempfile,unittest
from pathlib import Path
import process
class ProcessTest(unittest.TestCase):
 def fixture(self,d):
  image=Path(d)/'image';image.write_bytes(b'fake');h=process.sha(image);rows={};refs={};items=[]
  for i in range(49):
   row={'task':'fake','qid':str(i),'group_id':'g'+str(i),'image_path':str(image),'image_sha256':h,'message_sha256':'m','message':[]};ref={'processor':{'i':i},'image_start':1};identity={k:row[k] for k in ('task','qid','group_id','image_path','image_sha256','message_sha256')}|ref;rows[('fake',str(i))]=row;refs[('fake',str(i))]=ref;items.append(identity)
  return {'order':0,'block':0,'arm':'author','warmup':items[0],'measured':items[1:]},rows,refs
 def test_49_actual_order_and_completion(self):
  with tempfile.TemporaryDirectory() as d:
   job,rows,refs=self.fixture(d);out=Path(d)/'out';out.mkdir()
   def execute(row,ref,arm,raw):raw([[1,9]]);return {'status':'completed','arm':arm,**{k:row[k] for k in ('task','qid','group_id')}}
   result=process.run(job,out,rows,refs,execute);self.assertEqual(result['measured'],48);self.assertEqual(len((out/'journal.jsonl').read_text().splitlines()),98);self.assertFalse((out/'worker_complete.json').exists())
 def test_failure_keeps_raw_and_no_complete(self):
  with tempfile.TemporaryDirectory() as d:
   job,rows,refs=self.fixture(d);out=Path(d)/'out';out.mkdir()
   def execute(row,ref,arm,raw):raw([[1,9]]);raise RuntimeError('failure after token')
   with self.assertRaises(RuntimeError):process.run(job,out,rows,refs,execute)
   self.assertTrue((out/'raw_000.json').exists());self.assertFalse((out/'worker_complete.json').exists());self.assertIn('"event":"failed"',(out/'journal.jsonl').read_text())
if __name__=='__main__':unittest.main()
