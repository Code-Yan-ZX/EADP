import hashlib,json,tempfile,unittest
from pathlib import Path
from reference_contract import ANCHORS,RECIPE,reference,sha

def put(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
class Tests(unittest.TestCase):
 def fixture(self,d):
  d=Path(d);obs=d/'observations';reviewer=d/'reviewer.py';reviewer.write_text('# fake reviewer\n');proof=d/'review.json';started=d/'started.json';terminal=d/'terminal.json';files={}
  for task,qid in (*ANCHORS,('DocVQA_VAL','31078')):
   o={'task':task,'qid':qid,'group_id':qid.zfill(64),'image_sha256':'a'*64,'message_sha256':'b'*64,'sequence_length':1026,'status':'completed','threshold':.2,'anchored':True,'cpu_gpu_selection_equal':True,'generation_kwargs':RECIPE,'eos_token_ids':[2],'selected_indices':[0,3],'cpu_expected_indices':[0,3],'selected_count':2,'processor_identity':{'processor':{'input_ids':{'shape':[1,1026],'dtype':'torch.int64','sha256':hashlib.sha256(bytes(1026*8)).hexdigest()}}}}
   for name,value in ((task+'_'+qid+'.json',o),(task+'_'+qid+'_generated.json',{'prompt_length':1026,'full_sequence_ids':[0]*1026+[2],'generated_token_ids':[2]})):
    path=put(obs/name,value);files[str(path.resolve())]=sha(path)
  for i in range(8):
   path=put(d/('extra%d.json'%i),{'x':i});files[str(path.resolve())]=sha(path)
  put(started,{'cwd':str(d),'command':['python',str(reviewer),'--out',str(proof)],'worker_pid':42});put(terminal,{'worker_pid':42,'reason':'completed','worker_returncode':0,'remaining_process_group':[]})
  put(proof,{'status':'passed','independent_review':True,'images':4,'GT_opened':False,'scoring_performed':False,'errors':[],'reviewer_sha256':sha(reviewer),'files_sha256':files})
  return proof,reviewer,started,terminal,obs
 def test_positive_and_source_tamper(self):
  with tempfile.TemporaryDirectory() as d:
   args=self.fixture(d);result=reference(*args);self.assertEqual(len(result['images']),3);self.assertFalse(result['gpu_authorized']);args[1].write_text('replacement')
   with self.assertRaises(ValueError):reference(*args)
 def test_relative_supervisor_output_and_wrong_path(self):
  with tempfile.TemporaryDirectory() as d:
   args=self.fixture(d);start=json.loads(args[2].read_text());start['command'][-1]='review.json';put(args[2],start)
   self.assertEqual(len(reference(*args)['images']),3)
   start['command'][-1]='wrong.json';put(args[2],start)
   with self.assertRaises(ValueError):reference(*args)
 def test_incomplete_supervisor_and_replaced_observation(self):
  with tempfile.TemporaryDirectory() as d:
   args=self.fixture(d);put(args[3],{'worker_pid':42,'reason':'completed','worker_returncode':0,'remaining_process_group':[2]})
   with self.assertRaises(ValueError):reference(*args)
  with tempfile.TemporaryDirectory() as d:
   args=self.fixture(d);(args[4]/'DocVQA_VAL_16539.json').write_text('{}')
   with self.assertRaises(ValueError):reference(*args)
if __name__=='__main__':unittest.main(verbosity=2)
