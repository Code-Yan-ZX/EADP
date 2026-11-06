import hashlib,struct,unittest
import contract
class ContractTest(unittest.TestCase):
 def test_full_output_prompt_and_selection_are_bound(self):
  job={'arm':'t0_author'}; pb=lambda x:struct.pack('<%dq'%len(x),*x); prompt=[7,8];ref={'processor':{'input_ids':{'dtype':'torch.int64','shape':[1,2],'sha256':hashlib.sha256(pb(prompt)).hexdigest()}},'selected_indices':[1,5],'tokens':[9],'eos_ids':[9],'generation_kwargs':{'x':1}}
  identity={'status':'passed','job':job,'processor':ref['processor'],'selected_indices':[1,5],'output_ids':[9],'eos_ids':[9],'generation_kwargs':{'x':1},'GT_opened':False,'scoring_performed':False};raw={'job':job,'full_sequence_ids':[7,8,9],'accepted_input_length':2}
  self.assertEqual(contract.verify_identity(job,identity,raw,ref,pb)['selected_count'],2)
  for broken in ([8,7,9],[7,8,10]):
   with self.assertRaises(ValueError):contract.verify_identity(job,identity,dict(raw,full_sequence_ids=broken),ref,pb)
  with self.assertRaises(ValueError):contract.verify_identity(job,dict(identity,selected_indices=[5,1]),raw,ref,pb)
 def test_stage_order_and_duplicates(self):
  names=['barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end'];e=[{'event':x,'monotonic_ns':i+1} for i,x in enumerate(names)]
  self.assertEqual(contract.stages(e)[1]['release'],1)
  with self.assertRaises(ValueError):contract.stages(e+[e[0]])
  with self.assertRaises(ValueError):contract.stages(e[:-1])
 def test_nine_jobs_missing_or_reordered_fail(self):
  jobs=[{'order':i} for i in range(9)];journal=[x for j in jobs for x in ({'event':'started','job':j},{'event':'completed','job':j})];final={'status':'passed','jobs':9}
  contract.check_journal(journal,jobs,final)
  with self.assertRaises(ValueError):contract.check_journal(journal[:-1],jobs,final)
  journal[3]['job']=jobs[7]
  with self.assertRaises(ValueError):contract.check_journal(journal,jobs,final)
 def test_precuda_and_identity(self):
  x={'container_id':'abc','nonce':'n','mode':'t0_author','host_pid':3,'start_ticks':4,'boot_id':'b'};ready=dict(x,cuda_initialized=False,namespace_pid=1);release=dict(x,status='release',first_sample_ns=10)
  contract.check_handshake({'arm':'t0_author'},x,ready,release,x,{'query_end_ns':9})
  with self.assertRaises(ValueError):contract.check_handshake({'arm':'t0_author'},x,dict(ready,cuda_initialized=True),release,x,{'query_end_ns':9})
  with self.assertRaises(ValueError):contract.check_handshake({'arm':'t0_author'},x,ready,release,x,{'query_end_ns':11})
if __name__=='__main__':unittest.main()
