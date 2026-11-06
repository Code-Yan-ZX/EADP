import copy,unittest
import gate
class Tests(unittest.TestCase):
 def fixture(self):
  p={'status':'frozen','images':32,'jobs':380,'warmups':60,'measured':320,'processes':12,'GT_opened':False,'budget_end_unix':9999999999,'source_sha256':{'code':'Hcode'},'required_result_paths':['TIMING_JOURNAL'],
   'future':{'formal':'formal','independent':'independent','run_monitor':'runmon','launcher':'launcher','formal_monitor':'formalmon','verifier':'verifier','independent_monitor':'indmon','reviewer':'reviewer','execution_plan':'execplan'}}
  common={'status':'passed','images':32,'jobs':380,'warmups':60,'measured':320,'processes':12,'GT_opened':False,'errors':[],'execution_plan_sha256':'Hexecplan'}
  f={**common,'verifier_sha256':'Hverifier','files_sha256':{'TIMING_JOURNAL':'Hjournal'}}
  i={**common,'independent_review':True,'formal_sha256':'Hformal','reviewer_sha256':'Hreviewer','files_sha256':{'TIMING_JOURNAL':'Hjournal'},'run_started_sha256':'Hrunmon/started.json','run_terminal_sha256':'Hrunmon/terminal.json','formal_started_sha256':'Hformalmon/started.json','formal_terminal_sha256':'Hformalmon/terminal.json'}
  return p,{'formal':f,'independent':i}
 def test_no_journal_open_on_missing_or_failed_proof(self):
  for failure in ('missing','failed','wronghash','missing_file'):
   p,d=self.fixture()
   if failure=='missing':del d['independent']
   if failure=='failed':d['independent']['status']='failed'
   if failure=='wronghash':d['independent']['formal_sha256']='bad'
   if failure=='missing_file':d['formal']['files_sha256']={}
   touched=[]
   def read(path):
    touched.append(str(path))
    if str(path)=='TIMING_JOURNAL':raise AssertionError('opened timing')
    return d[str(path)]
   def sha(path):
    if str(path)=='TIMING_JOURNAL':raise AssertionError('hashed timing')
    return 'H'+str(path)
   with self.subTest(failure=failure),self.assertRaises((ValueError,KeyError)):gate.admit(p,read,sha,lambda *a:None)
   self.assertNotIn('TIMING_JOURNAL',touched)
 def test_only_post_admission_verifies_timing_bytes(self):
  p,d=self.fixture();receipt=gate.admit(p,lambda path:d[str(path)],lambda path:'H'+str(path),lambda *a:None)
  seen=[]
  def sha(path):seen.append(path);return 'Hjournal'
  self.assertTrue(gate.verify_files(receipt,sha));self.assertEqual(seen,['TIMING_JOURNAL'])
 def test_real_supervisor_not_exited_rejected(self):
  p,d=self.fixture()
  with self.assertRaisesRegex(ValueError,'exit'):
   gate.admit(p,lambda path:d[str(path)],lambda path:'H'+str(path),lambda *args:(_ for _ in ()).throw(ValueError('exit')))
