import copy,unittest
import gate
class Tests(unittest.TestCase):
 def fixture(self):
  p={'status':'frozen','images':32,'jobs':380,'warmups':60,'measured':320,'processes':12,'GT_opened':False,'budget_end_unix':9999999999,'source_sha256':{'code':'Hcode'},'prelaunch':{'proof':'preproof','monitor':'premon','reviewer':'prereviewer','analysis_plan':'analplan','statistics_plan':'statplan','cpu_proof':'cpu'},'worker_completion_paths':['W'+str(i) for i in range(12)],'observer_path':'WINDOWS','run_health_path':'WINDOWS','observer_health_path':'WINDOWS','automatic_health_windows_snapshot':'WINDOWS','automatic_health_events_snapshot':'WINDOWS','automatic_health_snapshot_metadata':'WINDOWS','required_result_paths':['WINDOWS']+['W'+str(i) for i in range(12)],'journal_paths':['TIMING_JOURNAL'+str(i) for i in range(12)],
   'future':{'formal':'formal','independent':'independent','run_monitor':'runmon','launcher':'launcher','formal_monitor':'formalmon','verifier':'verifier','independent_monitor':'indmon','reviewer':'reviewer','execution_plan':'execplan','observer_monitor':'obsmon','observer':'observer','snapshot_monitor':'snapmon','snapshot':'snapshot','source_manifest':'srcmanifest'}}
  common={'status':'passed','images':32,'jobs':380,'warmups':60,'measured':320,'processes':12,'GT_opened':False,'errors':[],'execution_plan_sha256':'Hexecplan'}
  f={**common,'verifier_sha256':'Hverifier','files_sha256':{'WINDOWS':'Hwindows',**{'W'+str(i):'Hw' for i in range(12)}},'journal_sha256':{'TIMING_JOURNAL'+str(i):'Hjournal' for i in range(12)},'source_manifest_sha256':'Hsrcmanifest'}
  i={**common,'independent_review':True,'formal_sha256':'Hformal','reviewer_sha256':'Hreviewer','files_sha256':{'WINDOWS':'Hwindows',**{'W'+str(i):'Hw' for i in range(12)}},'journal_sha256':{'TIMING_JOURNAL'+str(i):'Hjournal' for i in range(12)},'source_manifest_sha256':'Hsrcmanifest','run_started_sha256':'Hrunmon/started.json','run_terminal_sha256':'Hrunmon/terminal.json','formal_started_sha256':'Hformalmon/started.json','formal_terminal_sha256':'Hformalmon/terminal.json'}
  pre={'status':'passed','independent_review':True,'GT_opened':False,'timing_results_opened':False,'errors':[],'analysis_plan_sha256':'Hanalplan','statistics_plan_sha256':'Hstatplan','analysis_cpu_sha256':'Hcpu','source_sha256':{'code':'Hcode'},'reviewer_sha256':'Hprereviewer'}
  return p,{'formal':f,'independent':i,'preproof':pre}
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
    if str(path).startswith('TIMING_JOURNAL') or str(path)=='WINDOWS' or str(path).startswith('W'):raise AssertionError('opened timing')
    return d[str(path)]
   def sha(path):
    if str(path).startswith('TIMING_JOURNAL') or str(path)=='WINDOWS' or str(path).startswith('W'):raise AssertionError('hashed timing')
    return 'H'+str(path)
   with self.subTest(failure=failure),self.assertRaises((ValueError,KeyError)):gate.admit(p,read,sha,lambda *a:None)
   self.assertNotIn('TIMING_JOURNAL',touched)
 def test_only_post_admission_verifies_timing_bytes(self):
  p,d=self.fixture();receipt=gate.admit(p,lambda path:d[str(path)],lambda path:'H'+str(path),lambda *a:None)
  seen=[]
  def sha(path):seen.append(path);return 'Hwindows' if path=='WINDOWS' else ('Hw' if path.startswith('W') else 'Hjournal')
  self.assertTrue(gate.verify_files(receipt,sha));self.assertEqual(seen,['WINDOWS']+['W'+str(i) for i in range(12)]+['TIMING_JOURNAL'+str(i) for i in range(12)])
 def test_real_supervisor_not_exited_rejected(self):
  p,d=self.fixture()
  with self.assertRaisesRegex(ValueError,'exit'):
   gate.admit(p,lambda path:d[str(path)],lambda path:'H'+str(path),lambda *args:(_ for _ in ()).throw(ValueError('exit')))
