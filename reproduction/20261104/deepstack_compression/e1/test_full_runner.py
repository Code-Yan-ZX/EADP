"""CPU-only interruption/resume contracts for the full DEV journal."""
import copy
import unittest
import subprocess
import sys
from pathlib import Path
from full_runner import validate_prefix
from test_contract import sample_quartet


class FullResumeTests(unittest.TestCase):
    def setUp(self):
        self.records=sample_quartet()
        for r in self.records:
            r.update(forced=False,eos_ids=[2])
        r=self.records[0]
        self.panel=[{k:r[k] for k in ('task','qid','group_id','message_sha256')}]
        self.generated=[{k:r[k] for k in ('task','qid','group_id','arm','forced','output_ids','eos_ids')} for r in self.records]
        self.quartets=[{'task':r['task'],'qid':r['qid'],'group_id':r['group_id'],'forced':False}]

    def test_valid_complete_and_partial_prefix(self):
        self.assertEqual(validate_prefix(self.panel,self.records,self.generated,self.quartets),4)
        self.assertEqual(validate_prefix(self.panel,self.records[:2],self.generated[:2],[]),2)

    def test_damaged_or_unpaired_journal_is_never_replayed(self):
        with self.assertRaises(ValueError): validate_prefix(self.panel,self.records[:1],self.generated[:2],[])
        bad=copy.deepcopy(self.generated); bad[0]['output_ids']=[999]
        with self.assertRaises(ValueError): validate_prefix(self.panel,self.records,bad,self.quartets)
        with self.assertRaises(ValueError): validate_prefix(self.panel,self.records,self.generated,[])

    def test_wrong_sample_or_quartet_rejected(self):
        records=copy.deepcopy(self.records); records[0]['qid']='different'
        with self.assertRaises(ValueError): validate_prefix(self.panel,records,self.generated,self.quartets)

    def test_resume_entrypoints_fail_closed(self):
        here=Path(__file__).parent
        common=['--resume','--e0-proof','dummy','--manifest-dir','dummy','--inputs','dummy','--input-proof','dummy']
        for script,extra in [('full_runner.py',['--execute','--stage','full','--out','dummy']),
                             ('full_launch.py',['--run-id','test','--output-id','test','--timeout','60'])]:
            p=subprocess.run([sys.executable,str(here/script),*common,*extra],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('resume disabled',p.stderr)

    def test_full_launcher_exit_requires_container_identity_and_empty_gpu(self):
        import full_launch
        good={'timed_out':False,'interrupted':None,'returncode':0,'client_exited_while_container_running':False,
              'container_stopped_verified':True,'container_identity':{'container_id':'test'},'gpu_pids_after':[]}
        self.assertEqual(full_launch.exit_code(good),0)
        for field,value in [('container_identity',None),('gpu_pids_after',['123']),('gpu_pids_after',None),('container_stopped_verified',False)]:
            bad=dict(good); bad[field]=value
            self.assertNotEqual(full_launch.exit_code(bad),0)

if __name__=='__main__': unittest.main()
