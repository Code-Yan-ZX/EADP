import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import completion,metrics,paired_stats,score
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e1'))
from test_contract import sample_quartet

def panel_data():
    panel=[]; records=[]; markers=[]
    for i,task in enumerate(completion.TASKS):
        qid=str(i);gid='g'+qid
        panel.append({'task':task,'qid':qid,'group_id':gid,'split':'DEV'})
        for r in sample_quartet():
            r.update(task=task,qid=qid,group_id=gid,forced=False,output_ids=[7],answer='',eos=True,eos_ids=[7],truncated=False)
            records.append(r)
        markers.append({'task':task,'qid':qid,'group_id':gid,'forced':False})
    return panel,records,markers

class CompletionTests(unittest.TestCase):
    def test_clean_exit_and_failure_cases(self):
        terminal={'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
        old={'pid':123,'start_ticks':45,'boot_id':'old'}
        exit={'returncode':0,'timed_out':False,'interrupted':None,'container_stopped_verified':True,
              'client_exited_while_container_running':False,
              'gpu_pids_after':[],'container_identity':{'container_id':'id','host_init':old},'controller':old}
        completion.validate_terminal(terminal,exit)
        for mutate in ('interrupted','stopped','gpu','terminal','early_client','missing_identity'):
            a,b=copy.deepcopy(terminal),copy.deepcopy(exit)
            if mutate=='interrupted':b['interrupted']=15
            if mutate=='stopped':b['container_stopped_verified']=False
            if mutate=='gpu':b['gpu_pids_after']=[123]
            if mutate=='terminal':a['remaining_process_group']=[123]
            if mutate=='early_client':b['client_exited_while_container_running']=True
            if mutate=='missing_identity':b['container_identity']['host_init']={'boot_id':'old'}
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):completion.validate_terminal(a,b)

    def test_exit_must_belong_to_actual_full_launcher(self):
        ident={'pid':231234,'start_ticks':33,'boot_id':'old'}
        terminal={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':ident['pid']}
        exited={'returncode':0,'timed_out':False,'interrupted':None,'container_stopped_verified':True,
                'client_exited_while_container_running':False,
                'gpu_pids_after':[],'container_identity':{'container_id':'id','host_init':ident},
                'controller':ident,'name':'the-container'}
        launch={'controller':ident,'command':['docker','--name','the-container','--stage','full','--out','/tmp/e1run']}
        completion.validate_terminal(terminal,exited,launch,'/tmp/e1run')
        for mutate in ('pid','name','stage','out'):
            a,b,c=copy.deepcopy(terminal),copy.deepcopy(exited),copy.deepcopy(launch)
            if mutate=='pid':a['worker_pid']+=1
            if mutate=='name':b['name']='other'
            if mutate=='stage':c['command'][c['command'].index('--stage')+1]='smoke'
            if mutate=='out':c['command'][c['command'].index('--out')+1]='/tmp/other'
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):completion.validate_terminal(a,b,c,'/tmp/e1run')

    def test_full_worker_and_all_launch_snapshots_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);e1=base/'e1';e1.mkdir();snap=e1/'run_sources';snap.mkdir()
            names=('full_runner.py','contract.py','position_adapter.py','adapter.py','full_launch.py')
            for name in names:(snap/name).write_text('snapshot '+name)
            originals=[e1/'full_runner.py',e1/'contract.py',e1/'position_adapter.py',base/'e0_identity/adapter.py',e1/'full_launch.py']
            hashes={str(path):completion.sha(snap/name) for path,name in zip(originals,names)}
            frozen={field:completion.sha(snap/name) for field,name in
                    [('runner_sha256','full_runner.py'),('contract_sha256','contract.py'),
                     ('position_adapter_sha256','position_adapter.py'),('e0_adapter_sha256','adapter.py')]}
            launch={'source_sha256':hashes,'command':['docker',str(e1/'full_runner.py'),'--stage','full']}
            with patch.object(completion,'E1',e1),patch.object(completion,'HERE',base/'e1_analysis'):
                paths=completion.validate_sources(frozen,launch,e1/'run')
                self.assertEqual(paths['full_runner.py'],snap/'full_runner.py')
                bad=copy.deepcopy(launch);bad['command'][1]=str(e1/'runner.py')
                with self.assertRaisesRegex(ValueError,'full worker'):completion.validate_sources(frozen,bad,e1/'run')
                bad=copy.deepcopy(frozen);bad['runner_sha256']='smoke-runner-hash'
                with self.assertRaisesRegex(ValueError,'source mismatch'):completion.validate_sources(bad,launch,e1/'run')
                (snap/'full_runner.py').write_text('replaced after launch')
                with self.assertRaisesRegex(ValueError,'source mismatch'):completion.validate_sources(frozen,launch,e1/'run')

    def test_incomplete_arm_or_group_never_accepted(self):
        panel,records,markers=panel_data()
        self.assertEqual(len(completion.validate_records(panel,records,markers,expected_per_task=1)),12)
        with self.assertRaises(ValueError):completion.validate_records(panel,records[:-1],markers,expected_per_task=1)
        broken=copy.deepcopy(records);broken[0]['group_id']='unknown'
        with self.assertRaises(ValueError):completion.validate_records(panel,broken,markers,expected_per_task=1)
        broken=copy.deepcopy(markers);broken.append(broken[0])
        with self.assertRaises(ValueError):completion.validate_records(panel,records,broken,expected_per_task=1)
        broken=copy.deepcopy(markers);broken.append(dict(markers[0],group_id='forged'))
        with self.assertRaises(ValueError):completion.validate_records(panel,records,broken,expected_per_task=1)
        forced=copy.deepcopy(records[0]);forced['forced']=True
        with self.assertRaises(ValueError):completion.validate_records(panel,records+[forced],markers,expected_per_task=1)
        broken=copy.deepcopy(records);broken[0].update(eos=False,eos_ids=[8],truncated=False)
        with self.assertRaisesRegex(ValueError,'valid terminal'):completion.validate_records(panel,broken,markers,expected_per_task=1)

    def test_generated_journal_matches_exact_arm_outputs(self):
        _,records,_=panel_data();keys=('task','qid','group_id','arm','forced','output_ids','eos_ids')
        generated=[{k:r[k] for k in keys} for r in records]
        completion.validate_generated(records,generated)
        with self.assertRaises(ValueError):completion.validate_generated(records,generated[:-1])
        bad=copy.deepcopy(generated);bad[0]['output_ids']=[22]
        with self.assertRaises(ValueError):completion.validate_generated(records,bad)

    def test_independent_decoder_checks_raw_empty(self):
        class T:
            def decode(self,ids,**kw):return '' if ids==[7] else 'bad'
        panel,records,markers=panel_data()
        completion.validate_decoding(completion.validate_records(panel,records,markers,1),T())
        records[0]['answer']='changed'
        with self.assertRaises(ValueError):completion.validate_decoding(records,T())

    def test_invalid_acceptance_does_not_open_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p=d/'proof.json';p.write_text('{"status":"failed"}')
            with patch.object(score,'read_truths',side_effect=AssertionError('GT accessed')), \
                 patch.object(sys,'argv',['score.py','--acceptance',str(p),'--acceptance-started',str(d/'started'),
                                         '--acceptance-terminal',str(d/'terminal'),'--dataset-summary',str(d/'nonexistent'),'--out',str(d/'out')]), \
                 patch.object(score,'validate_acceptance_supervisor'):
                with self.assertRaises(ValueError):score.main()
            self.assertFalse((d/'out').exists())

    def test_complete_proof_inventory_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);files={}
            for name in completion.PROOF_FILES:
                path=d/name;path.write_text('{}')
                files[name]={'path':str(path),'sha256':completion.sha(path)}
            proof={'status':'passed','stage':'full','natural_questions':384,'natural_records':1536,
                   'tokenizer_decode_checked':1536,'GT_opened':False,'scoring_performed':False,
                   'verifier_sha256':completion.sha(Path(completion.__file__)),'files':files}
            path=d/'proof.json';path.write_text(json.dumps(proof))
            with patch.object(completion,'validate_terminal'),patch.object(completion,'read_jsonl',return_value=[]), \
                 patch.object(completion,'validate_records',return_value=[{}]*1536):
                accepted=score.recheck_acceptance(path)
                self.assertEqual(len(accepted[3]),1536)
                for missing in ('generated_arms','launcher_source'):
                    bad=copy.deepcopy(proof);del bad['files'][missing];path.write_text(json.dumps(bad))
                    with self.assertRaisesRegex(ValueError,'inventory'):score.recheck_acceptance(path)

    def test_acceptance_supervisor_binds_actual_proof_and_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);proof=d/'proof';started=d/'started';terminal=d/'terminal'
            info={'worker_pid':12345,'cwd':str(d),'command':['python',str(Path(completion.__file__)),'--out',str(proof)]}
            end={'worker_pid':12345,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
            started.write_text(json.dumps(info));terminal.write_text(json.dumps(end))
            score.validate_acceptance_supervisor(proof,started,terminal)
            for mutate in ('pid','exit','out'):
                a,b=copy.deepcopy(info),copy.deepcopy(end)
                if mutate=='pid':b['worker_pid']+=1
                if mutate=='exit':b['worker_returncode']=1
                if mutate=='out':a['command'][-1]=str(d/'other')
                started.write_text(json.dumps(a));terminal.write_text(json.dumps(b))
                with self.assertRaises(ValueError):score.validate_acceptance_supervisor(proof,started,terminal)

class TruthIsolationTests(unittest.TestCase):
    def test_non_dev_answers_are_not_decoded_or_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.tsv'
            path.write_bytes(b'index\tanswer\n17\tvalid\n18\t\xff\xfe\n')
            sources={'TextVQA_VAL':{'path':str(path),'sha256':completion.sha(path)}}
            self.assertEqual(score.read_truths(sources,{('TextVQA_VAL','17')}),{('TextVQA_VAL','17'):{'answers':['valid']}})
            path.write_bytes(b'answer\tindex\nvalid\t17\n\xff\xfe\t18\n')
            sources['TextVQA_VAL']['sha256']=completion.sha(path)
            self.assertEqual(score.read_truths(sources,{('TextVQA_VAL','17')}),{('TextVQA_VAL','17'):{'answers':['valid']}})
    def test_selected_malformed_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.tsv';path.write_bytes(b'index\tanswer\n17\ttoo\tmany\n')
            sources={'TextVQA_VAL':{'path':str(path),'sha256':completion.sha(path)}}
            with self.assertRaises(ValueError):score.read_truths(sources,{('TextVQA_VAL','17')})

class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.author=metrics.author_functions();cls.const=metrics._author_constants()
    def check_pair(self,task,pred,truth):
        r=metrics.score(task,pred,truth,self.author,self.const)
        self.assertAlmostEqual(r['official'],r['independent'],places=12)
        return r
    def test_doc_threshold_whitespace_and_empty(self):
        self.assertEqual(self.check_pair('DocVQA_VAL','  Foo  ',{'answers':['foo']})['official'],1)
        self.assertEqual(self.check_pair('DocVQA_VAL','zz',{'answers':['ab']})['official'],0)
        empty=self.check_pair('DocVQA_VAL','',{'answers':['none']})
        self.assertEqual(empty['normalized_prediction'],'nan')
        self.assertNotEqual(empty['official'],empty['literal_answer_diagnostic'])
    def test_text_consensus_normalization(self):
        answers=['two']*3+['three']*7
        self.assertAlmostEqual(self.check_pair('TextVQA_VAL','2',{'answers':answers})['official'],0.9)
        for s in ['The 1,200 cats!','x.y','1.2','cant','a-blue/green','', 'NA','three\n dogs']:
            r=self.check_pair('TextVQA_VAL',s,{'answers':[s]*10})
            self.assertTrue(0<=r['official']<=1)
    def test_ai2d_exact_matching_and_failure_fallback(self):
        truth={'answer':'B','choices':{'A':'cat','B':'dog','C':'bird'}}
        for pred in ['B','The answer is (B).','dog','A and B','', 'Failed to obtain answer via API: dog','Cannot determine the answer']:
            self.check_pair('AI2D_TEST',pred,truth)
        self.assertEqual(self.check_pair('AI2D_TEST','B',truth)['official'],1)

    @unittest.skipUnless(importlib.util.find_spec('xlsxwriter'),'xlsxwriter is part of the frozen author environment')
    def test_author_xlsx_roundtrip_preserves_denominator(self):
        rows=[{'task':t,'arm':a,'qid':str(i),'answer':v}
              for t in completion.TASKS for a in paired_stats.ARMS for i,v in enumerate(['','2','NA'])]
        converted=score.native_xlsx_predictions(rows)
        self.assertEqual(len(converted),len(rows))
        self.assertEqual(converted[('DocVQA_VAL','0','P1-D0')],'nan')
        self.assertEqual(converted[('DocVQA_VAL','2','P1-D0')],'nan')

class StatsTests(unittest.TestCase):
    def test_factorial_contrasts_and_cluster_bootstrap(self):
        rows=[{'group_id':g,'scores':dict(zip(paired_stats.ARMS,v)),
               'answers':dict(zip(paired_stats.ARMS,['a','b','a','c']))} for g,v in [
                 ('x',[.1,.3,.2,.7]),('x',[.2,.4,.3,.8]),('y',[.3,.5,.4,.9])]]
        s=paired_stats.analyze(rows,replicates=1000,seed=20261104)
        self.assertEqual(s['n_image_groups'],2)
        self.assertAlmostEqual(s['contrasts']['DS_at_P1']['difference_pp'],20)
        self.assertAlmostEqual(s['contrasts']['DS_at_P3']['difference_pp'],50)
        self.assertAlmostEqual(s['contrasts']['interaction_DS_by_position']['difference_pp'],30)
        self.assertEqual(s['contrasts']['DS_at_P1']['improved'],3)
        self.assertEqual(s['contrasts']['DS_at_P3']['answer_changed'],3)
        self.assertEqual(s,paired_stats.analyze(rows,replicates=1000,seed=20261104))

if __name__=='__main__':unittest.main()
