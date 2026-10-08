"""CPU reservation/exclusive/dependency/fairness tests; never launch a model."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('continuation',Path(__file__).with_name('schedule_streamwait_repair_continuation.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class ContinuationGates(unittest.TestCase):
    def job(self,full=False):return dict(exclusive_gpu=full,min_free_memory_mib=32000 if full else 22500)
    def gpu(self,contexts=None,free=44000):return dict(entries=contexts or [],free_mib=free)
    def context(self,pid=1,memory=21000,command='owned Next'):
        return dict(pid=pid,memory_mib=memory,command=command,alive=True)

    def test_dependency_not_done_blocks_all_generation(self):
        self.assertFalse(s.launch_gate(self.job(),self.gpu(),set(),{},False)['ready'])
        self.assertFalse(s.launch_gate(self.job(True),self.gpu(),set(),{},False)['ready'])

    def test_two_loading_reservations_are_two_slots(self):
        self.assertFalse(s.launch_gate(self.job(),self.gpu(),{1,2},{1:False,2:False},True)['ready'])

    def test_second_worker_waits_for_real_context_and_first_line(self):
        self.assertFalse(s.launch_gate(self.job(),self.gpu(),{1},{1:True},True)['ready'])
        self.assertFalse(s.launch_gate(self.job(),self.gpu([self.context()]),{1},{1:False},True)['ready'])
        self.assertTrue(s.launch_gate(self.job(),self.gpu([self.context()]),{1},{1:True},True)['ready'])

    def test_pruned_free_memory_gate(self):
        self.assertFalse(s.launch_gate(self.job(),self.gpu(free=22499),set(),{},True)['ready'])
        self.assertTrue(s.launch_gate(self.job(),self.gpu(free=22500),set(),{},True)['ready'])

    def test_FULL_requires_32000(self):
        self.assertFalse(s.launch_gate(self.job(True),self.gpu(free=31999),set(),{},True)['ready'])
        self.assertTrue(s.launch_gate(self.job(True),self.gpu(free=32000),set(),{},True)['ready'])

    def test_FULL_cannot_mix_even_with_enough_free_memory(self):
        self.assertFalse(s.launch_gate(self.job(True),self.gpu([self.context(memory=100)],40000),{1},{1:True},True)['ready'])
        self.assertFalse(s.launch_gate(self.job(True),self.gpu(),{1},{1:False},True)['ready'])

    def test_only_verified_zero_ollama_is_exempt(self):
        entry=self.context(4,0,'/usr/local/bin/ollama runner --model foreign')
        self.assertTrue(s.launch_gate(self.job(True),self.gpu([entry]),set(),{},True)['ready'])
        entry['memory_mib']=1;self.assertFalse(s.launch_gate(self.job(),self.gpu([entry]),set(),{},True)['ready'])
        entry.update(memory_mib=0,command='unknown CUDA');self.assertFalse(s.launch_gate(self.job(),self.gpu([entry]),set(),{},True)['ready'])

    def test_existing_partial_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);prediction=directory/'prediction.jsonl';protocol=directory/'protocol.json'
            prediction.write_text('partial');protocol.write_text('prepared')
            with self.assertRaises(FileExistsError):s.protected({'output_paths':{'prediction':str(prediction),'control_protocol':str(protocol)}})
            self.assertEqual(prediction.read_text(),'partial')
            prediction.unlink();s.protected({'output_paths':{'control_protocol':str(protocol)}})

    def test_only_two_fixed_worker_names_accepted(self):
        with tempfile.TemporaryDirectory() as temporary,mock.patch.object(s,'OUT',Path(temporary)):
            directory=Path(temporary)
            for name in ('streamwait_repair_worker.py','legacy_streamwait_worker.py'):
                self.assertEqual(s.worker_path({'worker_path':str(directory/name)}),directory/name)
            with self.assertRaises(ValueError):s.worker_path({'worker_path':str(directory/'arbitrary.py')})
            with self.assertRaises(ValueError):s.worker_path({'worker_path':'/outside/legacy_streamwait_worker.py'})

    def test_frozen_worker_input_controller_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temporary,mock.patch.object(s,'OUT',Path(temporary)):
            directory=Path(temporary);worker=directory/'streamwait_repair_worker.py';source=directory/'input.json'
            worker.write_text('worker');source.write_text('input')
            plan={'worker_path':str(worker),'worker_sha256':s.sha(worker),'controller_sha256':s.sha(s.__file__),
                  'source_and_input_sha256':{str(source):s.sha(source)}}
            s.verify(plan);source.write_text('changed')
            with self.assertRaises(ValueError):s.verify(plan)

    def test_fair_pair_same_budget_beta2_generation(self):
        group={'kind':'pair','task':'textvqa','budget':64,'job_ids':['E','AZ']}
        common={'beta':2.,'alpha':.5,'visual_token_num':64,'question_file':'same','max_new_tokens':128}
        jobs={'E':{'parameters':dict(common,anchorzip=False),'n':5000},'AZ':{'parameters':dict(common,anchorzip=True),'n':5000}}
        s.fairness(group,jobs);jobs['AZ']['parameters']['max_new_tokens']=64
        with self.assertRaises(ValueError):s.fairness(group,jobs)
        jobs['AZ']['parameters']['max_new_tokens']=128;jobs['AZ']['parameters']['beta']=1.
        with self.assertRaises(ValueError):s.fairness(group,jobs)

    def test_science_prediction_metadata_can_release_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            prediction=Path(temporary)/'sqa.jsonl';job={'output_paths':{'prediction':str(prediction)}}
            self.assertFalse(s.has_progress(job));prediction.write_text('')
            self.assertFalse(s.has_progress(job));prediction.write_text('{"text":"A","metadata":{"actual_visual_tokens_retained":160}}\n')
            self.assertTrue(s.has_progress(job))

    def test_fixed_12_jobs_7_groups_39119_questions_order(self):
        self.assertEqual(len(s.GROUP_SPECS),7)
        jobs=[key for _,_,_,_,keys in s.GROUP_SPECS for key in keys]
        self.assertEqual(len(jobs),12);self.assertEqual(len(set(jobs)),12)
        self.assertEqual(sum((5000 if task=='textvqa' else 2017)*len(keys) for _,task,_,_,keys in s.GROUP_SPECS),39119)
        self.assertEqual(jobs[:4],['next_text_E64','next_text_AZ64','next_text_E128','next_text_AZ128'])
        self.assertEqual([kind for _,_,_,kind,_ in s.GROUP_SPECS][-2:],['FULL','FULL'])

if __name__=='__main__':unittest.main()
