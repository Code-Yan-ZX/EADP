"""Legacy continuation CPU checks: no model import or GPU launch."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('legacy_continuation',Path(__file__).with_name('schedule_legacy_streamwait_continuation.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class LegacyGates(unittest.TestCase):
    def gpu(self,entries=(),free=44000):return dict(entries=list(entries),free_mib=free)
    def entry(self,pid=1,memory=19000,command='owned worker',alive=True):return dict(pid=pid,memory_mib=memory,command=command,alive=alive)
    def job(self,exclusive=False,free=22500):return dict(exclusive_gpu=exclusive,min_free_memory_mib=free)
    def gate(self,job=None,gpu=None,active=set(),progress=None,ready=True,active_jobs=()):
        return s.launch_gate(job or self.job(),gpu or self.gpu(),active,progress or {},ready,active_jobs)

    def test_fixed_scope_and_honest_group_kinds(self):
        self.assertEqual(len(s.GROUP_SPECS),29);self.assertEqual(len(s.FIXED_IDS),32);self.assertEqual(len(set(s.FIXED_IDS)),32)
        kinds=[g['kind'] for g in s.GROUP_SPECS]
        self.assertEqual(kinds.count('pair'),3);self.assertEqual(kinds.count('FULL'),3);self.assertEqual(kinds.count('single_audit'),23)
        self.assertEqual(sum(s.COUNTS[g['task']]*len(g['job_ids']) for g in s.GROUP_SPECS),224199)
        self.assertFalse(any(g['model']=='next' and g['kind']!='single_audit' for g in s.GROUP_SPECS))
        self.assertFalse(any(key=='v15_pope_LRMAIN00625' for key in s.FIXED_IDS))

    def test_MME_then_POPE_MMB_VizWiz_GQA_barriers(self):
        priorities=[s.TASK_ORDER.index(g['task']) for g in s.GROUP_SPECS]
        self.assertEqual(priorities,sorted(priorities))
        self.assertLess(s.FIXED_IDS.index('v15_mme_LRMAIN00625'),s.FIXED_IDS.index('v15_gqa_EGATHER'))
        self.assertLess(s.FIXED_IDS.index('next_mme_LRMAIN00625'),s.FIXED_IDS.index('v15_pope_EGATHER'))
        self.assertLess(s.FIXED_IDS.index('next_mmben_LRMAIN00625'),s.FIXED_IDS.index('next_vizwiz_LRMAIN00625'))

    def test_unfinished_dependency_blocks(self):
        self.assertFalse(self.gate(ready=False)['ready'])

    def test_CPU_loading_reservation_counts_as_a_slot(self):
        self.assertFalse(self.gate(active={1},progress={1:True})['ready'])
        self.assertFalse(self.gate(gpu=self.gpu([self.entry()]),active={1},progress={1:False})['ready'])
        self.assertTrue(self.gate(gpu=self.gpu([self.entry()]),active={1},progress={1:True})['ready'])
        self.assertFalse(self.gate(active={1,2},progress={1:False,2:False})['ready'])

    def test_v15_FULL_nonexclusive_spec_can_pair(self):
        first=self.job(exclusive=False)
        self.assertTrue(self.gate(gpu=self.gpu([self.entry()]),active={1},progress={1:True},active_jobs=[first])['ready'])
        self.assertFalse(self.gate(gpu=self.gpu(free=22499))['ready'])
        self.assertTrue(self.gate(gpu=self.gpu(free=22500))['ready'])

    def test_exclusive_spec_preserved_for_future_NeXT_FULL(self):
        full=self.job(exclusive=True,free=32000)
        self.assertFalse(self.gate(job=full,gpu=self.gpu(free=31999))['ready'])
        self.assertTrue(self.gate(job=full,gpu=self.gpu(free=32000))['ready'])
        self.assertFalse(self.gate(job=full,gpu=self.gpu([self.entry()],40000),active={1},progress={1:True})['ready'])
        self.assertFalse(self.gate(gpu=self.gpu([self.entry()],40000),active={1},progress={1:True},active_jobs=[full])['ready'])

    def test_exact_zero_Ollama_only(self):
        ollama=self.entry(pid=9,memory=0,command='/usr/local/bin/ollama runner --model')
        self.assertTrue(self.gate(gpu=self.gpu([ollama]))['ready'])
        self.assertFalse(self.gate(gpu=self.gpu([dict(ollama,memory_mib=1)]))['ready'])
        self.assertFalse(self.gate(gpu=self.gpu([dict(ollama,command='unknown zero CUDA')]))['ready'])
        self.assertFalse(self.gate(gpu=self.gpu([dict(ollama,alive=False)]))['ready'])

    def test_unknown_third_context_refused(self):
        self.assertFalse(self.gate(gpu=self.gpu([self.entry(1),self.entry(2),self.entry(3)]),active={1,2},progress={1:True,2:True})['ready'])

    def test_same_beta2_pair_parameters_only_anchorzip_differs(self):
        group=next(g for g in s.GROUP_SPECS if g['kind']=='pair')
        common=dict(beta=2.,alpha=.5,visual_token_num=128,max_new_tokens=128,question_file='same')
        jobs={key:dict(model=group['model'],task=group['task'],n=s.COUNTS[group['task']],parameters=dict(common,anchorzip=bool(i))) for i,key in enumerate(group['job_ids'])}
        s.fairness(group,jobs);jobs[group['job_ids'][1]]['parameters']['max_new_tokens']=64
        with self.assertRaises(ValueError):s.fairness(group,jobs)
        jobs[group['job_ids'][1]]['parameters'].update(max_new_tokens=128,beta=1.)
        with self.assertRaises(ValueError):s.fairness(group,jobs)

    def dependency_state(self):
        ids=[str(i) for i in range(12)]
        state=dict(status='complete',registered_jobs=ids,supervisor_pid=100,child_pids=[],jobs={key:dict(status='complete',pid=200+i) for i,key in enumerate(ids)})
        marker=dict(success=True,complete=True,n_jobs=12,n_groups=7,registered_jobs=ids,plan_sha256='frozen')
        return ids,state,marker
    def call_dependency(self,state,marker,original=True,alive=()):
        with mock.patch.object(s,'read',return_value={'jobs':{key:{} for key in state['registered_jobs']}}),mock.patch.object(s,'sha',return_value='frozen'):
            return s.dependency_gate(state,marker,original,lambda pid:{'alive':pid in alive})

    def test_phase1_success_markers_and_original_exit_required(self):
        ids,state,marker=self.dependency_state()
        self.assertTrue(self.call_dependency(state,marker))
        self.assertFalse(self.call_dependency(state,marker,original=False))
        self.assertFalse(self.call_dependency(state,dict(marker,success=False)))
        self.assertFalse(self.call_dependency(state,dict(marker,plan_sha256='drift')))
        state['jobs'][ids[0]]['status']='generating';self.assertFalse(self.call_dependency(state,marker))

    def test_prior_supervisor_and_children_must_really_exit(self):
        ids,state,marker=self.dependency_state()
        self.assertFalse(self.call_dependency(state,marker,alive={100}))
        self.assertFalse(self.call_dependency(state,marker,alive={200}))
        state['child_pids']=[200];self.assertFalse(self.call_dependency(state,marker))

    def test_finished_hardgate_missing_complete_exit_or_identity_rejected(self):
        job=dict(control_protocol='fake',control_protocol_sha256='frozen',n=2374,job_id='v15_mme_EGATHER')
        finished=dict(success=True,complete=True,n=2374,generation_exit_code=0,job_id=job['job_id'])
        worker=SimpleNamespace(validate_finished=mock.Mock(return_value={'finished':finished}))
        with mock.patch.object(s,'verify'),mock.patch.object(s,'sha',return_value='frozen'),mock.patch.object(s,'read',return_value={}),mock.patch.object(s,'module',return_value=worker):
            plan={'worker_path':str(s.WORKER)}
            self.assertTrue(s.validate_job(job,plan)['finished']['success'])
            for field,value in [('complete',False),('n',2373),('generation_exit_code',1),('job_id','other')]:
                worker.validate_finished.return_value={'finished':dict(finished,**{field:value})}
                with self.assertRaises(ValueError):s.validate_job(job,plan)

    def test_preserves_existing_partial_log_and_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'partial.jsonl';path.write_text('partial')
            with self.assertRaises(FileExistsError):s.PRIOR.protected({'output_paths':{'prediction':str(path)}})
            self.assertEqual(path.read_text(),'partial')

    def test_frozen_source_drift_and_arbitrary_worker_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=Path(tmp)/'legacy_streamwait_worker.py';source=Path(tmp)/'input';worker.write_text('worker');source.write_text('input')
            with mock.patch.object(s,'WORKER',worker):
                plan=dict(worker_path=str(worker),worker_sha256=s.sha(worker),controller_sha256=s.sha(s.__file__),source_and_input_sha256={str(source):s.sha(source)})
                s.verify(plan);source.write_text('changed')
                with self.assertRaises(ValueError):s.verify(plan)
                with self.assertRaises(ValueError):s.worker_path(dict(worker_path=str(worker.with_name('other.py'))))

if __name__=='__main__':unittest.main()

