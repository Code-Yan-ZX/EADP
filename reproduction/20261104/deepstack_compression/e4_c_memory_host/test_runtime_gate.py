"""Synthetic tests only: a checked allocation still does not execute GPU work."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import runtime_gate as gate


class RuntimeGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = {name: self.root / name for name in ('CPU', 'CPUMON', 'REVIEW', 'REVIEWER', 'MON', 'ALLOCATION')}
        self.paths['CPUMON'].mkdir()
        self.paths['MON'].mkdir()
        for name in ('started.json', 'terminal.json'):
            (self.paths['CPUMON'] / name).write_text('{}')
        self.paths['REVIEWER'].write_text('# synthetic reviewer\n')
        self.sources = {'/synthetic/source': 'a' * 64}
        self.cpu = {'status':'passed', 'gpu_used':False, 'gpu_authorized':False,
                    'source_sha256_before':self.sources, 'source_sha256_after':self.sources}
        self.write(self.paths['CPU'], self.cpu)
        self.review = {'status':'passed', 'independent_review':True,
                       'scope':'C-ON memory host prelaunch CPU review', 'gpu_authorized':False,
                       'GT_opened':False, 'scoring_performed':False, 'errors':[], 'source_sha256':self.sources,
                       'cpu_sha256':gate.rt.sha(self.paths['CPU']),
                       'cpu_started_sha256':gate.rt.sha(self.paths['CPUMON']/'started.json'),
                       'cpu_terminal_sha256':gate.rt.sha(self.paths['CPUMON']/'terminal.json'),
                       'worker_preflight_review_sha256':'b'*64,
                       'reviewer_sha256':gate.rt.sha(self.paths['REVIEWER'])}
        self.write(self.paths['REVIEW'], self.review)
        self.budget={'budget_start_unix':0, 'budget_end_unix':3000}
        self.write(self.root/'budget.json',self.budget)
        self.plan={'stage':'smoke','mode':'C-ON','jobs':24,'worker_timeout_seconds':1800,
                   'sampler_timeout_seconds':2100,'outer_timeout_seconds':2400,'grace_seconds':120,
                   'sample_interval_ns':500000000,'max_sample_gap_ns':5000000000,
                   'budget_source':'budget.json','budget_source_sha256':gate.rt.sha(self.root/'budget.json')}
        self.write(self.root/'host_plan.json',self.plan)
        self.allocation={'authorization':'root_scheduled_C_ON_memory_smoke','stage':'smoke','mode':'C-ON',
                         'gpu_uuid':'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee','budget_end_unix':3000,
                         'prelaunch_review_sha256':gate.rt.sha(self.paths['REVIEW']),'nonce':'c'*32}
        self.write(self.paths['ALLOCATION'],self.allocation)
        self.stack=[]
        for name,path in self.paths.items():self.stack.append(patch.object(gate,name,path))
        self.stack.extend([patch.object(gate,'HERE',self.root),patch.object(gate,'BASE',self.root),
            patch.object(gate.host_admission,'admit_cpu',return_value={'worker_preflight_review_sha256':'b'*64}),
            patch.object(gate.host_admission,'sources',return_value=self.sources),
            patch.object(gate.rt,'supervisor'),patch.object(gate.time,'time',return_value=1)])
        for p in self.stack:p.start()

    def tearDown(self):
        for p in reversed(self.stack):p.stop()
        self.tmp.cleanup()

    @staticmethod
    def write(path,obj):path.write_text(json.dumps(obj))

    def test_synthetic_review_and_allocation(self):
        self.assertEqual(gate.admit()['scope'],'C-ON isolated memory smoke')

    def test_missing_allocation_fails(self):
        self.paths['ALLOCATION'].unlink()
        with self.assertRaises(FileNotFoundError):gate.admit()

    def test_source_or_cpu_change_fails(self):
        self.cpu['source_sha256_after']={'/synthetic/source':'d'*64}
        self.write(self.paths['CPU'],self.cpu)
        with self.assertRaises(ValueError):gate.admit()

    def test_review_and_allocation_identity_fail(self):
        self.allocation['prelaunch_review_sha256']='e'*64
        self.write(self.paths['ALLOCATION'],self.allocation)
        with self.assertRaises(ValueError):gate.admit()
        self.allocation['prelaunch_review_sha256']=gate.rt.sha(self.paths['REVIEW'])
        self.allocation['authorization']='not_scheduled'
        self.write(self.paths['ALLOCATION'],self.allocation)
        with self.assertRaises(ValueError):gate.admit()

    def test_immutable_budget_and_deadline(self):
        self.write(self.root/'budget.json',{'budget_start_unix':0,'budget_end_unix':9999})
        with self.assertRaises(ValueError):gate.admit()
        self.write(self.root/'budget.json',self.budget)
        with patch.object(gate.time,'time',return_value=1000):
            with self.assertRaises(ValueError):gate.admit()


if __name__=='__main__':unittest.main()
