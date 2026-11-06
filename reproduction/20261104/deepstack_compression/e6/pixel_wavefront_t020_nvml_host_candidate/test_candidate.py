import copy,json,unittest
from pathlib import Path
import host_contract as h
import stages as s

DESIGN=Path(__file__).resolve().parent.parent/'pixel_wavefront_t020_nvml_nine/design001.json'
M=s.MIB

def good_job_result(job):
    i=job['order']+1
    fields=('running_inspect_verified','host_argv_cgroup_verified','worker_ready_pre_cuda','first_sample_before_release','release_nonce_uuid_pid_bound','stopped_inspect_verified','sampler_completed','identity_verified','weights_unchanged','source_proof_verified','no_foreign_compute_observed','stage_windows_valid','sampling_coverage_valid')
    return {'job':job,'status':'passed','container_id':f'{i:064x}','nonce':f'{i:032x}','process':{'host_pid':100+i,'start_ticks':1000+i,'boot_id':'boot','container_pid':1},'exit_code':0,'final_container_pid':0,'sampler_remaining_process_group':[],**{k:True for k in fields}}

def sample(a,b,procs,used=3*M):
    return {'status':'completed','query_start_ns':a,'query_end_ns':b,'gpu_uuid':'GPU-1','device_memory_bytes':{'total':8*M,'used':used,'free':5*M,'reserved':None},'processes':procs}

class TestHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.jobs=json.loads(DESIGN.read_text())['jobs']
    def test_all_nine_in_frozen_order(self):
        events=[]; result=h.run_nine(self.jobs,good_job_result,events.append)
        self.assertEqual(result['unique_processes'],9);self.assertEqual([e['job']['order'] for e in events],list(range(9)))
    def test_bad_grid(self):
        bad=copy.deepcopy(self.jobs);bad[1]['arm']=bad[4]['arm']
        with self.assertRaises(ValueError): h.validate_design(bad)
    def test_duplicate_identity_aborts_no_retry(self):
        events=[];calls=[]
        def run(j):
            calls.append(j['order']);r=good_job_result(j)
            if j['order']==1:r['container_id']=f'{1:064x}'
            return r
        with self.assertRaises(ValueError): h.run_nine(self.jobs,run,events.append)
        self.assertEqual(calls,[0,1]);self.assertIn('controller_error',events[-1])
    def test_failed_result_is_recorded_before_abort(self):
        events=[];calls=[]
        def run(j):
            calls.append(j['order']);r=good_job_result(j);r['sampler_remaining_process_group']=[99];return r
        with self.assertRaises(ValueError): h.run_nine(self.jobs,run,events.append)
        self.assertEqual(calls,[0]);self.assertIn('result',events[0]);self.assertIn('controller_error',events[1])
    def test_budget(self):
        self.assertEqual(h.validate_budget(100,17000,1770,90,16020,9),16020)
        with self.assertRaises(ValueError):h.validate_budget(100,16000,1770,90,16020,9)
        with self.assertRaises(ValueError):h.validate_budget(100,17000,1770,90,16019,9)
    def test_unverified_boolean_not_accepted(self):
        r=good_job_result(self.jobs[0]);r['sampling_coverage_valid']=1
        with self.assertRaises(ValueError):h.validate_result(self.jobs[0],r,set(),set(),set())

class TestStages(unittest.TestCase):
    def setUp(self):
        self.bounds={'release':20,'model_load_start':30,'model_load_end':60,'request_start':70,'request_end':100,'teardown_end':120}
        self.samples=[sample(10,15,[{'pid':8,'type':'G','used_bytes':M}]),sample(30,35,[{'pid':7,'type':'C','used_bytes':2*M},{'pid':8,'type':'G','used_bytes':M}]),sample(50,55,[{'pid':7,'type':'C','used_bytes':None}]),sample(70,75,[{'pid':7,'type':'C','used_bytes':3*M}]),sample(90,95,[{'pid':7,'type':'C','used_bytes':4*M}]),sample(110,125,[])]
    def test_distinguishes_unknown_absence_graphics_and_stage(self):
        out=s.summarize(self.samples,7,'GPU-1',self.bounds,40)
        self.assertEqual(out['stages']['model_load']['overlap_metrics']['target'],{'known':1,'unknown':1,'max_known_bytes':2*M})
        self.assertEqual(out['stages']['pre_cuda']['target_absent'],1)
        self.assertEqual(out['stages']['model_load']['target_present_unknown'],1)
        self.assertEqual(out['stages']['model_load']['graphics_pids'],[8])
        self.assertEqual(out['stages']['request']['overlap_metrics']['target']['max_known_bytes'],4*M)
        self.assertEqual(out['max_unsampled_gap_ns'],15)
        self.assertEqual(out['max_possible_sample_spacing_ns'],35)
    def test_foreign_compute_rejected(self):
        bad=copy.deepcopy(self.samples);bad[2]['processes'].append({'pid':99,'type':'C+G','used_bytes':M})
        with self.assertRaises(ValueError):s.summarize(bad,7,'GPU-1',self.bounds,40)
    def test_failed_missing_unit_or_gap_rejected(self):
        for field in ('failed','unit','gap','unknown_type'):
            bad=copy.deepcopy(self.samples)
            if field=='failed':bad[1]['status']='failed'
            if field=='unit':bad[1]['device_memory_bytes']['used']=7
            if field=='gap':bad[2]['query_start_ns']=56;bad[2]['query_end_ns']=76
            if field=='unknown_type':bad[1]['processes'][0]['type']='?'
            with self.subTest(field=field):
                with self.assertRaises(ValueError):s.summarize(bad,7,'GPU-1',self.bounds,40)
    def test_boundary_overlap_reported_not_asserted_contained(self):
        bad=copy.deepcopy(self.samples);bad[1]['query_start_ns']=25
        out=s.summarize(bad,7,'GPU-1',self.bounds,40)
        self.assertIn(1,out['stages']['model_load']['overlap_indices'])
        self.assertNotIn(1,out['stages']['model_load']['fully_contained_indices'])
    def test_empty_stage_retains_unknown_coverage(self):
        out=s.summarize(self.samples,7,'GPU-1',{**self.bounds,'request_start':76,'request_end':89},40)
        self.assertEqual(out['stages']['request']['overlap_metrics']['target']['known'],0)
        self.assertEqual(out['stages']['request']['overlap_metrics']['target']['max_known_bytes'],None)

if __name__=='__main__':unittest.main()
