import unittest
import host_contract as contract
class TestContract(unittest.TestCase):
    def test_bounded_allocation(self):
        p={'stage':'smoke','mode':'C-ON','jobs':24,'worker_timeout_seconds':1800,'sampler_timeout_seconds':2100,
           'outer_timeout_seconds':2400,'grace_seconds':120,'sample_interval_ns':500000000,'max_sample_gap_ns':5000000000}
        a={'authorization':'root_scheduled_C_ON_memory_smoke','stage':'smoke','mode':'C-ON','gpu_uuid':'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
           'budget_end_unix':3000,'prelaunch_review_sha256':'a'*64,'nonce':'b'*32}
        budget={'budget_start_unix':0,'budget_end_unix':3000}
        self.assertEqual(contract.validate(p,a,1,budget)['status'],'validated_fields_only')
        for key,val in [('jobs',23),('worker_timeout_seconds',0),('sampler_timeout_seconds',1800),
                        ('outer_timeout_seconds',2200),('grace_seconds',10),('sample_interval_ns',1),('max_sample_gap_ns',9999999999)]:
            x=dict(p);x[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):contract.validate(x,a,1,budget)
        for key,val in [('authorization','wrong'),('gpu_uuid','wrong'),('nonce','bad'),('budget_end_unix',100)]:
            x=dict(a);x[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):contract.validate(p,x,1,budget)
        for now in (float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError):contract.validate(p,a,now,budget)
        with self.assertRaises(ValueError):contract.validate(p,a,500,budget)
if __name__=='__main__':unittest.main()
