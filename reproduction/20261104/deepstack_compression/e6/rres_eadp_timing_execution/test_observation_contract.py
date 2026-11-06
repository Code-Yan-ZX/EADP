import copy
import unittest
import observation_contract as o


class Tests(unittest.TestCase):
    def rows(self):
        ns='time:[1]'; clock='clock_gettime(CLOCK_MONOTONIC)'
        snap={'event':'snapshot','index':0,'time_namespace':ns,'monotonic_clock':clock,'ps':{'returncode':0,'query_start_ns':10,'query_end_ns':11},'docker':{'returncode':0,'query_start_ns':12,'query_end_ns':13}}
        done={'event':'logging_complete','index':0,'time_namespace':ns,'monotonic_clock':clock,'first_fsync_end_ns':14}
        end={'event':'observer_end','time_namespace':ns,'monotonic_clock':clock,'samples':1,'end_ns':15,'reason':'stop'}
        return [snap,done,end],ns,clock
    def test_complete_window_and_limits(self):
        args=self.rows(); r=o.observer_windows(*args)
        self.assertEqual([x['source'] for x in r['windows']],['ps','docker','observer_first_fsync_conservative'])
        self.assertEqual(r['second_fsync'],'unknown')
    def test_missing_mixed_and_failed_rejected(self):
        for field in ('missing','namespace','clock','failed','fsync','no_samples'):
            rows,ns,clock=copy.deepcopy(self.rows())
            if field=='missing': rows.pop()
            if field=='namespace': rows[0]['time_namespace']='time:[2]'
            if field=='clock': rows[1]['monotonic_clock']='other'
            if field=='failed': rows[0]['ps']['returncode']=1
            if field=='fsync': rows[1]['first_fsync_end_ns']=12
            if field=='no_samples': rows=[{**rows[-1],'samples':0}]
            with self.subTest(field=field), self.assertRaises(ValueError): o.observer_windows(rows,ns,clock)
    def test_supervisor_and_automatic_unknown(self):
        ns='time:[1]'
        self.assertEqual(len(o.supervisor_windows([{'time_namespace':ns,'health_snapshot_window_monotonic_ns':[1,2]}],ns)),1)
        with self.assertRaises(ValueError): o.supervisor_windows([{'time_namespace':ns}],ns)
        with self.assertRaises(ValueError): o.automatic_windows([{'event':'manual_health_check','time_namespace':ns,'start_monotonic_ns':1,'end_monotonic_ns':2,'details':{'health_error':'timeout'}}],ns)
        self.assertEqual(o.automatic_windows([],ns)['unrecorded_activity'],'unknown')

    def test_sample_times_must_follow_prior_fsync(self):
        rows,ns,clock=self.rows()
        second=copy.deepcopy(rows[:2])
        for row in second: row['index']=1
        second[0]['ps'].update(query_start_ns=13,query_end_ns=15)
        second[0]['docker'].update(query_start_ns=16,query_end_ns=17)
        second[1]['first_fsync_end_ns']=18
        rows=rows[:2]+second+[{**rows[-1],'samples':2,'end_ns':20}]
        with self.assertRaisesRegex(ValueError,'sample ordering'): o.observer_windows(rows,ns,clock)
        second[0]['ps'].update(query_start_ns=14,query_end_ns=15)
        self.assertEqual(o.observer_windows(rows,ns,clock)['samples'],2)


if __name__=='__main__': unittest.main()
