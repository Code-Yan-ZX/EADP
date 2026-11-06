import json
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
import nvml_source as s
import verify_metadata
import real_sampler
from sampler import sample_process

XML='''<nvidia_smi_log><gpu><uuid>GPU-abcd</uuid><fb_memory_usage>
<total>100 MiB</total><used>45 MiB</used><free>50 MiB</free><reserved>5 MiB</reserved>
</fb_memory_usage><processes>
<process_info><pid>42</pid><type>C</type><process_name>secret-argument</process_name><used_memory>20 MiB</used_memory></process_info>
<process_info><pid>77</pid><type>G</type><process_name>other-secret</process_name><used_memory>8 MiB</used_memory></process_info>
</processes></gpu></nvidia_smi_log>'''
CMD=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits']

class TestNvmlSource(unittest.TestCase):
    def test_real_sampler_state_machine_with_synthetic_xml_windows(self):
        expected={'mode':'t0_author','nonce':'a'*32,'container_id':'b'*64,'gpu_uuid':'GPU-abcd',
                  'host_pid':42,'start_ticks':123,'boot_id':'boot','max_sample_gap_ns':100}
        absent=XML.replace('<process_info><pid>42</pid><type>C</type><process_name>secret-argument</process_name><used_memory>20 MiB</used_memory></process_info>','')
        queries=iter([absent,XML,absent]);alive=iter([True,True,True,False]);tick=iter(range(1,1000));events=[]
        with tempfile.TemporaryFile(mode='w+') as log:
            source=s.source(expected,log,run=lambda *a,**k:SimpleNamespace(stdout=next(queries)),clock=lambda:next(tick))
            def query():return source(CMD).stdout
            result=sample_process(expected,lambda _:next(alive),query,lambda:next(tick),lambda _:None,
                                  events.append,lambda *_:None,lambda _:0,1,1000)
            log.seek(0);metadata=[json.loads(line) for line in log]
        samples=[x for x in events if x.get('type')=='sample']
        x=verify_metadata.verify(expected,metadata,samples)
        self.assertEqual(result['samples'],3);self.assertEqual(x['sample_count'],3)
        self.assertEqual(x['worker_used_max_of_present_samples_bytes'],20*1048576)
        self.assertEqual(x['device_used_max_of_known_samples_bytes'],45*1048576)
        self.assertEqual(x['worker_presence'],{'present':1,'absent_before_first':1,'absent_between_first_and_last':0,'absent_after_last':1})
        self.assertEqual(len(x['graphics_observations']),3)

    def test_actual_sampler_modules_resolve_to_memory_directory(self):
        memory=Path(real_sampler.MEMORY).resolve()
        self.assertEqual(Path(sys.modules['sampler'].__file__).resolve(),memory/'sampler.py')
        self.assertEqual(Path(sys.modules['contract'].__file__).resolve(),memory/'contract.py')

    def test_sampler_bounds_and_injected_source(self):
        with tempfile.TemporaryDirectory() as td:
            d=__import__('pathlib').Path(td)
            expected={'mode':'t0_author','nonce':'a'*32,'container_id':'b'*64,'gpu_uuid':'GPU-abcd',
                      'host_pid':42,'start_ticks':123,'boot_id':'boot','max_sample_gap_ns':1000000000}
            (d/'expected.json').write_text(json.dumps(expected))
            seen=[]
            def sample(*args,**kw):seen.append((args,kw));return {'status':'completed','samples':3}
            result=real_sampler.launch(d,100000000,1000000000,sample=sample,source_factory=lambda e,stream:'fake-source')
            self.assertEqual(result['sample_count'],3);self.assertEqual(seen[0][1]['run'],'fake-source')
            self.assertEqual(result['scope'],'NVML process/device samples, not timing')
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):real_sampler.launch(td,1,2)

    def test_one_to_one_window_and_worker_device_separation(self):
        m=s.parse_xml(XML,'GPU-abcd');m.update(status='completed',query_start_ns=101,query_end_ns=110)
        sample={'gpu_uuid':'GPU-abcd','host_pid':42,'query_start_ns':100,'query_end_ns':111,
                'monotonic_ns':111,'used_bytes':20*1048576,'other_gpu_processes':[]}
        x=verify_metadata.verify({'gpu_uuid':'GPU-abcd','host_pid':42},[m],[sample])
        self.assertEqual(x['device_used_max_of_known_samples_bytes'],45*1048576)
        self.assertEqual(x['worker_used_max_of_present_samples_bytes'],20*1048576)
        self.assertEqual(x['device_total_known_values_bytes'],[100*1048576])
        self.assertEqual(x['graphics_observations'][0]['other_graphics_processes'][0]['pid'],77)
        for badm,bads in [([], [sample]),([dict(m,query_start_ns=99)],[sample]),
                          ([m],[dict(sample,used_bytes=45*1048576)]),
                          ([dict(m,gpu_uuid='GPU-other')],[sample])]:
            with self.assertRaises(ValueError):verify_metadata.verify({'gpu_uuid':'GPU-abcd','host_pid':42},badm,bads)

    def test_unknown_device_sample_and_absent_worker_remain_unknown(self):
        first=s.parse_xml(XML,'GPU-abcd');first.update(status='completed',query_start_ns=101,query_end_ns=110)
        second=s.parse_xml(XML,'GPU-abcd');second.update(status='completed',query_start_ns=201,query_end_ns=210)
        second['device_memory_bytes']['used']=None;second['device_memory_bytes']['total']=None
        second['processes']=[p for p in second['processes'] if p['type']!='C']
        a={'gpu_uuid':'GPU-abcd','host_pid':42,'query_start_ns':100,'query_end_ns':111,
           'monotonic_ns':111,'used_bytes':20*1048576,'other_gpu_processes':[]}
        b={'gpu_uuid':'GPU-abcd','host_pid':42,'query_start_ns':200,'query_end_ns':211,
           'monotonic_ns':211,'used_bytes':None,'other_gpu_processes':[]}
        x=verify_metadata.verify({'gpu_uuid':'GPU-abcd','host_pid':42},[first,second],[a,b])
        self.assertEqual(x['device_field_coverage']['used'],{'known':1,'unknown':1,'complete':False})
        self.assertEqual(x['device_field_coverage']['total'],{'known':1,'unknown':1,'complete':False})
        self.assertEqual(x['device_used_max_of_known_samples_bytes'],45*1048576)
        self.assertEqual(x['worker_presence']['absent_after_last'],1)
        self.assertEqual(x['worker_used_max_of_present_samples_bytes'],20*1048576)

    def test_device_and_process_are_distinct_and_sanitized(self):
        meta=s.parse_xml(XML,'GPU-abcd')
        self.assertEqual(meta['device_memory_bytes']['total'],100*1048576)
        self.assertEqual(meta['device_memory_bytes']['used'],45*1048576)
        self.assertEqual(meta['processes'],[{'pid':42,'type':'C','used_bytes':20*1048576},
                                            {'pid':77,'type':'G','used_bytes':8*1048576}])
        self.assertNotIn('secret',json.dumps(meta))
    def test_query_window_and_graphics_record(self):
        with tempfile.TemporaryFile(mode='w+') as log:
            ticks=iter([100,110]);calls=[]
            def run(args,**kwargs):calls.append((args,kwargs));return SimpleNamespace(stdout=XML)
            value=s.source({'gpu_uuid':'GPU-abcd'},log,run=run,clock=lambda:next(ticks))(CMD)
            self.assertEqual(value.stdout,'GPU-abcd, 42, 20')
            log.seek(0);saved=json.loads(log.read());self.assertEqual((saved['query_start_ns'],saved['query_end_ns']),(100,110))
            self.assertEqual(saved['processes'][1]['type'],'G');self.assertNotIn('secret',json.dumps(saved))
            self.assertEqual(calls[0][0],['nvidia-smi','-q','-x']);self.assertEqual(calls[0][1]['timeout'],10)
    def test_unknown_and_invalid_values(self):
        self.assertIsNone(s.mib('N/A'))
        with self.assertRaises(ValueError):s.mib('20 MB')
        for text,uuid in [(XML,'GPU-other'),(XML.replace('<type>G</type>','<type>?</type>'),'GPU-abcd'),
                          (XML.replace('<pid>77</pid>','<pid>42</pid>'),'GPU-abcd')]:
            with self.assertRaises(ValueError):s.parse_xml(text,uuid)
    def test_failure_record_never_saves_raw_error(self):
        with tempfile.TemporaryFile(mode='w+') as log:
            ticks=iter([100,110])
            def fail(*_a,**_kw):raise RuntimeError('secret-error-content')
            q=s.source({'gpu_uuid':'GPU-abcd'},log,run=fail,clock=lambda:next(ticks))
            with self.assertRaises(RuntimeError) as failure:q(CMD)
            self.assertNotIn('secret',str(failure.exception))
            log.seek(0);x=log.read();self.assertNotIn('secret',x);self.assertEqual(json.loads(x)['error_type'],'RuntimeError')

if __name__=='__main__':unittest.main()
