"""Synthetic failure classification: no CUDA, GT, or real result access."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import verify as v

class TestVerify(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.d=Path(self.tmp.name);self.run=self.d/'run_smoke001';self.obs=self.run/'observations';self.mon=self.d/'monitor_smoke001'
        self.obs.mkdir(parents=True);self.mon.mkdir()
        self.entries=[]
        for i in range(32):
            for k in (128,256,512):
                path=self.d/f'case_{i}_{k}.pt';path.write_bytes(b'synthetic')
                self.entries.append({'image_index':i,'task':'DEV','qid':str(i),'group_id':str(i),'budget':k,'path':str(path),'sha256':v.sha(path)})
        self.write(self.d/'plan001.json',{'case_manifest':self.entries})
        self.write(self.d/'preflight001.json',{});self.write(self.d/'independent_prelaunch_review.json',{})
        command=['docker','run','--entrypoint','/bin/python','--user','1000','--workdir','/work']
        controller={'pid':100};self.write(self.run/'launch.json',{'plan_sha256':v.sha(self.d/'plan001.json'),'stage':'smoke','controller':controller,'container_name':'test','command':command})
        self.write(self.run/'exit.json',{'controller':controller,'returncode':0,'container_exit_code':0,'container_stopped_verified':True,'timed_out':False,'interrupted':None,'failure_type':None,'client_exited_while_container_running':False,'gpu_pids_after':[],'container_identity':{'container_id':'a','image':'im'}})
        self.write(self.mon/'started.json',{'worker_pid':100});self.write(self.mon/'terminal.json',{'worker_pid':100,'reason':'completed','worker_returncode':0,'remaining_process_group':[]})
        self.attempts=[{'case':f'k_{k:03d}','returncode':-15,'timed_out':True} for k in (128,256,512)]
        self.write(self.obs/'worker_complete.json',{'status':'attempts_recorded','cases':self.attempts,'GT_opened':False,'scoring_performed':False})
        (self.obs/'progress.jsonl').write_text('{}\n')
        for a in self.attempts:
            (self.obs/(a['case']+'.stdout.log')).write_text('');(self.obs/(a['case']+'.stderr.log')).write_text('')
        obj={'Id':'a','Image':'im','Name':'/test','State':{'Running':False,'Pid':0,'ExitCode':0},
             'Config':{'Entrypoint':['/bin/python'],'Cmd':[str(self.d/'worker.py'),'--out',str(self.obs)],'User':'1000','WorkingDir':'/work'},
             'HostConfig':{'NetworkMode':'none','PidMode':'','Devices':[{'PathOnHost':x} for x in ('/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools')]}}
        self.patches=[patch.object(v,'HERE',self.d),patch.object(v,'RUN',self.run),patch.object(v,'OBS',self.obs),patch.object(v,'MON',self.mon),
            patch.object(v.inventory,'inventory',return_value=self.entries),patch.object(v.inventory,'require_scatter_review',return_value={}),
            patch.object(v.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps([obj]))),patch.object(v.torch.cuda,'is_initialized',return_value=False)]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    @staticmethod
    def write(path,obj):path.write_text(json.dumps(obj))
    def test_all_timeout_is_complete_evidence_not_identity_success(self):
        result=v.check();self.assertEqual(result['status'],'passed');self.assertFalse(result['complete_96_identity'])
        self.assertTrue(all(x['outcome']=='timeout' for x in result['budgets']))
    def test_missing_32_cannot_be_reported_success(self):
        self.attempts[0].update(returncode=0,timed_out=False)
        self.write(self.obs/'worker_complete.json',{'status':'attempts_recorded','cases':self.attempts,'GT_opened':False,'scoring_performed':False})
        with self.assertRaises(ValueError):v.check()
    def test_numerical_mismatch_distinct_from_capture_failure(self):
        self.attempts[0].update(returncode=1,timed_out=False)
        self.write(self.obs/'worker_complete.json',{'status':'attempts_recorded','cases':self.attempts,'GT_opened':False,'scoring_performed':False})
        item=self.entries[0];author=list(range(128));actual=author.copy();actual[0],actual[1]=actual[1],actual[0]
        row={'event':'case_compared','k':128,**{x:item[x] for x in ('image_index','task','qid','group_id')},'case_sha256':item['sha256'],
             'author_order':[author],'graph_order':[actual],'author_quota':[128],'graph_quota':[128],'graph_equals_author':False,
             'author_equals_archive':True,'repeat_same_input_equal':True,'input_copy_plus_replay_device_ms':1.,'host_wall_ms':2.,'graph_only_device_ms':.5}
        (self.obs/'k_128.jsonl').write_text(json.dumps(row)+'\n'+json.dumps({'event':'failed','error_type':'ValueError'})+'\n')
        with patch.object(v.torch,'load',return_value={'ordered':torch.tensor([author])}):result=v.check()
        self.assertFalse(result['complete_96_identity']);self.assertEqual(result['budgets'][0]['identity_mismatch_images'],[0])

if __name__=='__main__':unittest.main(verbosity=2)
