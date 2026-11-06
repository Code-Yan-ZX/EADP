"""Synthetic CPU tests for locked full-panel admission and terminal gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import panel_gate,panel_launch,panel_verify
from decode_observer import validate_trace


class PanelTests(unittest.TestCase):
    def test_actual_32_decode_steps_and_cache_corruption(self):
        length=12;delta=7
        trace=[{'cache_position':list(range(length)),'rope_delta':[[delta]],'layer_cache_lengths':[],
                'layer_cache_lengths_after':[length]*36,'position_ids':None}]
        for i in range(32):
            pos=length+i
            trace.append({'cache_position':[pos],'position_ids':[[[pos+delta]]]*3,'rope_delta':[[delta]],
                          'layer_cache_lengths':[pos]*36,'layer_cache_lengths_after':[pos+1]*36})
        self.assertTrue(validate_trace(trace,[5]*33,length,delta))
        for field in ('cache_position','position_ids','rope_delta','layer_cache_lengths','layer_cache_lengths_after'):
            bad=copy.deepcopy(trace);bad[17][field]=[]
            with self.assertRaises(ValueError):validate_trace(bad,[5]*33,length,delta)
    def test_full_panel_explicitly_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(panel_launch,'PANEL_APPROVAL_PATH',Path(tmp)/'missing.json'):
                with self.assertRaisesRegex(ValueError,'review'):panel_launch.check_panel_approval()

    def test_smoke_acceptance_requires_full_arrays_and_supervisor_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);run=base/'smoke';run.mkdir();(run/'arrays').mkdir()
            e1=base/'e1.json';e1.write_text('{}')
            required=[e1]+[run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','forced_records.jsonl','generated.jsonl')]
            for f in required:f.write_text('{}')
            array={}
            for i in range(264):
                p=run/'arrays'/str(i);p.write_text(str(i));array[str(p)]=panel_gate.sha(p)
            compared=[{'task':'T','qid':str(i),'group_id':'g'+str(i),'arm':arm} for i in range(32) for arm in ('P3-D0','P3-D1')]
            proof={'status':'passed','all_arms_validated':256,'forced_arms_validated':8,'forced_decode_steps':32,'GT_opened':False,'scoring_performed':False,
                   'verifier_sha256':panel_gate.sha(panel_gate.HERE/'verify_k256.py'),
                   'files_sha256':{str(p.resolve()):panel_gate.sha(p) for p in required},'array_sha256':array,'compared':compared}
            proof_path=base/'proof.json';proof_path.write_text(json.dumps(proof))
            started=base/'started';terminal=base/'terminal'
            s={'worker_pid':123,'cwd':str(base),'command':['python',str(panel_gate.HERE/'verify_k256.py'),'--out',str(proof_path)]}
            t={'worker_pid':123,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
            started.write_text(json.dumps(s));terminal.write_text(json.dumps(t))
            panel_gate.validate_smoke_gate(e1,run,proof_path,started,terminal)
            bad=copy.deepcopy(proof);bad['array_sha256'].pop(next(iter(bad['array_sha256'])));proof_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'arrays incomplete'):panel_gate.validate_smoke_gate(e1,run,proof_path,started,terminal)
            proof_path.write_text(json.dumps(proof));t['worker_returncode']=1;terminal.write_text(json.dumps(t))
            with self.assertRaisesRegex(ValueError,'supervisor'):panel_gate.validate_smoke_gate(e1,run,proof_path,started,terminal)

    def test_panel_terminal_rejects_interruption_before_any_docker_inspection(self):
        run=panel_gate.HERE/'panel_x';ident={'pid':123,'start_ticks':5,'boot_id':'old'}
        started={'worker_pid':123};terminal={'worker_pid':123,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
        launch={'controller':ident};exited={'controller':ident,'container_identity':{'container_id':'x','host_init':ident},
            'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':15,'container_stopped_verified':True,
            'client_exited_while_container_running':False,'gpu_pids_after':[]}
        with patch.object(panel_verify,'alive',return_value=False),patch.object(panel_verify.subprocess,'run',side_effect=AssertionError('unexpected Docker')):
            with self.assertRaisesRegex(ValueError,'container did not exit cleanly'):
                panel_verify.validate_terminal(run,started,terminal,launch,exited)

if __name__=='__main__':unittest.main()
