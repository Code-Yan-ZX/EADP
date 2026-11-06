"""Synthetic CPU tests for locked full-panel admission and terminal gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from panel_v2_imports import load_local
panel_gate=load_local('panel_gate');panel_launch=load_local('panel_launch');panel_verify=load_local('panel_verify')
native_panel_gate=load_local('native_panel_gate')
from decode_observer import validate_trace


class PanelTests(unittest.TestCase):
    def test_host_native_receipt_is_bound_and_requires_live_inspection(self):
        import runpy
        worker=runpy.run_path(str(panel_gate.HERE/'panel_worker.py'),run_name='panel_test')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);native={'proof_sha256':'p','review_sha256':'r'};receipt=out/'host_native_gate.json'
            data={'native_gate':native,'native_gate_source_sha256':panel_gate.sha(panel_gate.HERE/'native_panel_gate.py'),
                  'host_live_docker_inspection':True}
            receipt.write_text(json.dumps(data));launch={'host_native_gate':{'path':str(receipt),'sha256':panel_gate.sha(receipt)}}
            (out/'launch.json').write_text(json.dumps(launch));worker['validate_host_receipt'](out,native)
            data['host_live_docker_inspection']=False;receipt.write_text(json.dumps(data))
            with self.assertRaises(ValueError):worker['validate_host_receipt'](out,native)
            launch['host_native_gate']['sha256']=panel_gate.sha(receipt);(out/'launch.json').write_text(json.dumps(launch))
            with self.assertRaises(ValueError):worker['validate_host_receipt'](out,native)
    def test_local_modules_ignore_preloaded_legacy_names(self):
        import sys,types
        old=types.ModuleType('panel_launch');old.__file__=str(panel_gate.E4/'panel_launch.py')
        with patch.dict(sys.modules,{'panel_launch':old}):
            module=load_local('panel_launch')
            self.assertEqual(Path(module.__file__).resolve(),panel_gate.HERE/'panel_launch.py')
            self.assertIsNot(module,old)
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
                   'verifier_sha256':panel_gate.sha(panel_gate.E4/'verify_k256.py'),
                   'files_sha256':{str(p.resolve()):panel_gate.sha(p) for p in required},'array_sha256':array,'compared':compared}
            proof_path=base/'proof.json';proof_path.write_text(json.dumps(proof))
            started=base/'started';terminal=base/'terminal'
            s={'worker_pid':123,'cwd':str(base),'command':['python',str(panel_gate.E4/'verify_k256.py'),'--out',str(proof_path)]}
            t={'worker_pid':123,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
            started.write_text(json.dumps(s));terminal.write_text(json.dumps(t))
            panel_gate.validate_smoke_gate(e1,run,proof_path,started,terminal)
            bad=copy.deepcopy(proof);bad['array_sha256'].pop(next(iter(bad['array_sha256'])));proof_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'arrays incomplete'):panel_gate.validate_smoke_gate(e1,run,proof_path,started,terminal)
            proof_path.write_text(json.dumps(proof));t['worker_returncode']=1;terminal.write_text(json.dumps(t))
            with self.assertRaisesRegex(ValueError,'supervisor'):panel_gate.validate_smoke_gate(e1,run,proof_path,started,terminal)

    def test_panel_terminal_rejects_interruption_before_any_docker_inspection(self):
        run=panel_gate.HERE/'panel_x';ident={'pid':123,'start_ticks':5,'boot_id':'old'}
        started={'worker_pid':123,'process_group':123,'timeout_seconds':10920,'stale_seconds':1800,'snapshot_seconds':3600,
                 'poll_seconds':15,'grace_seconds':90,'progress_paths':[str((run/'generated.jsonl').resolve())]}
        terminal={'worker_pid':123,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
        launch={'controller':ident,'started_unix':1793810000};exited={'controller':ident,'container_identity':{'container_id':'x','host_init':ident},
            'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':15,'container_stopped_verified':True,
            'client_exited_while_container_running':False,'gpu_pids_after':[]}
        with patch.object(panel_verify,'alive',return_value=False),patch.object(panel_verify.subprocess,'run',side_effect=AssertionError('unexpected Docker')):
            with self.assertRaisesRegex(ValueError,'container did not exit cleanly'):
                panel_verify.validate_terminal(run,started,terminal,launch,exited)

    def test_budget_monitor_requires_bounded_and_watched_configuration(self):
        run=panel_gate.HERE/'panel_x'
        good={'worker_pid':123,'process_group':123,'timeout_seconds':10920,'stale_seconds':1800,'snapshot_seconds':3600,
              'poll_seconds':15,'grace_seconds':90,'progress_paths':[str((run/'generated.jsonl').resolve())]}
        panel_gate.validate_budget_supervisor(good,run,123)
        for key,value in [('stale_seconds',None),('timeout_seconds',99999),('progress_paths',[]),('worker_pid',124)]:
            bad=dict(good);bad[key]=value
            with self.assertRaises(ValueError):panel_gate.validate_budget_supervisor(bad,run,123)

    def test_budget_deadline_is_bound_to_three_day_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'initial.json'
            path.write_text(json.dumps({'budget_start_unix':100,'budget_end_unix':1000,'budget_end_beijing':'fixed'}))
            plan={'budget_source':str(path),'budget_source_sha256':panel_gate.sha(path),'budget_start_unix':100,
                  'budget_end_unix':1000,'budget_end_beijing':'fixed','outer_timeout_seconds':100}
            panel_gate.validate_deadline(plan,900)
            with self.assertRaisesRegex(ValueError,'remaining'):panel_gate.validate_deadline(plan,901)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'source changed'):panel_gate.validate_deadline(plan,900)

    def test_smoke_independent_review_and_exit_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);proof=d/'smoke.json';proof.write_text('{}');vt=d/'verify_terminal';vt.write_text('{}')
            source=d/'review.py';source.write_text('review');review=d/'review.json';started=d/'started';terminal=d/'terminal'
            r={'status':'passed','natural_verified':256,'forced_verified':8,'errors':[],
               'proof_sha256':panel_gate.sha(proof),'verifier_terminal_sha256':panel_gate.sha(vt),'reviewer_sha256':panel_gate.sha(source)}
            review.write_text(json.dumps(r));started.write_text(json.dumps({'worker_pid':1,'cwd':str(d),
                'command':['python',str(source),'--out',str(review)]}))
            t={'worker_pid':1,'reason':'completed','worker_returncode':0,'remaining_process_group':[]};terminal.write_text(json.dumps(t))
            with patch.object(panel_gate,'REVIEW',review),patch.object(panel_gate,'REVIEW_SOURCE',source), \
                 patch.object(panel_gate,'REVIEW_STARTED',started),patch.object(panel_gate,'REVIEW_TERMINAL',terminal):
                panel_gate.validate_smoke_review(proof,vt)
                t['worker_returncode']=1;terminal.write_text(json.dumps(t))
                with self.assertRaisesRegex(ValueError,'review incomplete'):panel_gate.validate_smoke_review(proof,vt)

    def test_native_gate_rejects_missing_independent_review_or_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);native=base/'native';native.mkdir();run=native/'native_001';run.mkdir();(run/'arrays').mkdir()
            for name in ('native_verify.py',): (native/name).write_text(name)
            for name in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl','comparisons.jsonl'):
                (run/name).write_text('{}')
            run_started=base/'run_started';run_terminal=base/'run_terminal';vs=base/'vs';vt=base/'vt'
            for path in (run_started,run_terminal,vs,vt):path.write_text('{}')
            sources=[run_started,run_terminal]+[run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl','comparisons.jsonl')]
            arrays={}
            for i in range(12):
                path=run/'arrays'/str(i);path.write_text(str(i));arrays[str(path)]=native_panel_gate.sha(path)
            proof=base/'proof';proof_data={'status':'passed','stage':'stock_native_k1024','tasks':3,'pairs':6,'natural_pairs':3,'forced_pairs':3,
                 'forced_decode_steps':32,'kv_fingerprints_compared':True,'GT_opened':False,'scoring_performed':False,
                 'verifier_sha256':native_panel_gate.sha(native/'native_verify.py'),
                 'files_sha256':{str(p.resolve()):native_panel_gate.sha(p) for p in sources},'array_sha256':arrays}
            proof.write_text(json.dumps(proof_data));review_src=base/'review.py';review_src.write_text('review')
            review=base/'review.json';review.write_text(json.dumps({'status':'passed','pairs_expected':6,'pairs_verified':6,'errors':[],
                'formal_proof_sha256':native_panel_gate.sha(proof),'formal_verifier_terminal_sha256':native_panel_gate.sha(vt),
                'reviewer_sha256':native_panel_gate.sha(review_src)}))
            rs=base/'review_started';rt=base/'review_terminal'
            rs.write_text(json.dumps({'worker_pid':1,'cwd':str(base),'command':['python',str(review_src)]}))
            rt.write_text(json.dumps({'worker_pid':1,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}))
            import native_verify
            changes={'NATIVE':native,'RUN':run,'PROOF':proof,'RUN_STARTED':run_started,'RUN_TERMINAL':run_terminal,
                'VERIFIER_STARTED':vs,'VERIFIER_TERMINAL':vt,'REVIEW_PROOF':review,'REVIEW_SOURCE':review_src,
                'REVIEW_STARTED':rs,'REVIEW_TERMINAL':rt}
            from contextlib import ExitStack
            with ExitStack() as stack:
                for key,value in changes.items():stack.enter_context(patch.object(native_panel_gate,key,value))
                stack.enter_context(patch.object(native_panel_gate,'validate_supervisor'))
                stack.enter_context(patch.object(native_verify,'terminal'))
                native_panel_gate.check_native()
                bad=dict(proof_data);bad['array_sha256']=dict(arrays);bad['array_sha256'].pop(next(iter(arrays)));proof.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'array evidence'):native_panel_gate.check_native()
                proof.write_text(json.dumps(proof_data));failed=json.loads(review.read_text());failed['status']='failed';review.write_text(json.dumps(failed))
                with self.assertRaisesRegex(ValueError,'independent native review'):native_panel_gate.check_native()

if __name__=='__main__':unittest.main()
