"""CPU synthetic score boundary tests, without loading any real GT."""
import copy,os,re,string,unittest
from unittest.mock import patch
import pandas as pd
import score_terminal as s

class Tests(unittest.TestCase):
    def test_failed_exit_gate_never_opens_GT(self):
        with patch.object(s,'check_proof',side_effect=AssertionError('not complete')),patch.object(s.c,'sha',side_effect=AssertionError('must not hash GT')):
            with self.assertRaisesRegex(AssertionError,'not complete'):s.score()
    def test_public_text_matching_can_differ_from_local_letter(self):
        ns=dict(cp=copy,os=os,re=re,string=string,pd=pd)
        s.original_helpers(s.UTILS/'utils/matching_util.py',['can_infer_option','can_infer_text','can_infer'],ns)
        s.original_helpers(s.UTILS/'dataset/utils/multiple_choice.py',['build_choices','prefetch_answer','prefetch_circular_group'],ns)
        row=dict(A='apple',B='banana',prediction='banana',GT='B')
        self.assertEqual(ns['prefetch_answer'](pd.Series(row)),'B')
        self.assertIsNone(re.search(r'\b([A-E])\b',row['prediction']))
        self.assertEqual(ns['prefetch_circular_group'](pd.DataFrame([row]))['hit'],1)
        self.assertIsNone(ns['prefetch_circular_group'](pd.DataFrame([{**row,'prediction':'unclear'}])))
        self.assertEqual(ns['prefetch_circular_group'](pd.DataFrame([{**row,'prediction':'A'}]))['hit'],0)

class ProofChainTests(unittest.TestCase):
    def test_bound_processes_hashes_and_order_reject_tampering(self):
        import hashlib,json,tempfile
        from pathlib import Path
        def save(path,obj):path.write_text(json.dumps(obj),encoding='utf-8')
        def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            here=Path(td);out=here/'execution';out.mkdir();(here/'inputs').mkdir();planpath=here/'plan.json'
            plan={'run_id':'test'};save(planpath,plan);psha=sha(planpath)
            allocation={'authorization':'explicit_root_gpu_allocation','run_id':'test','plan_sha256':psha};save(here/'root_gpu_allocation.json',allocation);asha=sha(here/'root_gpu_allocation.json')
            controller=dict(pid=11,start_ticks=1,boot_id='boot');worker=dict(pid=12,start_ticks=2,boot_id='boot');client=dict(pid=13,start_ticks=3,boot_id='boot')
            claim=dict(controller=controller,plan_sha256=psha,allocation_sha256=asha);save(here/'launch.claim.json',claim)
            ident=dict(identity=worker,parent=controller,plan_sha256=psha,allocation_sha256=asha);save(out/'worker_identity.json',ident)
            state=dict(status='complete',plan_sha256=psha,allocation_sha256=asha,controller=controller,worker=worker,verifier_client=client,native_exit_code=0,verifier_exit_code=0,verifier_container_absent=True);save(out/'state.json',state)
            release=dict(status='native_exited_validation_pending',native_exit_code=0,original_worker_exited=True,GPU_pid_absent=True,plan_sha256=psha,allocation_sha256=asha,controller=controller,worker=worker);save(out/'native_release.json',release)
            save(out/'anchor_gate.json',{'passed':True});save(out/'runtime.json',{'x':1})
            rows=[dict(index=i,group_id=i%1176) for i in range(4377)];save(here/'inputs/manifest.json',dict(rows=rows))
            ordered={};pred_sha={};obs_sha={}
            for arm in s.ARMS:
                pred=out/(arm+'_predictions.jsonl');obs=out/(arm+'_observations.jsonl')
                pred.write_text(''.join(json.dumps(dict(position=i,index=r['index'],group_id=r['group_id'],text='A'))+'\n' for i,r in enumerate(rows)))
                obs.write_text(''.join(json.dumps(dict(position=i,index=r['index']))+'\n' for i,r in enumerate(rows)))
                ordered[arm]=dict(predictions=dict(path=str(pred),sha256=sha(pred)),observations=dict(path=str(obs),sha256=sha(obs)))
                pred_sha[arm]=sha(pred);obs_sha[arm]=sha(obs)
            complete=dict(status='complete',plan_sha256=psha,n_rows=4377,n_generations=8754,scoring=False,ordered=ordered,anchor_gate=dict(sha256=sha(out/'anchor_gate.json')),runtime=dict(sha256=sha(out/'runtime.json')));save(out/'native_complete.json',complete)
            report=dict(success=True,plan_sha256=psha,n_rows=4377,n_generations=8754,n_anchor_rows=60,CPU_only=True,model_loaded=False,CUDA_initialized=False,GT_read=False,scoring_started=False,native_complete_sha256=sha(out/'native_complete.json'),native_release_sha256=sha(out/'native_release.json'),anchor_gate_sha256=sha(out/'anchor_gate.json'),runtime_sha256=sha(out/'runtime.json'),prediction_sha256=pred_sha,observation_sha256=obs_sha);save(out/'generation_verification.json',report)
            proof=dict(root_pty_actual_exit_code=0,root_pty_exit_observed_by='functions.write_stdin',root_pty_session=123,verifier_container_not_running=True,state_sha256=sha(out/'state.json'),verifier_sha256=sha(out/'generation_verification.json'),allocation_sha256=asha,process_checks=[dict(role=role,original=identity,original_process_absent=True) for role,identity in [('controller',controller),('worker',worker),('verifier_client',client)]])
            proofpath=here/'root_exit_proof.json';save(proofpath,proof)
            with patch.object(s.c,'HERE',here),patch.object(s.c,'OUT',out),patch.object(s.c,'PLAN',planpath),patch.object(s.c,'check'),patch.object(s.c,'identity',return_value=None):
                self.assertEqual(len(s.check_proof()[-1]['source']),4377)
                for key,value in [('root_pty_actual_exit_code',1),('root_pty_exit_observed_by','other'),('root_pty_session',0),('verifier_sha256','0'*64),('allocation_sha256','0'*64)]:
                    with self.subTest(key=key):
                        save(proofpath,{**proof,key:value})
                        with self.assertRaises(AssertionError):s.check_proof()
                save(proofpath,{**proof,'process_checks':proof['process_checks'][:-1]})
                with self.assertRaises(AssertionError):s.check_proof()
                save(proofpath,proof)
                for path in (here/'root_gpu_allocation.json',out/'native_complete.json',out/'native_release.json',out/'source_predictions.jsonl',out/'source_observations.jsonl',out/'question_only_predictions.jsonl',out/'question_only_observations.jsonl'):
                    original=path.read_bytes();path.write_bytes(original+b' ')
                    try:
                        with self.subTest(path=path.name),self.assertRaises((AssertionError,ValueError)):s.check_proof()
                    finally:path.write_bytes(original)

if __name__=='__main__':unittest.main()
