import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import b_gate

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path

class GateTests(unittest.TestCase):
    def test_conflicting_e4_dependency_is_rejected(self):
        import sys,types
        fake=types.ModuleType('run_cpu_checks');fake.__file__='/tmp/wrong/run_cpu_checks.py'
        with patch.dict(sys.modules,{'run_cpu_checks':fake}):
            with self.assertRaisesRegex(ValueError,'dependency module identity mismatch'):
                b_gate._analysis()
    def test_reject_replaced_cached_panel_gate_module(self):
        import sys,types
        fake=types.ModuleType('eadp_b_panel_scoring_gate');fake.__file__='/tmp/score_e4.py'
        with patch.dict(sys.modules,{'eadp_b_panel_scoring_gate':fake}):
            with self.assertRaisesRegex(ValueError,'identity mismatch'):b_gate._analysis()
    def test_synthetic_acceptance_and_missing_or_changed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);here=root/'b';here.mkdir();(here/'dummy.py').write_text('source')
            source=str(here/'dummy.py');digest=sha(source);panelrun=root/'panel';panelrun.mkdir()
            verifier=put(root/'e4_panel_v2/panel_verify.py',{})
            fields={k:root/k for k in b_gate.PANEL_FIELDS if k!='run'};fields['run']=panelrun
            for k,p in fields.items():
                if k not in ('run','panel_proof'):put(p,{})
            for name in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl'):
                (panelrun/name).write_text('opaque not JSON')
            required=[fields[x] for x in ('e1_acceptance','e1_started','e1_terminal','run_started','run_terminal')]
            required += [panelrun/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl')]
            put(fields['panel_proof'],{'status':'passed','stage':'full','n_questions':384,'all_arms_validated':3072,
                'GT_opened':False,'scoring_performed':False,'verifier_sha256':sha(verifier),
                'files_sha256':{str(p.resolve()):sha(p) for p in required}})
            cpu=put(root/'cpu',{'status':'passed','returncode':0,'timeout':False,'source_sha256_before':{source:digest},'source_sha256_after':{source:digest}})
            cpu_started=put(root/'cpu_started',{});cpu_terminal=put(root/'cpu_terminal',{})
            (here/'b_worker.py').write_text('worker');(here/'b_import_regression.py').write_text('regression')
            (here/'b_gate.py').write_text('gate')
            imp=put(here/'b_import_preflight_002.json',{'status':'passed','returncode':0,'container_running':False,'container_exit_code':0,
                'timed_out':False,'worker_sha256':sha(here/'b_worker.py'),'regression_sha256':sha(here/'b_import_regression.py'),
                'gate_sha256':sha(here/'b_gate.py'),'result':{'cuda_initialized':False}})
            review=put(root/'review',{'status':'passed','independent_review':True,'cpu_proof_sha256':sha(cpu),
                'source_sha256':{source:digest},'panel_proof_sha256':sha(fields['panel_proof']),'panel_review_sha256':sha(fields['review_proof']),
                'import_preflight_sha256':sha(imp),'cpu_started_sha256':sha(cpu_started),'cpu_terminal_sha256':sha(cpu_terminal)})
            manifest=put(here/'approval',{'cpu_proof':str(cpu),'cpu_started':str(cpu_started),'cpu_terminal':str(cpu_terminal),
                'b_review_proof':str(review),'panel':{k:str(v) for k,v in fields.items()}})
            fake=SimpleNamespace(validate_supervisor=lambda *_:None,validate_independent_review=lambda *_:None)
            with patch.object(b_gate,'HERE',here),patch.object(b_gate,'ROOT',root),patch.object(b_gate,'APPROVAL',manifest), \
                 patch.object(b_gate,'SOURCES',('dummy.py',)),patch.object(b_gate,'_analysis',return_value=fake):
                b_gate.check_approval(preflight_only=True)
                with self.assertRaisesRegex(ValueError,'preflight'):b_gate.check_approval()
                (panelrun/'generated.jsonl').write_text('changed')
                with self.assertRaisesRegex(ValueError,'artifact'):b_gate.check_approval(preflight_only=True)

if __name__=='__main__':unittest.main()
