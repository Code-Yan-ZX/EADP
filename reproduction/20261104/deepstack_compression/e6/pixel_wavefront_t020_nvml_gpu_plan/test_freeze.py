"""Fake-only freeze main: nine jobs and preflight proof are mandatory."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import freeze,freeze_formal,freeze_analysis
class FreezeTest(unittest.TestCase):
 def test_main_with_fake_inputs_then_bad_review(self):
  with tempfile.TemporaryDirectory() as td:
   b=Path(td);e=b/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8';e.mkdir(parents=True);p=b/'e6/pixel_wavefront_t020_nvml_execution_preflight_v2';p.mkdir(parents=True);i=b/'e1_independent_review';i.mkdir();d=b/'e6/pixel_wavefront_t020_nvml_gpu_plan';d.mkdir(parents=True)
   jobs=[{'order':i} for i in range(9)];source={str((e/'gate.py').resolve()):'digest',str((b/'e6/pixel_wavefront_t020_timing_final_v9/gpu_gate.py').resolve()):'digest'};fields=('plan_sha256','worker_preflight_sha256','launch_sha256','identity_sha256','exit_sha256','started_sha256','terminal_sha256','reviewer_sha256');review={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'errors':[],**{k:'digest' for k in fields}}
   def read(path):
    name=Path(path).name
    if name=='plan001.json':return {'source_sha256':source}
    if name=='pixel_t020_nvml_execution_preflight_v2_result_review.json':return review
    if name in ('cpu_proof002.json','cpu_proof004.json'):return {'status':'passed','GPU_used':False,'GT_opened':False,'timing_results_opened':False,'source_sha256_before':{},'source_sha256_after':{}}
    if name=='cpu_proof001.json':return {'status':'passed','GPU_used':False,'GT_opened':False,'source_sha256_before':{},'source_sha256_after':{}}
    if name=='design001.json':return {'jobs':jobs}
    if name=='initial_plan.json':return {'budget_end_unix':2000000000}
    raise AssertionError(name)
   gate=SimpleNamespace(terminal=lambda *x:None);weight=SimpleNamespace(weight_preflight=lambda:{'worker_preflight_sha256':'w','expected':[{'path':str(i)} for i in range(4)]})
   def execute():
    with patch.multiple(freeze,B=b,D=d,E=e,P=p,I=i),patch.object(freeze,'read',side_effect=read),patch.object(freeze,'sha',return_value='digest'),patch.object(freeze,'load',side_effect=lambda n,*args:gate if n.endswith('plan_gate') else weight):freeze.main()
   execute();out=e/'gpu_plan001.json';v=json.loads(out.read_text());self.assertEqual(len(v['jobs']),9);self.assertFalse(v['GPU_used']);out.unlink();review['status']='failed'
   with self.assertRaises(ValueError):execute()
   self.assertFalse(out.exists())
class FormalFreezeTest(unittest.TestCase):
 def test_formal_binding_and_source_drift(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);e=root/'e';a=root/'a';e.mkdir();a.mkdir();gpu=e/'gpu_plan001.json';gpu.write_text('{}');source={'file':'digest'};cpu={'status':'passed','source_sha256_before':source,'source_sha256_after':source};plan={'GPU_used':False,'source_sha256':source}
   def read(path):return plan if Path(path)==gpu else cpu
   with patch.multiple(freeze_formal,E=e,A=a),patch.object(freeze,'read',side_effect=read),patch.object(freeze,'sha',return_value='digest'):
    freeze_formal.main();out=a/'verification_plan001.json';self.assertEqual(json.loads(out.read_text())['gpu_plan_sha256'],'digest');out.unlink();cpu['source_sha256_after']={'file':'changed'}
    with self.assertRaises(ValueError):freeze_formal.main()
    self.assertFalse(out.exists())
class AnalysisFreezeTest(unittest.TestCase):
 def test_bound_before_run_and_bad_proof_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);e=root/'e';a=root/'a';s=root/'s'
   for p in (e,a,s):p.mkdir()
   gpu=e/'gpu_plan001.json';formal=a/'verification_plan001.json';gpu.write_text('{}');formal.write_text('{}');source={'file':'digest'};cp={'status':'passed','GPU_used':False,'GT_opened':False,'timing_results_opened':False,'source_sha256_before':source,'source_sha256_after':source}
   def read(path):
    if Path(path)==gpu:return {'GPU_used':False,'source_sha256':source}
    if Path(path)==formal:return {'gpu_plan_sha256':'digest'}
    return cp
   with patch.multiple(freeze_analysis,E=e,A=a,S=s),patch.object(freeze,'read',side_effect=read),patch.object(freeze,'sha',return_value='digest'):
    freeze_analysis.main();out=s/'analysis_plan001.json';self.assertEqual(json.loads(out.read_text())['formal_plan_sha256'],'digest');out.unlink();cp['timing_results_opened']=True
    with self.assertRaises(ValueError):freeze_analysis.main()
    self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
