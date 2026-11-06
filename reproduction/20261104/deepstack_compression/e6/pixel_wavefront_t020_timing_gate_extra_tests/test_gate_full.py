import importlib.util,json,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
B=Path(__file__).resolve().parents[2];G=B/'e6/pixel_wavefront_t020_timing_final_v4/gpu_gate.py'
spec=importlib.util.spec_from_file_location('gate_extra',G);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return p
class Tests(unittest.TestCase):
 def test_actual_admission_positive_and_tamper_rejection(self):
  with tempfile.TemporaryDirectory() as t:
   b=Path(t);d=b/'final';i=b/'independent';a=b/'e6/pixel_wavefront_t020_timing_accept_final';c=b/'e6/pixel_wavefront_t020_model_timing_v2';d.mkdir();i.mkdir();a.mkdir(parents=True);c.mkdir(parents=True)
   worker=d/'worker.py';worker.write_text('worker');formal_source=a/'formal.py';formal_source.write_text('formal')
   jobs=[{'order':n} for n in range(24)];design=put(c/'design001.json',{'jobs':jobs});source={str(worker.resolve()):sha(worker)};fs={str(formal_source.resolve()):sha(formal_source)}
   plan=put(d/'gpu_plan001.json',{'scope':'pixel_t020_isolated_process_ttft_24','jobs':jobs,'design_sha256':sha(design),'source_sha256':source,'weight_reference_sha256':'weight-sha','weight_files':['a','b','c','d']})
   put(a/'verification_plan001.json',{'gpu_plan_sha256':sha(plan),'source_sha256':fs});put(a/'cpu_proof001.json',{'status':'passed','GT_opened':False,'GPU_used':False,'predictions_opened':False,'source_sha256_before':fs,'source_sha256_after':fs})
   reviewer=i/'review_pixel_wavefront_t020_timing_final_v4_prelaunch.py';reviewer.write_text('reviewer')
   prereqs=['pixel_wavefront_t020_timing_runtime_v4_preflight_result_review.json','pixel_wavefront_t020_timing_run_v2_preflight_result_review.json','pixel_wavefront_t020_timing_final_v4_preflight_result_review.json']
   for p in prereqs:put(i/p,{'status':'passed'})
   proof=i/'pixel_wavefront_t020_timing_final_v4_prelaunch_review.json'
   q={'status':'passed','independent_review':True,'scope':'pixel_t020_timing_gpu_prelaunch','errors':[],'GT_opened':False,'scoring_performed':False,'source_sha256':source,'plan_sha256':sha(plan),'reviewer_sha256':sha(reviewer),'runtime_review_sha256':sha(i/prereqs[0]),'run_preflight_review_sha256':sha(i/prereqs[1]),'final_preflight_review_sha256':sha(i/prereqs[2]),'formal_plan_sha256':sha(a/'verification_plan001.json')};put(proof,q)
   weights={'worker_preflight_sha256':'weight-sha','expected':[{'path':x} for x in ['a','b','c','d']]}
   with patch.object(gate,'B',b),patch.object(gate,'D',d),patch.object(gate,'I',i),patch.object(gate,'prerequisites',return_value=None),patch.object(gate,'weight_preflight',return_value=weights),patch.object(gate,'terminal'):
    self.assertEqual(gate.admit()['jobs'],jobs)
    q['scope']='wrong';put(proof,q)
    with self.assertRaises(ValueError):gate.admit()
    q['scope']='pixel_t020_timing_gpu_prelaunch';put(proof,q);worker.write_text('tampered')
    with self.assertRaises(ValueError):gate.admit()
if __name__=='__main__':unittest.main()
