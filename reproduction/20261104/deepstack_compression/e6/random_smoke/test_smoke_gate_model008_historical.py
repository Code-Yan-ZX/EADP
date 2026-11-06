import unittest
import re
import hashlib
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import smoke_gate
class Tests(unittest.TestCase):
 def test_missing_plan_refuses_both_cpu_and_gpu(self):
  with patch.object(smoke_gate,'D',smoke_gate.D/'absent_dir'):
   for preflight in (True,False):
    with self.subTest(preflight=preflight),self.assertRaisesRegex(ValueError,'plan missing'):smoke_gate.admit(preflight)
 def test_worker_and_gate_bind_same_plan(self):
  d=Path(__file__).resolve().parent
  gate=set(re.findall(r'execution_plan[0-9]+\.json',(d/'smoke_gate.py').read_text()))
  worker=set(re.findall(r'execution_plan[0-9]+\.json',(d/'model_worker.py').read_text()))
  self.assertEqual(len(gate),1);self.assertEqual(gate,worker)
 def test_worker_and_launcher_pin_actual_gate(self):
  d=Path(__file__).resolve().parent;expected=hashlib.sha256((d/'smoke_gate.py').read_bytes()).hexdigest()
  for name in ('model_worker.py','model_launch.py'):
   found=re.findall(r"hexdigest\(\)!=\'([0-9a-f]{64})\'",(d/name).read_text())
   self.assertEqual(found,[expected])
 def test_cpu_supervisor_exit_and_command(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);mon=d/'monitor_model_cpu008';mon.mkdir();source=d/'source.py';source.write_text('source');cpu=d/'model_cpu_proof008.json';freeze=d/'freeze_model_execution.py';freeze.write_text('freeze')
   digest=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
   sources={str(source.resolve()):digest(source)};proof={'status':'passed','test_returncode':0,'model_loaded':False,'GPU_authorized':False,'GT_opened':False,'predictions_opened':True,'source_sha256_before':sources,'source_sha256_after':sources};cpu.write_text(json.dumps(proof));plan={'source_sha256':{**sources,str(cpu.resolve()):digest(cpu)}}
   start={'cwd':str(d),'command':['python',str(freeze)],'worker_pid':100};terminal={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':100};(mon/'started.json').write_text(json.dumps(start));(mon/'terminal.json').write_text(json.dumps(terminal))
   with patch.object(smoke_gate,'D',d),patch.object(smoke_gate,'B',d):
    self.assertIn('cpu_started_sha256',smoke_gate.cpu(plan))
    for change in ({'reason':'running'},{'worker_returncode':1},{'remaining_process_group':[1]},{'worker_pid':101}):
     (mon/'terminal.json').write_text(json.dumps({**terminal,**change}))
     with self.assertRaises(ValueError):smoke_gate.cpu(plan)
    (mon/'terminal.json').write_text(json.dumps(terminal));(mon/'started.json').write_text(json.dumps({**start,'command':['python','foreign.py']}))
    with self.assertRaises(ValueError):smoke_gate.cpu(plan)
    (mon/'started.json').write_text(json.dumps(start));source.write_text('changed');bad=dict(plan['source_sha256']);bad[str(source.resolve())]=digest(source)
    with self.assertRaises(ValueError):smoke_gate.cpu({'source_sha256':bad})
 def test_independent_review_must_bind_cpu_supervisor_hashes(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);r=d/'review';r.mkdir();proof=r/'random_model_preflight_review.json';reviewer=r/'review_random_model_preflight.py';reviewer.write_text('reviewer');(d/'execution_plan008.json').write_text('plan')
   digest=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();source={'x':'y'};fields={'cpu_proof_sha256':'a','cpu_started_sha256':'b','cpu_terminal_sha256':'c'}
   data={'status':'passed','independent_review':True,'scope':'noGPU preflight only','errors':[],'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'plan_sha256':digest(d/'execution_plan008.json'),'reviewer_sha256':digest(reviewer),'source_sha256':source,**fields};proof.write_text(json.dumps(data))
   with patch.object(smoke_gate,'D',d),patch.object(smoke_gate,'R',r),patch.object(smoke_gate,'plan',return_value={'source_sha256':source}),patch.object(smoke_gate,'cpu',return_value=fields),patch.object(smoke_gate,'terminal'):
    self.assertEqual(smoke_gate.admit(True)['plan_sha256'],digest(d/'execution_plan008.json'))
    for k in fields:
     proof.write_text(json.dumps({**data,k:'tampered'}))
     with self.assertRaises(ValueError):smoke_gate.admit(True)
if __name__=='__main__':unittest.main()
