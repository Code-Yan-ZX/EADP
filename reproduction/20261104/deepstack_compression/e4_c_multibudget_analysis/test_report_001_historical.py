import hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import report,metrics
from test_metrics import Tests as Fixtures
class Tests(unittest.TestCase):
 def test_actual_main_1520_synthetic_jobs(self):
  jobs,events=Fixtures().fixture();annotations=[{'order':i,**{k:j[k] for k in ('phase','task','budget','protocol')},'ttft_health_overlap_indices':([0] if i==0 else []),'full_job_health_overlap_indices':([0] if i==0 else [])} for i,j in enumerate(jobs)]
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);d=root/'analysis';d.mkdir();gatefile=d/'analysis_gate.py';gatefile.write_text('synthetic gate');journal=root/'e4_c_multibudget_worker_v3/run_001/journal.jsonl';journal.parent.mkdir(parents=True);journal.write_text(''.join(json.dumps(e)+'\n' for e in events));formal={'annotations':annotations,'health_windows':[[1,2]]}
   fake=types.SimpleNamespace(admit=lambda:{'passed':True},read=lambda p: {'jobs':jobs} if Path(p).name=='pilot_plan001.json' else formal,sha=lambda p:'a'*64)
   def load(name,path):return fake if name=='_multibudget_analysis_gate' else metrics
   with patch.object(report,'B',root),patch.object(report,'D',d),patch.object(report,'EXPECTED_GATE_SHA',hashlib.sha256(gatefile.read_bytes()).hexdigest()),patch.object(report,'load',side_effect=load),patch.object(metrics.np,'__version__','2.2.6'),patch.object(sys,'argv',['report.py','--out',str(d/'analysis001.json')]):report.main()
   r=json.loads((d/'analysis001.json').read_text());self.assertEqual((r['images'],r['measured']),(32,1280));self.assertTrue(r['predictions_opened']);self.assertEqual(r['health_overlap']['DocVQA_VAL/natural/256']['measured_jobs'],55)
 def test_refused_gate_never_reads_journal(self):
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);f=d/'analysis_gate.py';f.write_text('gate');fake=types.SimpleNamespace(admit=lambda:(_ for _ in ()).throw(ValueError('blocked')))
   with patch.object(report,'D',d),patch.object(report,'EXPECTED_GATE_SHA',hashlib.sha256(f.read_bytes()).hexdigest()),patch.object(report,'load',return_value=fake),patch.object(sys,'argv',['report.py','--out',str(d/'analysis001.json')]):
    with self.assertRaisesRegex(ValueError,'blocked'):report.main()
   self.assertFalse((d/'analysis001.json').exists())
if __name__=='__main__':unittest.main()
