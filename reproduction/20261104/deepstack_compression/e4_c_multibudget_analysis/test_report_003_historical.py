import hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import report,metrics,watcher_contract
from test_watcher_contract import fixture,NS,T
from test_metrics import Tests as Fixtures
class Tests(unittest.TestCase):
 def test_actual_main_1520_synthetic_jobs(self):
  jobs,events=Fixtures().fixture();annotations=[{'order':i,**{k:j[k] for k in ('phase','task','budget','protocol')},'ttft_health_overlap_indices':[],'full_job_health_overlap_indices':[]} for i,j in enumerate(jobs)]
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);d=root/'analysis';d.mkdir();gatefile=d/'analysis_gate.py';gatefile.write_text('synthetic gate');journal=root/'e4_c_multibudget_worker_v3/run_001/journal.jsonl';journal.parent.mkdir(parents=True);journal.write_text(''.join(json.dumps(e)+'\n' for e in events));formal={'annotations':annotations,'health_windows':[[1,2]]}
   samples,health,manual=fixture();w=root/'e4_c_multibudget_environment';(w/'monitor_watch001').mkdir(parents=True)
   for path,items in ((w/'samples001.jsonl',samples),(w/'monitor_watch001/health.jsonl',health),(w/'manual_checks001.jsonl',manual)):path.write_text(''.join(json.dumps(x)+'\n' for x in items))
   def read(p):
    n=Path(p).name
    if n=='pilot_plan001.json':return {'jobs':jobs}
    if n=='launch.json':return {'time_namespace':NS}
    if n=='started.json':return {'created_utc':T}
    if n=='terminal.json':return {'utc':T}
    return formal
   fake=types.SimpleNamespace(admit=lambda:{'passed':True},read=read,sha=lambda p:'a'*64)
   def load(name,path):return {'_multibudget_analysis_gate':fake,'_multibudget_analysis_metrics':metrics,'_multibudget_analysis_watcher':watcher_contract}[name]
   with patch.object(report,'B',root),patch.object(report,'D',d),patch.object(report,'EXPECTED_GATE_SHA',hashlib.sha256(gatefile.read_bytes()).hexdigest()),patch.object(report,'load',side_effect=load),patch.object(metrics.np,'__version__','2.2.6'),patch.object(sys,'argv',['report.py','--out',str(d/'analysis003.json')]):report.main()
   r=json.loads((d/'analysis003.json').read_text());self.assertEqual((r['images'],r['measured']),(32,1280));self.assertTrue(r['predictions_opened']);self.assertEqual(r['health_overlap']['DocVQA_VAL/natural/256']['measured_jobs'],55);self.assertEqual(len(r['jobs_detail']),1520);self.assertEqual(r['jobs_detail'][0]['token_count'],1)
 def test_refused_gate_never_reads_journal(self):
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);f=d/'analysis_gate.py';f.write_text('gate');fake=types.SimpleNamespace(admit=lambda:(_ for _ in ()).throw(ValueError('blocked')))
   with patch.object(report,'D',d),patch.object(report,'EXPECTED_GATE_SHA',hashlib.sha256(f.read_bytes()).hexdigest()),patch.object(report,'load',return_value=fake),patch.object(sys,'argv',['report.py','--out',str(d/'analysis003.json')]):
    with self.assertRaisesRegex(ValueError,'blocked'):report.main()
   self.assertFalse((d/'analysis003.json').exists())
if __name__=='__main__':unittest.main()
