"""All 5940 accepted timing jobs, frozen DEV clustered analysis."""
import argparse,hashlib,importlib.metadata,importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
GATE_SHA256='0301124ca7b7e67c9542b700fdb9bcc19a55a67c2b982b604b794d1f3533de15'
def load_exact(name,path,digest):
 path=Path(path).resolve()
 if hashlib.sha256(path.read_bytes()).hexdigest()!=digest or name in sys.modules:raise ValueError('analysis source changed or import collision')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 if Path(m.__file__).resolve()!=path:raise ValueError('analysis module path mismatch')
 return m
gate=load_exact('eadp_latency_analysis_gate',HERE/'analysis_gate.py',GATE_SHA256)
def read(path):return json.loads(Path(path).read_text())
def lines(path):
 with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 approval=gate.admit() # No full timing values decoded before independent prelaunch.
 core=load_exact('eadp_latency_analysis_core',HERE/'analysis_core.py',approval['prelaunch']['source_sha256'][str((HERE/'analysis_core.py').resolve())])
 full=BASE/'e4_c_latency_full';inputs=lines(approval['upstream']['inputs_path']);frozen=read(full/'full_001/frozen.json');events=lines(full/'full_001/journal.jsonl');formal=lines(full/'full_001_annotations.jsonl');manual=lines(HERE/'manual_overlap001/annotations.jsonl');health=lines(full/'monitor_full001/health.jsonl');manual_windows=lines(full/'manual_health_windows.jsonl')
 windows=core.validate_windows(health,manual_windows,approval['upstream']['namespace']);records=core.build_records(frozen['schedule'],inputs,events,formal,manual,approval['upstream']['namespace'],windows);tasks=core.analyze(records);per_window=core.summarize_windows(records,windows)
 windows={phase:{'n':sum(x['phase']==phase for x in records),'supervisor':sum(x['phase']==phase and x['monitor_overlap'] for x in records),'manual_health':sum(x['phase']==phase and x['manual_health_overlap'] for x in records),'manual_nvml':sum(x['phase']==phase and x['manual_nvml_overlap'] for x in records)} for phase in ('warmup','measure')}
 if gate.admit()!=approval:raise ValueError('analysis evidence changed')
 out=a.out.resolve()
 if out!=(HERE/'analysis001').resolve():raise ValueError('unexpected output directory')
 out.mkdir(exist_ok=False)
 with (out/'records.jsonl').open('x') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 result={'status':'completed','n_jobs':len(records),'tasks':tasks,'interference':windows,'interference_by_window':per_window,'all_samples_retained':True,'n_warmup':180,'n_measured':5760,
         'plan_sha256':gate.PLAN_SHA,'supplement_sha256':gate.SUP_SHA,'approval':approval,'records_sha256':gate.sha(out/'records.jsonl'),'numpy_version':importlib.metadata.version('numpy'),
         'scope':'exploratory batch1 warm-cache DEV; no GT or quality score','limitations':read(HERE/'analysis_plan001.json')['reporting']+['Image bootstrap conditions on this run and its observed repeats; it does not separately estimate within-image noise, temporal drift or between-run variation.','Full-run analysis plan was frozen during the run before viewing full-run values; earlier pilot results were known.','Other short stat/ps, source reads, notes and remote queries lacked exact time boundaries and cannot be assigned to individual jobs.']}
 with (out/'analysis.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'completed','n_jobs':len(records),'n_measured':5760}))
if __name__=='__main__':main()
