"""Freeze independent static and synthetic review; never opens audit output."""
import hashlib,json,os
from pathlib import Path
from independent_recalc import supervised,digest,ROOT,R,D
H=Path(__file__).resolve().parent
expected='40a1b865096a625b48c1d1abe0aa32ac88290de2ea86b7717cc59faf81772bd5'
def main():
 code=D/'document_overlap_analysis_v2.py'
 if digest(code)!=expected:raise ValueError('v2 changed after static review')
 tests=[('monitor','test_original.py','Ran 4 tests'),('monitor_recalc_cpu002','test_independent_recalc.py','Ran 4 tests'),('monitor_v2_cpu001','test_v2_independent.py','Ran 3 tests')]
 for mon,script,pattern in tests:
  d=H/mon;supervised(d,H/script,[])
  log=(d/'stderr.log').read_text()
  if pattern not in log or not log.rstrip().endswith('OK'):raise ValueError('synthetic test failed')
 files=[code,H/'original.py',H/'test_original.py',H/'independent_recalc.py',H/'test_independent_recalc.py',H/'test_v2_independent.py',Path(__file__)]
 proof={'status':'passed','independent_review':True,'scope':'static and synthetic review of document overlap v2 only; real data remains gated by complete audit and independent audit; not a training approval','source_sha256':{str(f):digest(f) for f in files},'synthetic_monitors':{mon:{k:digest(H/mon/k) for k in ('started.json','terminal.json','stderr.log')} for mon,_,_ in tests},'GT_opened':False,'predictions_opened':False,'training_performed':False}
 with (R/'document_overlap_v2_preflight_review.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
