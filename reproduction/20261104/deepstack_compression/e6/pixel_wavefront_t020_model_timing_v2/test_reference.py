import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import reference
class Tests(unittest.TestCase):
 def test_six_records_and_tamper(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);run=d/'run';a=d/'a';i=d/'i';run.mkdir();a.mkdir();i.mkdir();jobs=[];files={}
   for n in range(6):
    job={'order':n,'task':'t'+str(n//2),'qid':str(n//2),'arm':('author' if n%2==0 else 'wavefront'),'group_id':'g'+str(n//2),'image_sha256':'x','message_sha256':'y'};jobs.append(job);p=run/'run_001'/f'arm_{n:03d}'/'observations'/'record.json';p.parent.mkdir(parents=True);rec={**{k:job[k] for k in ('task','qid','group_id','image_sha256','message_sha256')},'path':job['arm'],'status':'completed','GT_opened':False,'scoring_performed':False,'processor':{},'selected_indices':[1],'output_ids':[2],'eos_ids':[2],'generation_scalars':{}};p.write_text(json.dumps(rec));files[str(p.resolve())]=reference.admission.sha(p)
   (run/'execution_plan001.json').write_text(json.dumps({'jobs':jobs}));(a/'verification001.json').write_text(json.dumps({'files_sha256':files}));(i/'pixel_wavefront_t020_run_v2_result_review.json').write_text(json.dumps({'files_sha256':files}))
   with patch.object(reference,'RUN',run),patch.object(reference,'A',a),patch.object(reference,'I',i),patch.object(reference.admission,'require_identity',return_value={'status':'fake'}):
    refs,_=reference.load();self.assertEqual(len(refs),6);p.write_text('{}')
    with self.assertRaises(ValueError):reference.load()
if __name__=='__main__':unittest.main()
