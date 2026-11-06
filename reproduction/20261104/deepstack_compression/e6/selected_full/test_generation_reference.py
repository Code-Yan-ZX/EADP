import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from full_imports import HERE,load
ref=load('eadp_selected_full_generation_reference',HERE/'generation_reference.py')
class ReferenceTests(unittest.TestCase):
 def data(self,root):
  smoke=Path(root);run=smoke/'smoke_001';run.mkdir();rows=[]
  for method in ('div','cd'):
   for task in ref.TASKS:
    for path in ('author_stock','adapted'):
     rows.append({'method':method,'task':task,'path':path,'forced':False,'generation_controls':{'do_sample':False,'num_beams':1}})
  records=run/'records.jsonl';records.write_text(''.join(json.dumps(x)+'\n' for x in rows))
  proof=smoke/'smoke_001_verification.json';proof.write_text(json.dumps({'status':'passed','files_sha256':{str(records.resolve()):ref.sha(records)}}))
  return smoke,records,proof,rows
 def test_accepted_six_method_task_controls_and_changed_file(self):
  with tempfile.TemporaryDirectory() as d:
   smoke,records,proof,rows=self.data(d)
   with patch.object(ref,'SMOKE',smoke):
    out=ref.controls();self.assertEqual(len(out),2);self.assertEqual(len(out['div']),3)
    records.write_text(records.read_text()+'{}\n')
    with self.assertRaisesRegex(ValueError,'changed'):ref.controls()
 def test_two_paths_cannot_agree_only_after_changing_the_reference(self):
  with tempfile.TemporaryDirectory() as d:
   smoke,records,proof,rows=self.data(d);rows[1]['generation_controls']['num_beams']=2
   records.write_text(''.join(json.dumps(x)+'\n' for x in rows));proof.write_text(json.dumps({'status':'passed','files_sha256':{str(records.resolve()):ref.sha(records)}}))
   with patch.object(ref,'SMOKE',smoke):
    with self.assertRaisesRegex(ValueError,'differ'):ref.controls()
if __name__=='__main__':unittest.main()
