import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import admission,synthetic_pipeline as p
class Tests(unittest.TestCase):
 def test_fail_closed(self):
  with patch.object(admission,'read',side_effect=FileNotFoundError):
   with self.assertRaises(FileNotFoundError):admission.require_identity()
  with self.assertRaises(RuntimeError):admission.authorize_gpu()
 def test_24_samples_and_failed_sample_retained(self):
  anchors=[dict(task=str(i),qid=str(i),group_id=str(i),image_sha256='i',message_sha256='m') for i in range(3)];ref={(a['task'],arm):dict(processor={'i':1},selected_indices=[1],output_ids=[2],eos_ids=[2],generation_scalars={'do_sample':False}) for a in anchors for arm in ('author','wavefront')}
  def mutate(job,row):
   if job['order']==5:row['selected_indices']=[9]
   return row
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'fake';r=p.run_fake(anchors,ref,lambda j:(1,2,3),out,mutate);self.assertEqual(len(r),24);self.assertEqual(sum(x['status']=='failed' for x in r),1);self.assertEqual(len(out.joinpath('journal.jsonl').read_text().splitlines()),24)
if __name__=='__main__':unittest.main()
