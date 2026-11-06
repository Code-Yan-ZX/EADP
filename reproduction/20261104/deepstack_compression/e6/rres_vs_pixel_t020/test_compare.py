import os,subprocess,sys,unittest
from pathlib import Path
import compare
class Tests(unittest.TestCase):
 def test_question_weighted_groups_and_direction(self):
  rows=[dict(group_id='a',rres=1.,pixel=0.,rres_eos=True,pixel_eos=True,rres_truncated=False,pixel_truncated=False),dict(group_id='a',rres=0.,pixel=0.,rres_eos=True,pixel_eos=True,rres_truncated=False,pixel_truncated=False),dict(group_id='b',rres=0.,pixel=1.,rres_eos=True,pixel_eos=True,rres_truncated=False,pixel_truncated=False)]
  a=compare.summarize(rows);self.assertEqual(a,compare.summarize(rows));self.assertEqual((a['questions'],a['image_groups']),(3,2));self.assertAlmostEqual(a['difference_pp'],0);self.assertEqual((a['improved'],a['decreased'],a['unchanged']),(1,1,1))
 def test_exact_controls_and_scores(self):
  x={'generation_kwargs':{**compare.CONTROLS,'num_beams':1,'use_cache':True},'eos_token_ids':[151645,151643]};y={'generation_scalars':dict(compare.CONTROLS),'eos_ids':[151645,151643]};compare.controls(x,y)
  for bad in ({}, {'do_sample':False}):
   with self.assertRaises(ValueError):compare.controls(x,{'generation_scalars':bad,'eos_ids':[151645,151643]})
  with self.assertRaises(ValueError):compare.check_score({'score':.3,'independent_score':.4})
 def test_deadline_rejects_before_plan_or_scores(self):
  from unittest.mock import patch
  class Expired:
   @staticmethod
   def now(_):return compare.DEADLINE
  with patch.object(compare,'datetime',Expired),patch.object(compare,'read',side_effect=AssertionError('premature read')):
   with self.assertRaisesRegex(ValueError,'deadline'):compare.main()
 def test_bad_prelaunch_rejected(self):
  from unittest.mock import patch
  q={'status':'failed','independent_review':True,'GT_opened':False,'scores_parsed':False,'errors':[]}
  with patch.object(compare,'D',Path('/tmp')),patch.object(compare,'terminal'),patch.object(compare,'read',side_effect=[q,{'status':'passed'},{'cwd':'/tmp','worker_pid':7,'command':['/tmp/cpu_check.py']},{'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':7}]),patch.object(compare,'sha',return_value='test'):
   with self.assertRaisesRegex(ValueError,'prelaunch proof'):compare.require_prelaunch({'source_sha256':{}})
 def test_prelaunch_supervisor_wrong_output_rejected(self):
  import json,tempfile
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);mon=base/'monitor';mon.mkdir();reviewer=base/'review.py';reviewer.write_text('');good=base/'proof.json';start={'cwd':str(base),'worker_pid':10,'command':['python',str(reviewer),'--out',str(good)]};end={'worker_pid':10,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
   (mon/'terminal.json').write_text(json.dumps(end));(mon/'started.json').write_text(json.dumps(start))
   with patch.object(compare,'I',base):
    compare.terminal('monitor',reviewer,good);start['command'][-1]=str(base/'wrong.json');(mon/'started.json').write_text(json.dumps(start))
    with self.assertRaises(ValueError):compare.terminal('monitor',reviewer,good)
 def test_actual_cli_missing_prelaunch_cannot_parse_scores(self):
  code='''import os,runpy,sys\nfrom pathlib import Path\nbanned=[str(Path(x).resolve()) for x in sys.argv[3:]]\ndef audit(event,args):\n if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):\n  p=str(Path(os.fsdecode(args[0])).resolve())\n  if p in banned: raise RuntimeError('SCORE_SENTINEL')\nsys.addaudithook(audit)\nsys.argv=[sys.argv[1],'--out',sys.argv[2]]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n'''
  # Scores can be opened for source hashing before the gate; only parsing must be prohibited.
  # Here the score files are SHA-checked first, so the audit hook cannot distinguish reads;
  # instead patch JSONL parser in a separate real-main process to reject premature parsing.
  code='''import importlib.util,sys\nfrom pathlib import Path\np=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('_cmp_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\ndef forbidden(*a,**k):raise RuntimeError('SCORE_PARSE_SENTINEL')\nm.lines=forbidden\nsys.argv=[str(p),'--out',sys.argv[2]]\nm.main()\n'''
  out=compare.D/'result001.json';p=subprocess.run([sys.executable,'-c',code,str(compare.D/'compare.py'),str(out)],capture_output=True,text=True,timeout=120)
  self.assertNotEqual(p.returncode,0);self.assertNotIn('SCORE_PARSE_SENTINEL',p.stderr);self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
