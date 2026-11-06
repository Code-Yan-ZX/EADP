"""Execute real scoring main using only temporary synthetic files and fake scorers."""
import hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
def put(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))
def lines(path,rows):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class TestMain(unittest.TestCase):
 def test_actual_main_and_gate(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td)/'root';work=base/'e6/stacktok_full_analysis';work.mkdir(parents=True)
   for name in ('score.py','paired.py'):(work/name).write_text((D/name).read_text())
   (work/'admission.py').write_text("""import hashlib,json,os
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require():
 if os.getenv('TEST_REJECT')=='1':raise ValueError('synthetic gate rejects')
 return {'source_sha256':{}}
""")
   put(work/'score_plan001.json',{'synthetic':True});src=base/'e1_analysis';src.mkdir();(src/'completion.py').write_text('');(src/'metrics.py').write_text("def author_functions():return {}\ndef _author_constants():return {}\ndef score(task,pred,truth,author,constants,scoring_prediction=None):\n v=1. if pred=='A' else .5\n return {'official':v,'independent':v}\n");(src/'score.py').write_text("import os\nfrom pathlib import Path\ndef read_truths(sources,needed):\n Path(os.environ['TEST_TRUTH_MARKER']).write_text('opened')\n return {k:['synthetic'] for k in needed}\n")
   full=base/'e6/stacktok_full';full.mkdir(parents=True);(full/'full_refs.py').write_text("def e4_records(rows):return {(r['task'],str(r['qid'])):{'processor':r['processor'],'generation_scalars':{'do_sample':False},'eos_ids':[9]} for r in rows}\ndef validate_reference(r,ref):return True\n")
   sources={}
   for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
    f=base/('synthetic_'+task+'.txt');f.write_text('synthetic');sources[task]={'path':str(f),'sha256':sha(f)}
   summary=base/'e1/summary.json';put(summary,{'sources':sources});put(base/'e1/e1full001_completion.json',{'files':{'dataset_summary':{'path':str(summary),'sha256':sha(summary)}}})
   anchors=[];records=[];old=[]
   for i in range(384):
    task=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')[i//128];q=str(i);x={'task':task,'qid':q,'group_id':'g'+q,'message_sha256':'m','image_sha256':'i','processor':{'input_ids':'synthetic'}};anchors.append(x);records.append({**x,'path':'local_stacktok','budget':256,'generation_scalars':{'do_sample':False},'eos_ids':[9],'answer':'A','eos':True,'truncated':False})
    old.append({**x,'arms':{'e4_k256_d1':{'answer':'E','score':.25,'independent_score':.25,'eos':True,'truncated':False},'pixel_t0':{'answer':'P0','score':.5,'independent_score':.5,'eos':True,'truncated':False},'pixel_t045':{'answer':'P45','score':.75,'independent_score':.75,'eos':True,'truncated':False}}})
   put(base/'e6/pixelprune_t045_dev384/plan001.json',{'anchors':anchors});lines(full/'run_001/records.jsonl',records);lines(base/'e6/pixelprune_t045_dev384_analysis/score001/scores.jsonl',old)
   marker=base/'truth_marker';env=dict(os.environ,TEST_TRUTH_MARKER=str(marker),TEST_REJECT='1',PYTHONDONTWRITEBYTECODE='1');cmd=[sys.executable,str(work/'score.py'),'--out',str(base/'rejected')];fail=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=120);self.assertNotEqual(fail.returncode,0);self.assertFalse(marker.exists());self.assertFalse((base/'rejected').exists())
   env['TEST_REJECT']='0';run=subprocess.run([sys.executable,str(work/'score.py'),'--out',str(base/'result')],capture_output=True,text=True,env=env,timeout=120);self.assertEqual(run.returncode,0,run.stderr);self.assertTrue(marker.exists());result=json.loads((base/'result/analysis.json').read_text());out=[json.loads(x) for x in (base/'result/scores.jsonl').read_text().splitlines()];self.assertEqual(result['questions'],384);self.assertEqual(len(out),384)
   for t in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
    x=result['tasks'][t];self.assertEqual(x['n_questions'],128);self.assertEqual(x['bootstrap']['replicates'],10000);self.assertEqual(x['contrasts']['stacktok_minus_eadp']['difference_pp'],75.);self.assertEqual(x['contrasts']['stacktok_minus_pixel_t0']['difference_pp'],50.);self.assertEqual(x['contrasts']['stacktok_minus_pixel_t045']['difference_pp'],25.)
if __name__=='__main__':unittest.main(verbosity=2)
