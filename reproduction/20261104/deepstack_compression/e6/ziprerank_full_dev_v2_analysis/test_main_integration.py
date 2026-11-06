"""Run the actual scorer CLI on 384 synthetic inputs and a synthetic gate."""
import hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))

class MainIntegration(unittest.TestCase):
 def test_real_main_synthetic_and_gate_failure(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td)/'root';work=base/'e6/ziprerank_full_dev_v2_analysis';work.mkdir(parents=True)
   fake_admission='''import hashlib,json,os\nfrom pathlib import Path\nHERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];PLAN=BASE/'e6/ziprerank_full_dev_v2_accept/score_plan001.json'\ndef sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()\ndef read(p):return json.loads(Path(p).read_text())\ndef admission():\n if os.getenv('TEST_REJECT')=='1':raise ValueError('synthetic gate denied')\n return {'source_sha256':{}}\n'''
   (work/'admission.py').write_text(fake_admission);source=(HERE/'score.py').read_text().replace("ADMISSION_SHA='"+digest(HERE/'admission.py')+"'", "ADMISSION_SHA='"+digest(work/'admission.py')+"'")
   (work/'score.py').write_text(source);(work/'paired.py').write_text((HERE/'paired.py').read_text())
   e1src=base/'e1_analysis';e1src.mkdir();(e1src/'completion.py').write_text('')
   (e1src/'metrics.py').write_text("def author_functions():return {}\ndef _author_constants():return {}\ndef score(task,pred,truth,author,constants,scoring_prediction=None):\n v=1.0 if pred=='A' else 0.5\n return {'official':v,'independent':v}\n")
   (e1src/'score.py').write_text("import os\nfrom pathlib import Path\ndef read_truths(sources,needed):\n Path(os.environ['TEST_TRUTH_MARKER']).write_text('opened')\n return {k:['synthetic'] for k in needed}\n")
   put(base/'e6/ziprerank_full_dev_v2_accept/score_plan001.json',{'denominator':384})
   sources={}
   for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
    path=base/('synthetic_'+task+'.txt');path.write_text('synthetic only');sources[task]={'path':str(path),'sha256':digest(path)}
   summary=base/'e1/summary.json';put(summary,{'sources':sources});put(base/'e1/e1full001_completion.json',{'files':{'dataset_summary':{'path':str(summary),'sha256':digest(summary)}}})
   scalars={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0};anchors=[];e1=[];es=[]
   obs=base/'e6/ziprerank_full_dev_v2/run_smoke001/observations'
   for i in range(384):
    task=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')[i//128];qid=str(i);group='g'+qid;a={'task':task,'qid':qid,'group_id':group,'message_sha256':'m','image_sha256':'i'};anchors.append(a);prefix=task+'_'+qid;processor={'input_ids':'synthetic'}
    put(obs/(prefix+'_processor.json'),{'processor':processor})
    for name,answer in (('author_prefix_256','A'),('user_text_256','B')):
     z={k:a[k] for k in ('task','qid','group_id')};z.update(arm=name,status='completed',local_rope_adapter=True,full_user_text_adapter=(name=='user_text_256'),generation_controls={**scalars,'eos_token_id':[9]},answer=answer,eos=True,truncated=False);put(obs/(prefix+'_'+name+'.json'),z)
    e={**a,'arm':'P3-D1','forced':False,'processor':processor,'generation_scalars':scalars,'eos_ids':[9],'answer':'base','eos':True};e1.append(e)
    es.append({'task':task,'qid':qid,'group_id':group,'arms':{'P3-D1':{'raw_answer':'base','score':.25,'independent_score':.25,'eos':True,'truncated':False}}})
   put(base/'e6/ziprerank_full_dev_v2/plan001.json',{'anchors':anchors})
   for path,rows in ((base/'e1/e1full001/raw_predictions.jsonl',e1),(base/'e1/e1full001_score/scores.jsonl',es)):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
   marker=base/'truth_marker';env=dict(os.environ,TEST_TRUTH_MARKER=str(marker),TEST_REJECT='1',PYTHONDONTWRITEBYTECODE='1');cmd=[sys.executable,str(work/'score.py'),'--out',str(base/'rejected')]
   fail=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=120);self.assertNotEqual(fail.returncode,0);self.assertFalse(marker.exists());self.assertFalse((base/'rejected').exists())
   env['TEST_REJECT']='0';ok=subprocess.run([sys.executable,str(work/'score.py'),'--out',str(base/'result')],capture_output=True,text=True,env=env,timeout=120)
   self.assertEqual(ok.returncode,0,ok.stderr);self.assertTrue(marker.exists());report=json.loads((base/'result/analysis.json').read_text());self.assertEqual(report['questions'],384);self.assertEqual(report['zip_generations'],768)
   for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
    r=report['tasks'][task];self.assertEqual(r['n_questions'],128);self.assertEqual(r['contrasts']['prefix_minus_eadp']['difference_pp'],75.0);self.assertEqual(r['contrasts']['user_text_minus_prefix']['difference_pp'],-50.0)
if __name__=='__main__':unittest.main(verbosity=2)
