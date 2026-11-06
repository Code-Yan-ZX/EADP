"""Run actual score.py main on 384 wholly synthetic questions; no real data."""
import hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
def put(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))
def lines(path,rows):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class FullMainTest(unittest.TestCase):
 def test_actual_main_full_denominator_and_gate(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td)/'root';work=base/'e6/pixelprune_t045_dev384_analysis';work.mkdir(parents=True)
   for f in ('score.py','paired.py'):(work/f).write_text((HERE/f).read_text())
   (work/'admission.py').write_text("""import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];PLAN=HERE/'score_plan001.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def admission():
 if os.getenv('TEST_REJECT')=='1':raise ValueError('synthetic gate rejects')
 return {'source_sha256':{}}
""")
   put(work/'score_plan001.json',{'synthetic':True});e1src=base/'e1_analysis';e1src.mkdir();(e1src/'completion.py').write_text('')
   (e1src/'metrics.py').write_text("def author_functions():return {}\ndef _author_constants():return {}\ndef score(task,pred,truth,author,constants,scoring_prediction=None):\n v=1. if pred=='A' else .5\n return {'official':v,'independent':v}\n")
   (e1src/'score.py').write_text("import os\nfrom pathlib import Path\ndef read_truths(sources,needed):\n Path(os.environ['TEST_TRUTH_MARKER']).write_text('opened')\n return {k:['synthetic'] for k in needed}\n")
   sources={}
   for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
    path=base/('synthetic_'+task+'.txt');path.write_text('synthetic');sources[task]={'path':str(path),'sha256':sha(path)}
   summary=base/'e1/summary.json';put(summary,{'sources':sources});put(base/'e1/e1full001_completion.json',{'files':{'dataset_summary':{'path':str(summary),'sha256':sha(summary)}}})
   scalars={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.};config={'PIXELPRUNE_ENABLED':'true','PIXELPRUNE_METHOD':'pred_2d','PIXELPRUNE_METRIC':'max','PIXELPRUNE_THRESHOLD':'0.45','PIXELPRUNE_ANCHORED':'true','PIXELPRUNE_VIT_LAYER':'0'}
   anchors=[];e1=[];es=[];e4=[];bs=[];old_scores=[]
   for i in range(384):
    task=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')[i//128];qid=str(i);group='g'+qid;processor={'input_ids':'synthetic'};a={'task':task,'qid':qid,'group_id':group,'message_sha256':'m','image_sha256':'i','processor':processor};anchors.append(a)
    for directory,answer in [('pixelprune_t045_dev384','A'),('pixelprune_dev384','B')]:
     row={**a,'processor_identity':{'processor':processor},'generation_kwargs':{**scalars,'num_beams':1,'use_cache':True},'eos_token_ids':[9],'decoded':answer,'ends_eos':True,'selected_count':256,'pixelprune_config':config};put(base/('e6/'+directory+'/run_smoke001/observations')/(task+'_'+qid+'.json'),row)
    e={k:v for k,v in a.items() if k!='image_sha256'};e.update(arm='P3-D1',forced=False,generation_scalars=scalars,eos_ids=[9],answer='E',eos=True);e1.append(e)
    es.append({**a,'arms':{'P3-D1':{'raw_answer':'E','score':.25,'independent_score':.25,'eos':True,'truncated':False}}})
    b_arms={}
    for budget in ('256','1024'):
     b={**a,'forced':False,'ds_on':True,'budget':budget,'generation_scalars':scalars,'eos_ids':[9],'answer':'C'+budget,'eos':True};e4.append(b);b_arms['K'+budget+'-D1']={'answer':'C'+budget,'score':.25,'independent_score':.25,'eos':True,'truncated':False}
    bs.append({**a,'arms':b_arms});old_scores.append({**a,'arms':{'pixel_dynamic':{'answer':'B','score':.5,'independent_score':.5,'eos':True}}})
   put(base/'e6/pixelprune_t045_dev384/plan001.json',{'anchors':anchors});lines(base/'e1/e1full001/raw_predictions.jsonl',e1);lines(base/'e1/e1full001_score/scores.jsonl',es);lines(base/'e4_panel_v2/panel_003/records.jsonl',e4);lines(base/'e4_analysis/score_panel003/scores.jsonl',bs);lines(base/'e6/pixelprune_dev384_analysis/score001/scores.jsonl',old_scores)
   marker=base/'truth_marker';env=dict(os.environ,TEST_TRUTH_MARKER=str(marker),TEST_REJECT='1',PYTHONDONTWRITEBYTECODE='1');cmd=[sys.executable,str(work/'score.py'),'--out',str(base/'rejected')]
   failed=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=120);self.assertNotEqual(failed.returncode,0);self.assertFalse(marker.exists());self.assertFalse((base/'rejected').exists())
   env['TEST_REJECT']='0';run=subprocess.run([sys.executable,str(work/'score.py'),'--out',str(base/'result')],capture_output=True,text=True,env=env,timeout=120);self.assertEqual(run.returncode,0,run.stderr);self.assertTrue(marker.exists())
   result=json.loads((base/'result/analysis.json').read_text());records=[json.loads(x) for x in (base/'result/scores.jsonl').read_text().splitlines()];self.assertEqual(result['questions'],384);self.assertEqual(len(records),384)
   for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
    q=result['tasks'][task];self.assertEqual(q['n_questions'],128);self.assertEqual(q['bootstrap']['replicates'],10000);self.assertEqual(q['contrasts']['t045_minus_t0']['difference_pp'],50.);self.assertEqual(q['contrasts']['t045_minus_e4_k256']['difference_pp'],75.);self.assertEqual(q['contrasts']['t045_minus_e4_k1024']['difference_pp'],75.)
if __name__=='__main__':unittest.main(verbosity=2)
