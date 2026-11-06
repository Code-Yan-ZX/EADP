"""Full synthetic exercise of the actual 384-question scoring entry point."""
import hashlib, importlib.util, json, shutil, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_rres_synthetic_driver',HERE/'score.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj)+'\n')
def jsonl(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
class Gate:
 read=staticmethod(lambda p:json.loads(Path(p).read_text()))
 sha=staticmethod(sha)
 @staticmethod
 def require():return {'source_sha256':{},'plan_sha256':'synthetic'}
class MainTest(unittest.TestCase):
 def test_full_main(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);d=b/'e6/rres_k256_full_analysis';d.mkdir(parents=True);shutil.copy(HERE/'paired.py',d/'paired.py')
   e=b/'e1_analysis';e.mkdir();(e/'completion.py').write_text('')
   (e/'metrics.py').write_text("def author_functions(): return {}\ndef _author_constants(): return {}\ndef score(*a,**k): return {'official':.75,'independent':.75}\n")
   (e/'score.py').write_text('def read_truths(sources,needed): return {k:["fake"] for k in needed}\n')
   anchors=[];e1=[];eadp=[];rr=[];rs=[];controls={'top_p':.001};high={'input_ids':{'shape':[1,100]}};low={'input_ids':{'shape':[1,260]},'pixel_values':{'shape':[1024,1536]},'image_grid_thw':{'shape':[1,3]}}
   for task in driver.TASKS:
    for i in range(128):
     a={'task':task,'qid':str(i),'group_id':f'{task}-{i}','image_sha256':'image','message_sha256':'message','processor':high};anchors.append(a)
     e1.extend([{**a,'arm':arm,'forced':False,'answer':'old','eos':True,'generation_scalars':controls,'eos_ids':[42]} for arm in ('P3-D1','P1-D1')])
     old={'raw_answer':'old','eos':True,'score':.5,'independent_score':.5};eadp.append({'task':task,'qid':str(i),'group_id':a['group_id'],'arms':{'P3-D1':old,'P1-D1':old}})
     target={**a,'processor':low,'resolution':512,'visual_tokens':256,'generation_scalars':controls,'eos_ids':[42],'hooks_and_methods_restored':True,'answer':'fabricated','eos':True,'truncated':False}
     write(b/f'e6/rres_k256_dev384/run001/observations/{task}_{i}.json',target)
     arms={'eadp':old}
     for s in driver.SEEDS:
      rr.append({**a,'seed':s,'answer':'rnd','eos':True,'truncated':False,'generation_scalars':controls,'eos_ids':[42],'budget':256,'ds_on':True,'protocol':'natural','forced':False})
      arms[str(s)]={'score':.4,'independent_score':.4,'answer':'rnd','eos':True,'truncated':False}
     rs.append({'task':task,'qid':str(i),'group_id':a['group_id'],'arms':arms})
   write(b/'e6/pixelprune_t020_dev384/plan001.json',{'anchors':anchors});jsonl(b/'e1/e1full001/raw_predictions.jsonl',e1);jsonl(b/'e1/e1full001_score/scores.jsonl',eadp)
   jsonl(b/'e6/random_full_exec_v2/run_001/records.jsonl',rr);jsonl(b/'e6/random_full_analysis_v3/score001/scores.jsonl',rs)
   sources={}
   for task in driver.TASKS:
    p=b/(task+'.tsv');p.write_text('synthetic only\n');sources[task]={'path':str(p),'sha256':sha(p)}
   summary=b/'summary.json';write(summary,{'sources':sources});write(b/'e1/e1full001_completion.json',{'files':{'dataset_summary':{'path':str(summary),'sha256':sha(summary)}}})
   saved_path=list(sys.path);names=('paired','completion','metrics','score');saved={n:sys.modules.pop(n,None) for n in names}
   try:
    with patch.object(driver,'B',b),patch.object(driver,'D',d),patch.object(driver,'load_gate',return_value=Gate()),patch.object(sys,'argv',['score.py','--out',str(d/'score001')]):driver.main()
    result=json.loads((d/'score001/analysis.json').read_text());self.assertEqual(result['questions'],384)
    self.assertEqual(len((d/'score001/scores.jsonl').read_text().splitlines()),384)
    for task in driver.TASKS:
     v=result['tasks'][task];self.assertEqual(v['questions'],128);self.assertAlmostEqual(v['rres_minus']['eadp']['difference_pp'],25);self.assertAlmostEqual(v['rres_minus']['e1_full']['difference_pp'],25);self.assertAlmostEqual(v['rres_minus']['random_three_seed_mean']['difference_pp'],35)
   finally:
    sys.path[:]=saved_path
    for n in names:
     sys.modules.pop(n,None)
     if saved[n] is not None:sys.modules[n]=saved[n]
if __name__=='__main__':unittest.main()
