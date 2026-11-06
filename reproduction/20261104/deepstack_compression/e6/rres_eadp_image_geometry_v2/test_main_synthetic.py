"""Actual main with synthetic 384 images, metadata and accepted-proof stubs."""
import hashlib,importlib.metadata,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import run

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x)+'\n')
def jsonl(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(v)+'\n' for v in x))
class MainTest(unittest.TestCase):
 def test_full_synthetic_main_writes_both_outputs(self):
  with tempfile.TemporaryDirectory() as tmp:
   b=Path(tmp);d=b/'e6/rres_eadp_image_geometry_v2';d.mkdir(parents=True);i=b/'e1_independent_review';i.mkdir()
   write(b/'initial_plan.json',{'budget_end_unix':run.END});plan={'status':'frozen','GT_opened':False,'scores_parsed':False,'individual_dimensions_read':False,'pillow_version':importlib.metadata.version('Pillow'),'source_sha256':{}};write(d/'plan001.json',plan);write(d/'cpu_proof001.json',{'status':'passed','plan_sha256':sha(d/'plan001.json'),'source_sha256':{}})
   image=b/'im.png';v=io.BytesIO();Image.new('RGB',(40,60)).save(v,format='PNG');image.write_bytes(v.getvalue());digest=sha(image)
   anchors=[];inputs=[];scores=[]
   for task in run.TASKS:
    for n in range(128):
     k={'task':task,'qid':str(n),'group_id':task+str(n),'image_sha256':digest,'message_sha256':'f'*64};anchors.append(k);inputs.append({**k,'image_path':str(image)});scores.append({**k,'rres':{'score':.7,'independent_score':.7},'eadp':{'score':.6,'independent_score':.6}})
   write(b/'e6/rres_k256_dev384/design001.json',{'anchors':anchors});inp=b/'e1/input_preparation_002/inputs.jsonl';jsonl(inp,inputs);ip=b/'e1/input_preparation_002/independent_verification.json';write(ip,{})
   ef=b/'e1/e1full001_completion.json';write(ef,{'status':'passed','natural_questions':384,'GT_opened':False,'files':{'inputs':{'path':str(inp),'sha256':sha(inp)},'input_proof':{'path':str(ip),'sha256':sha(ip)}}})
   for name in ('review_full.py','review_rres_k256_dev384_result.py','review_rres_k256_dev384_scores.py'):(i/name).write_text('')
   write(i/'e1full001_review.json',{'status':'passed','n_verified':384,'errors':[],'completion_proof_sha256':sha(ef),'reviewer_sha256':sha(i/'review_full.py')})
   formal=b/'e6/rres_k256_dev384/verification001.json';write(formal,{'status':'passed','images':384,'GT_opened':False})
   rr=i/'rres_k256_dev384_result_review.json';write(rr,{'status':'passed','independent_review':True,'images':384,'errors':[],'formal_sha256':sha(formal),'reviewer_sha256':sha(i/'review_rres_k256_dev384_result.py')})
   score=b/'e6/rres_k256_full_analysis_v2/score001/scores.jsonl';jsonl(score,scores);write(i/'rres_k256_dev384_scores_review.json',{'status':'passed','independent_review':True,'questions':384,'errors':[],'score_sha256':sha(score),'result_review_sha256':sha(rr),'reviewer_sha256':sha(i/'review_rres_k256_dev384_scores.py')})
   out=d/'result001.json'
   with patch.object(run,'B',b),patch.object(run,'D',d),patch.object(run,'I',i),patch.object(run,'terminal'),patch.object(sys,'argv',['run.py','--out',str(out)]):run.main()
   result=json.loads(out.read_text());self.assertEqual(result['questions'],384);self.assertEqual(len((d/'rows001.jsonl').read_text().splitlines()),384)
   for task in run.TASKS:
    self.assertEqual(result['tasks'][task]['aspect']['portrait']['questions'],128);self.assertAlmostEqual(result['tasks'][task]['all']['difference_pp'],10.)
if __name__=='__main__':unittest.main()
