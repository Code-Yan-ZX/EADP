"""CPU integrity boundaries and after-return observation; never generate a GPU run."""
import contextlib,copy,importlib.util,io,json,os,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('legacy_worker_under_test',HERE/'legacy_streamwait_worker.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w)

class LegacyBoundaries(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.params=dict(model_path='model',model_base=None,alpha=.5,beta=2.,visual_token_num=32,temperature=0.,top_p=None,num_beams=1,max_new_tokens=128,question_file=str(self.root/'input'),image_folder='images',conv_mode='vicuna_v1',anchorzip=False)
  (self.root/'input').write_text('fixed ordered input')
  self.expected=[dict(question_id='image.jpg',prompt='First?',image='image.jpg'),dict(question_id='image.jpg',prompt='Second?',image='image.jpg')]
  self.p={k:self.root/(k+'.json') for k in ['prediction','runtime','native_protocol','control_protocol','score','state','finished']}
  self.wrapper=self.root/'native.py';self.wrapper.write_text('original native source')
  self.ident=self.root/'identity.json';w.dump(self.ident,dict(rows=self.expected))
  self.protocol=dict(job_id='fixture',n=2,model='v15',task='mme',method='EADP',budget=32,parameters=self.params,origin={'native_entry':str(self.wrapper)},original_generate_command=['python','native'],identity_manifest=str(self.ident),identity_manifest_sha256=w.sha(self.ident),image_manifest_sha256='imageSHA')
  w.dump(self.p['control_protocol'],self.protocol)
  self.rows=[dict(question_id=q['question_id'],prompt=q['prompt'],text='Yes') for q in self.expected]
  self.trace=[dict(question_position=i,question_id=q['question_id'],image=q['image'],prompt_sha256=w.hashlib.sha256(q['prompt'].encode()).hexdigest(),guidance_sha256=w.hashlib.sha256(q['prompt'].encode()).hexdigest(),input_sha256=w.sha(self.params['question_file']),visual_token_budget_parameter=32,image_tensor_shape=[1,3,336,336],actual_visual_tokens_retained=32,native_generate_parameters=dict(temperature=0.,do_sample=False,top_p=None,num_beams=1,max_new_tokens=128,use_cache=True),loaded_method_parameters={k:self.params[k] for k in ['model_path','model_base','alpha','beta','visual_token_num']}) for i,q in enumerate(self.expected)]
  self.protocol['AST_validation']={'wait_only':True}
  w.dump(self.p['native_protocol'],dict(parameters=self.params,native_runtime_argv=['native'],wrapper_sha256=w.sha(self.wrapper),question_file_sha256=w.sha(self.params['question_file']),AST_validation=self.protocol['AST_validation']))
  self.protocol['native_runtime_argv']=['native']
  self.write()
 def tearDown(self):self.tmp.cleanup()
 def write(self):
  for key,data in [('prediction',self.rows),('runtime',self.trace)]:self.p[key].write_text(''.join(json.dumps(r)+'\n' for r in data))
 def validate(self):
  with patch.object(w,'verify_sources'),patch.object(w,'verify_images'),patch.object(w,'job_specs',return_value={'fixture':self.protocol}),patch.object(w,'scoring',return_value=dict(value=100.,metric='fixture')):
   return w.validate_artifacts(self.protocol,self.p)
 def reject(self):
  self.write()
  with self.assertRaises((RuntimeError,KeyError)):self.validate()
 def test_inventory_and_fixed_paths(self):
  specs=w.job_specs();self.assertEqual((len(specs),sum(j['n'] for j in specs.values())),(32,224199))
  self.assertFalse(specs['v15_mme_FULL']['exclusive_gpu']);self.assertEqual(specs['v15_mme_FULL']['min_free_memory_mib'],22500)
  self.assertEqual(specs['next_mmbcn_LRMAIN025']['parameters']['lang'],'en')
  with self.assertRaises(ValueError):w.paths('../../escape')
  with self.assertRaises(ValueError):w.paths('v15_mme_FULL',0)
 def test_composite_shared_image_ids_legal(self):self.assertEqual(self.validate()['n'],2)
 def test_partial_raw_rejected(self):self.rows.pop();self.reject()
 def test_partial_runtime_rejected(self):self.trace.pop();self.reject()
 def test_prediction_order_rejected(self):self.rows.reverse();self.reject()
 def test_runtime_order_rejected(self):self.trace.reverse();self.reject()
 def test_duplicate_full_key_rejected(self):self.rows[1]=copy.deepcopy(self.rows[0]);self.reject()
 def test_failed_output_rejected(self):self.rows[0]['text']='FAILED crash';self.reject()
 def test_complete_empty_answer_is_model_error_not_missing(self):
  self.rows[0]['text']='';self.write();self.assertEqual(self.validate()['n'],2)
 def test_input_sha_rejected(self):self.trace[0]['input_sha256']='wrong';self.reject()
 def test_prompt_sha_rejected(self):self.trace[0]['prompt_sha256']='wrong';self.reject()
 def test_image_identity_rejected(self):self.trace[0]['image']='other.jpg';self.reject()
 def test_runtime_parameter_rejected(self):self.trace[0]['native_generate_parameters']['max_new_tokens']=1024;self.reject()
 def test_runtime_model_parameters_rejected(self):self.trace[0]['loaded_method_parameters']['beta']=1.;self.reject()
 def test_budget_parameter_rejected(self):self.trace[0]['visual_token_budget_parameter']=64;self.reject()
 def test_actual_budget_rejected(self):self.trace[0]['actual_visual_tokens_retained']=33;self.reject()
 def test_bad_shape_rejected(self):self.trace[0]['image_tensor_shape']=[1,3,224,224];self.reject()
 def test_next_requires_anyres_shape(self):self.protocol['model']='next';self.reject()
 def test_next_arbitrary_cropcount_legal(self):
  self.protocol['model']='next'
  for t in self.trace:t['image_tensor_shape']=[1,7,3,336,336];t['actual_visual_tokens_retained']=222
  self.write();self.assertEqual(self.validate()['n'],2)
 def test_full_geometry_rejected(self):
  self.protocol['budget']=0;self.params['visual_token_num']=0
  for t in self.trace:t['visual_token_budget_parameter']=0;t['loaded_method_parameters']['visual_token_num']=0
  self.reject()
 def test_native_sidecar_drift_rejected(self):
  d=json.loads(self.p['native_protocol'].read_text());d['parameters']['beta']=1.;w.dump(self.p['native_protocol'],d)
  with self.assertRaises(RuntimeError):self.validate()
 def test_identity_manifest_drift_rejected(self):
  self.ident.write_text('{}')
  with self.assertRaises(RuntimeError):self.validate()
 def test_frozen_source_drift_rejected(self):
  self.protocol.update(launcher_sha256=w.sha(w.__file__),source_and_input_sha256={str(self.wrapper):'wrong'})
  with self.assertRaises(RuntimeError):w.verify_sources(self.protocol)
 def test_failed_generation_cannot_release_finished(self):
  self.protocol['attempt']=1;self.protocol['output_paths']={k:str(v) for k,v in self.p.items()}
  w.dump(self.p['state'],dict(generation_exit_code=1));w.dump(self.p['finished'],dict(success=True,complete=True,n=2,generation_exit_code=0))
  with patch.object(w,'paths',return_value=self.p),self.assertRaises(RuntimeError):w.validate_finished(self.protocol)
 def test_boolean_exit_is_not_real_exit0(self):
  self.protocol['attempt']=1;self.protocol['output_paths']={k:str(v) for k,v in self.p.items()}
  w.dump(self.p['state'],dict(generation_exit_code=False));w.dump(self.p['finished'],dict(success=True,complete=True,n=2,generation_exit_code=0))
  with patch.object(w,'paths',return_value=self.p),self.assertRaises(RuntimeError):w.validate_finished(self.protocol)
 def test_in_memory_protocol_drift_rejected(self):
  self.protocol['attempt']=1;self.protocol['output_paths']={k:str(v) for k,v in self.p.items()};w.dump(self.p['control_protocol'],self.protocol)
  self.protocol['budget']=64
  with patch.object(w,'paths',return_value=self.p),self.assertRaisesRegex(RuntimeError,'immutable'):w.validate_finished(self.protocol)
 def test_marker_job_identity_rejected(self):
  self.protocol['attempt']=1;self.protocol['output_paths']={k:str(v) for k,v in self.p.items()};w.dump(self.p['control_protocol'],self.protocol)
  w.dump(self.p['state'],dict(status='complete',complete=True,generation_exit_code=0));w.dump(self.p['finished'],dict(job_id='other',success=True,complete=True,n=2,generation_exit_code=0))
  with patch.object(w,'paths',return_value=self.p),self.assertRaisesRegex(RuntimeError,'exit0'):w.validate_finished(self.protocol)
 def test_after_generate_return_observation(self):
  returned=object();seen=[]
  class Images:
   @property
   def shape(self):
    if not seen:raise AssertionError('Shape observed before generation')
    return (1,3,336,336)
  model=types.SimpleNamespace(generate=lambda *a,**kw:(seen.append(1) or returned,32))
  builder=types.SimpleNamespace(load_pretrained_model=lambda *a,**kw:('tokenizer',model,'processor',4096))
  llava=types.ModuleType('llava');llava_model=types.ModuleType('llava.model');llava_model.builder=builder
  self.wrapper.write_text("from llava.model import builder\n_,model,_,_=builder.load_pretrained_model('model',None,'name',alpha=.5,beta=2.,visual_token_num=32)\nfor prompt in ['First?','Second?']:\n model.generate(None,images=IMAGES,texts=prompt,temperature=0.,do_sample=False,top_p=None,num_beams=1,max_new_tokens=128,use_cache=True)\n")
  self.p['runtime'].unlink()
  original=runpy=w.runpy.run_path
  def native_run(path,run_name):return original(path,run_name=run_name,init_globals={'IMAGES':Images()})
  with patch.dict(sys.modules,{'llava':llava,'llava.model':llava_model}),patch.object(w.runpy,'run_path',side_effect=native_run):w.native_runtime(self.protocol,self.p)
  trace=[json.loads(l) for l in self.p['runtime'].read_text().splitlines()]
  self.assertEqual([t['question_position'] for t in trace],[0,1]);self.assertEqual(len(seen),2);self.assertEqual(trace[0]['actual_visual_tokens_retained'],32)

class RealOfficialCPU(unittest.TestCase):
 def test_mme_canonical_labels_not_rebuilt(self):
  spec=w.job_specs()['v15_mme_FULL'];protocol=dict(spec);result=w.scoring(protocol,Path(spec['origin']['original_prediction']))
  reference=json.loads((w.AUDIT/'mme_official_gt_rescore.json').read_text())['arms']['FULL']
  self.assertEqual(result['summary']['n'],2374);self.assertTrue(result['summary']['complete']);self.assertAlmostEqual(result['value'],reference['official_gt_perception'],places=7)
  for category,score in reference['category_scores'].items():self.assertAlmostEqual(result['categories'][category]['score'],score,places=7)
  self.assertEqual(result['protocol'],'mme-canonical-gt-20261008-v1')
 def test_wrong_mme_archive_rejected(self):
  sys.path.insert(0,str(w.WRAP));import mme_canonical_score
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'wrong.zip';p.write_bytes(b'wrong GT')
   with self.assertRaises(ValueError):mme_canonical_score.canonical_gt(p)
 def test_mmbench_real_complete_circular(self):
  spec=w.job_specs()['next_mmben_LRMAIN025'];score=w.scoring(spec,Path(spec['origin']['original_prediction']))
  self.assertEqual(score['n_groups'],1292);self.assertEqual(score['failed_groups_counted_0'],0)
 def test_mmbench_complete_unparseable_and_empty_answers_count_wrong(self):
  import pandas as pd
  spec=w.job_specs()['next_mmben_LRMAIN025'];frame=pd.read_table(spec['parameters']['question_file'])
  rows=[dict(question_id=int(r['index']),text=r['answer']) for _,r in frame.iterrows()]
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'complete.jsonl'
   for answer in ['Unable to determine the option.','']:
    rows[0]['text']=answer;p.write_text(''.join(json.dumps(r)+'\n' for r in rows));score=w.scoring(spec,p)
    self.assertEqual(score['n_rows'],4876);self.assertEqual(score['n_groups'],1292);self.assertEqual(score['unparsed_rows_counted_wrong'],1)
    self.assertEqual(score['circular_correct'],1291);self.assertAlmostEqual(score['value'],100*1291/1292)
 def test_mme_complete_empty_answers_keep_all_denominators(self):
  spec=w.job_specs()['v15_mme_FULL'];rows=[json.loads(l) for l in Path(spec['origin']['original_prediction']).read_text().splitlines()]
  for row in rows:row['text']=''
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'complete.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows));score=w.scoring(spec,p)
   self.assertEqual(score['summary']['n'],2374);self.assertTrue(score['summary']['complete']);self.assertEqual(score['value'],0)
 def test_gqa_complete_empty_answers_keep_all_denominators(self):
  spec=w.job_specs()['v15_gqa_FULL'];rows=[json.loads(l) for l in Path(spec['origin']['original_prediction']).read_text().splitlines()]
  for row in rows:row['text']=''
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'complete.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows));score=w.scoring(spec,p)
   self.assertEqual(score['summary']['n_lines'],12578);self.assertEqual(score['summary']['n_gt_ids'],12578);self.assertTrue(score['summary']['complete']);self.assertEqual(len(score['per_question']),12578)
 def test_nonempty_scores_match_frozen_originals(self):
  sys.path.insert(0,str(w.WRAP));import official_score,mme_canonical_score
  for job,spec in w.job_specs().items():
   pred=Path(spec['origin']['original_prediction']);qfile=spec['parameters']['question_file'];gt=spec['origin']['score_data'][0]['path']
   if spec['task']in ['mmben','mmbcn']:
    captured=io.StringIO()
    with patch.object(sys,'argv',[spec['origin']['score_entry'],'--question-file',qfile,'--result-file',str(pred)]),contextlib.redirect_stdout(captured):w.runpy.run_path(spec['origin']['score_entry'],run_name='__main__')
    original=json.loads(captured.getvalue())
   else:original=mme_canonical_score.score(pred,qfile,gt)[0] if spec['task']=='mme' else official_score.score(spec['task'],pred,question_file=qfile,annotation_dir=str(Path(qfile).parent),gt_file=gt)
   if spec['task']in ['gqa','vizwiz']:
    order={str(row['question_id']):i for i,row in enumerate(w.questions(spec))};original['per_question'].sort(key=lambda row:order[str(row['question_id'])])
   patched=w.scoring(spec,pred);self.assertEqual({k:v for k,v in patched.items() if k not in ['value','metric','n_rows','unparsed_rows_counted_wrong']},original,job)
 def test_complete_gqa_viz_two_real_cpu_hash_seeds_identical(self):
  for job in ['v15_gqa_FULL','next_vizwiz_LRMAIN025']:
   code="import hashlib,importlib.util,json;from pathlib import Path;p=Path("+repr(str(Path(w.__file__)))+");s=importlib.util.spec_from_file_location('hashseed_worker',p);w=importlib.util.module_from_spec(s);s.loader.exec_module(w);spec=w.job_specs()["+repr(job)+"];r=w.scoring(spec,Path(spec['origin']['original_prediction']));assert [str(p['question_id']) for p in r['per_question']]==[str(p['question_id']) for p in w.questions(spec)];print(hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest())"
   digests=[]
   for seed in ['1','2']:
    env=dict(os.environ,PYTHONHASHSEED=seed);digests.append(subprocess.check_output([sys.executable,'-c',code],env=env,text=True).strip())
   self.assertEqual(digests[0],digests[1],job);self.assertEqual(len(digests[0]),64)
 def test_cpu_empty_gate_exact_ast_restoration_and_no_global_mutation(self):
  sys.path.insert(0,str(w.WRAP));import official_score
  before=official_score.checked_predictions
  _,proof=w.empty_gate_patch(w.WRAP/'official_score.py','checked_predictions')
  w.complete_answer_scorer(w.WRAP/'official_score.py','checked_predictions')
  self.assertTrue(proof['restoration_ast_identical']);self.assertTrue(proof['isolated_globals']);self.assertEqual(proof['removed_expression'],'not str(value).strip()');self.assertIs(official_score.checked_predictions,before)

if __name__=='__main__':unittest.main(verbosity=2)
