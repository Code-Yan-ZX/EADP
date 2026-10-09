"""Independent predeclared post-batch worker derived from frozen legacy validator.

Default/import/prepare/validate use no GPU. --run is only for the coordinator.
Recipes are recovered archived script settings, not proven historical argv/env.
"""
from __future__ import annotations
import argparse,ast,base64,datetime,hashlib,importlib.util,json,os,runpy,sys
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
OUT=Path(__file__).resolve().parent
ROOT=Path('/media/disk2/YZX/research/EADP_amp');LLAVA=ROOT/'LLaVA'
WRAP=ROOT/'Qwen_vl/scripts/stage1_roundtrip_pilot'
AUDIT=ROOT/'Qwen_vl/outputs/audit_followup_20261008'
INVENTORY=OUT/'inventory.json'
UTIL=AUDIT/'next_gap_diagnosis_20261008/stability/next_text_stream_panel.py'
PATCH_HELPER=AUDIT/'rerun_fast/run_pope_streamwait_control.py'
PYTHON='/home/dell/miniconda3/envs/llava_pruner/bin/python'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(tmp,path)
def utility():
    s=importlib.util.spec_from_file_location('legacy_frozen_util',UTIL);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def job_specs():
    specs=json.loads(INVENTORY.read_text())['specs']
    expected={'next_gqa_E128','v15_vizwiz_E128','v15_vizwiz_AZ128','next_vizwiz_E128','next_vizwiz_AZ128'}
    if set(specs)!=expected or sum(j['n'] for j in specs.values())!=29854:raise RuntimeError('Fixed followup scope differs')
    for key,j in specs.items():
        p=j['parameters'];gqa=j['task']=='gqa'
        if j['budget']!=128 or p['visual_token_num']!=128 or p['alpha']!=(.5 if gqa else 0.) or p['beta']!=(2. if gqa else 1.) or p['max_new_tokens']!=128 or p['temperature']!=0 or p['conv_mode']!='vicuna_v1' or p['num_beams']!=1 or p['num_chunks']!=1 or p['chunk_idx']!=0 or p['top_p'] is not None or p['anchorzip']!=(j['method']=='AnchorZip'):raise RuntimeError('Frozen declared followup recipe differs')
        if j['min_free_memory_mib']!=22500 or j['exclusive_gpu']:raise RuntimeError('Pruned followup GPU gate differs')
    return specs

def paths(job_id,attempt=1):
    if job_id not in job_specs() or type(attempt)is not int or attempt<1:raise ValueError('Unknown legacy job/attempt')
    stem=job_id+'_streamwait'+(f'_attempt{attempt}' if attempt!=1 else '');p=OUT/(stem+'.jsonl')
    return dict(prediction=p,runtime=Path(str(p)+'.runtime.jsonl'),native_protocol=Path(str(p)+'.protocol.json'),control_protocol=OUT/(stem+'.control.protocol.json'),score=OUT/(stem+'.official.score.json'),state=OUT/(stem+'.state.json'),finished=OUT/(stem+'.finished.json'),log=OUT/(stem+'.log'))

def mmbench_inputs(spec):
    import pandas as pd
    tree=ast.parse(Path(spec['origin']['native_entry']).read_text());helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['is_none','get_options']]
    ns={'math':__import__('math')};exec(compile(ast.fix_missing_locations(ast.Module(body=helpers,type_ignores=[])),'frozen_mmbench_prompt_helpers','exec'),ns)
    frame=pd.read_table(spec['parameters']['question_file']);rows=[];embedded={}
    for _,row in frame.iterrows():
        idx=int(row['index']);image=row['image'];base=idx%1000000
        if isinstance(image,str) and len(image)>100:embedded[base]=image
        options=ns['get_options'](row,list('ABCD'));prompt=row['question']
        if not ns['is_none'](row['hint']):prompt=row['hint']+'\n'+prompt
        for letter,option in zip('ABCD',options):prompt+='\n'+letter+'. '+option
        rows.append(dict(question_id=idx,text=prompt,image='MMBench-base:'+str(base),options=options,round_id=0))
    if any(int(r['question_id'])%1000000 not in embedded for r in rows):raise RuntimeError('Missing embedded MMbench base image')
    return rows,embedded

def questions(spec):
    if spec['task'] in ['mmben','mmbcn']:return mmbench_inputs(spec)[0]
    return [json.loads(l) for l in Path(spec['parameters']['question_file']).open()]

def identity_manifests(spec):
    qfile=spec['parameters']['question_file'];rows=questions(spec)
    if len(rows)!=spec['n']:raise RuntimeError('Question count differs from registered inventory')
    expected=[dict(question_id=r['question_id'],prompt=r['text'],image=r['image'],**({'options':r['options'],'round_id':0} if 'options'in r else {})) for r in rows]
    if len({(str(r['question_id']),r['prompt']) for r in expected})!=spec['n']:raise RuntimeError('Input full composite keys not unique')
    old_path=Path(spec['origin']['original_prediction'])
    if sha(old_path)!=spec['origin']['original_prediction_sha256']:raise RuntimeError('Original predictions changed')
    old=[json.loads(l) for l in old_path.open()]
    if [(str(r['question_id']),r['prompt']) for r in old]!=[(str(r['question_id']),r['prompt']) for r in expected]:raise RuntimeError('Native prompt reconstruction differs from original complete predictions')
    ident_path=OUT/('legacy_'+spec['task']+'_identities.manifest.json');im_path=OUT/('legacy_'+spec['task']+'_images.manifest.json')
    # Same task shares the original question/image dataset across models and arms.
    ident=dict(n=len(expected),question_file=qfile,question_sha256=sha(qfile),rows=expected)
    if ident_path.exists():
        if json.loads(ident_path.read_text())!=ident:raise RuntimeError('Preserve differing existing identity manifest')
    else:dump(ident_path,ident)
    if not im_path.exists():
        if spec['task'] in ['mmben','mmbcn']:
            embedded=mmbench_inputs(spec)[1]
            records=[dict(image='MMBench-base:'+str(base),embedded_base=base,bytes_sha256=hashlib.sha256(base64.b64decode(encoded)).hexdigest()) for base,encoded in sorted(embedded.items())]
            images=dict(kind='embedded_base64',question_file=qfile,question_sha256=sha(qfile),records=records)
        else:
            records=[dict(image=image,path=str(Path(spec['parameters']['image_folder'])/image),sha256=sha(Path(spec['parameters']['image_folder'])/image)) for image in sorted({r['image'] for r in rows})]
            images=dict(kind='files',question_file=qfile,question_sha256=sha(qfile),records=records)
        dump(im_path,images)
    else:
        images=json.loads(im_path.read_text())
        if images['question_sha256']!=sha(qfile):raise RuntimeError('Existing image source differs')
    return ident_path,im_path

def verify_sources(protocol):
    if sha(__file__)!=protocol['launcher_sha256']:raise RuntimeError('Frozen legacy launcher changed')
    for path,digest in protocol['source_and_input_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen source/input changed: '+path)

def verify_images(protocol):
    path=Path(protocol['image_manifest'])
    if sha(path)!=protocol['image_manifest_sha256']:raise RuntimeError('Image manifest changed')
    manifest=json.loads(path.read_text())
    if sha(manifest['question_file'])!=manifest['question_sha256']:raise RuntimeError('Image question source changed')
    if manifest['kind']=='files':
        for record in manifest['records']:
            if sha(record['path'])!=record['sha256']:raise RuntimeError('Image bytes changed: '+record['path'])
    elif manifest['kind']=='embedded_base64':
        embedded=mmbench_inputs(protocol)[1]
        if len(embedded)!=len(manifest['records']):raise RuntimeError('Embedded image count changed')
        for record in manifest['records']:
            if hashlib.sha256(base64.b64decode(embedded[record['embedded_base']])).hexdigest()!=record['bytes_sha256']:raise RuntimeError('Embedded image bytes changed')
    else:raise RuntimeError('Unknown image manifest kind')

def prepare(job_id,attempt=1):
    OUT.mkdir(parents=True,exist_ok=True);spec=job_specs()[job_id];p=paths(job_id,attempt)
    frozen={str(OUT/'POLICY.json'):sha(OUT/'POLICY.json'),str(Path(__file__).resolve()):sha(__file__),str(INVENTORY):sha(INVENTORY),str(UTIL):sha(UTIL),str(PATCH_HELPER):sha(PATCH_HELPER)}
    for source in [Path(spec['origin']['native_entry']),Path(spec['origin']['score_entry']),WRAP/'official_score.py',WRAP/'rescore_official_20261008.py',WRAP/'vizwiz_official_normalization.py',WRAP/'mme_canonical_score.py',WRAP/'mmben_circular_score.py',LLAVA/'llava/model/llava_arch.py',LLAVA/'llava/model/llava_arch_anchorzip.py',LLAVA/'llava/model/builder.py',LLAVA/'llava/model/multimodal_encoder/clip_encoder.py',LLAVA/'llava/model/multimodal_encoder/builder.py',LLAVA/'llava/model/multimodal_projector/builder.py',LLAVA/'llava/model/language_model/llava_llama.py',LLAVA/'llava/mm_utils.py',LLAVA/'llava/conversation.py',LLAVA/'llava/constants.py',ROOT/'Qwen_vl/scripts/anchor_merge_pilot/amp_common.py',Path(spec['parameters']['model_path'])/'config.json',Path(spec['parameters']['model_path'])/'generation_config.json']:
        frozen[str(source)]=sha(source)
    origin=spec['origin'];frozen[origin['native_question_file']]=origin['question_sha256'];frozen[origin['original_prediction']]=origin['original_prediction_sha256']
    native_evidence=[r for r in origin['source_evidence'] if r['path']==origin['native_entry']]
    if not native_evidence or any(r['sha256']!=sha(origin['native_entry']) for r in native_evidence):raise RuntimeError('Inventory native entry identity changed')
    for score_input in origin['score_data']:frozen[score_input['path']]=score_input['sha256']
    for source,digest in frozen.items():
        if sha(source)!=digest:raise RuntimeError('Registered source/GT/original predictions changed: '+source)
    module,validation=utility().patch_module();compile(module,'legacy_wait_only','exec')
    score_gate_proof=empty_gate_patch(WRAP/('mme_canonical_score.py' if spec['task']=='mme' else 'official_score.py'),'checked_records' if spec['task']=='mme' else 'checked_predictions')[1] if spec['task']not in ['mmben','mmbcn'] else {'patch_required':False}
    identities,images=identity_manifests(spec)
    argv=list(spec['declared_generate_command'][1:]);argv[argv.index('--answers-file')+1]=str(p['prediction'])
    protocol=dict(spec,protocol='post_batch_prespecified_full_pairs',attempt=attempt,prepared_utc=now(),source_and_input_sha256=frozen,launcher_sha256=sha(__file__),AST_validation=validation,CPU_score_gate_AST_validation=score_gate_proof,
      identity_manifest=str(identities),identity_manifest_sha256=sha(identities),image_manifest=str(images),image_manifest_sha256=sha(images),native_runtime_argv=argv,
      output_paths={k:str(v) for k,v in p.items()},generate_command=[PYTHON,str(Path(__file__).resolve()),'--run','--job-id',job_id,'--attempt',str(attempt)],lambda_completion=.25,
      production_modified=False,pre_vision_GPU_hash_or_sync=False,index_trace=False,record_stream_added=False,
      scoring_empty_output_policy='A complete native empty EOS answer remains in the official denominator and uses the original official parser. CPU-only scorer AST removes only the empty-string integrity rejection in checked_predictions/checked_records; identity, GT, explicit FAILED checks and score math remain unchanged. Scorer source files are unchanged.',
      scoring_report_order='GQA and VizWiz per_question report entries and unrounded value reduction follow the complete original question-file order; official per-question correctness and summary are unchanged. This makes full-score validation reproducible across independent Python hash seeds.',
      runtime_observation='Only native generate return and metadata observed after generation; original complete order and prompts retained',
      historical_identity_limit='Explicit new prespecified protocol; public defaults are not proven paper argv. Original files only supply identity references. GQA reuses completed same-recipe AZ; VizWiz regenerates both methods.',
      source_evidence_policy='Driver paths/SHA are provenance only, not publisher archive payload; original model math/native entry/scorers are frozen sources')
    if p['control_protocol'].exists():
        old=json.loads(p['control_protocol'].read_text());a=dict(old);b=dict(protocol);a.pop('prepared_utc');b.pop('prepared_utc')
        if a!=b:raise RuntimeError('Preserve existing differing legacy protocol')
        return old,p
    dump(p['control_protocol'],protocol);return protocol,p

def guidance(protocol,prompt):
    return prompt.replace('\nAnswer the question using a single word or phrase.','') if Path(protocol['origin']['native_entry']).name=='llava_eval_arm_loader.py' else prompt

def empty_gate_patch(path,gate_name):
    # The worker already verifies complete native generation. Empty EOS is a
    # model answer, so relax only the redundant empty-string gate in memory.
    tree=ast.parse(Path(path).read_text());original_ast=ast.dump(tree,include_attributes=False);gate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==gate_name);removed=[]
    exact='not answer.strip()' if gate_name=='checked_records' else 'not str(value).strip()';expression=ast.dump(ast.parse(exact,mode='eval').body,include_attributes=False)
    for node in ast.walk(gate):
        if isinstance(node,ast.BoolOp) and isinstance(node.op,ast.Or):
            keep=[]
            for value in node.values:
                if ast.dump(value,include_attributes=False)==expression:removed.append((node,len(keep),value))
                else:keep.append(value)
            node.values=keep
    if len(removed)!=1:raise RuntimeError('Frozen CPU empty-answer gate AST changed')
    patched_ast=ast.dump(tree,include_attributes=False);node,index,value=removed[0];node.values.insert(index,value)
    if ast.dump(tree,include_attributes=False)!=original_ast:raise RuntimeError('CPU score AST changes beyond exact empty-only clause')
    node.values.pop(index)
    proof=dict(patch_required=True,function=gate_name,removed_expression=exact,original_ast_sha256=hashlib.sha256(original_ast.encode()).hexdigest(),patched_ast_sha256=hashlib.sha256(patched_ast.encode()).hexdigest(),restoration_ast_identical=True,source_file_modified=False,isolated_globals=True)
    return tree,proof

def complete_answer_scorer(path,gate_name):
    tree,_=empty_gate_patch(path,gate_name)
    ns={'__name__':'legacy_complete_answer_scorer','__file__':str(path)};exec(compile(ast.fix_missing_locations(tree),str(path)+'[empty-only-gate-relaxed]','exec'),ns)
    return SimpleNamespace(**ns)

def scoring(protocol,prediction):
    sys.path.insert(0,str(WRAP));task=protocol['task'];qfile=protocol['parameters']['question_file'];gt=protocol['origin']['score_data'][0]['path']
    if task=='mme':
        mme_canonical_score=complete_answer_scorer(WRAP/'mme_canonical_score.py','checked_records')
        result,_=mme_canonical_score.score(prediction,qfile,gt);value=result['summary']['perception'];metric='MME perception acc+paired-image-acc category sum'
    elif task in ['pope','gqa','vizwiz']:
        official_score=complete_answer_scorer(WRAP/'official_score.py','checked_predictions')
        result=official_score.score(task,prediction,question_file=qfile,annotation_dir=str(Path(qfile).parent),gt_file=gt)
        if not result['summary']['complete']:raise RuntimeError('Official complete denominator failed')
        if task in ['gqa','vizwiz']:
            order={str(row['question_id']):i for i,row in enumerate(questions(protocol))}
            result['per_question'].sort(key=lambda row:order[str(row['question_id'])])
        if task=='pope':value=sum(200*result['categories'][c]['TP']/(2*result['categories'][c]['TP']+result['categories'][c]['FP']+result['categories'][c]['FN']) if 2*result['categories'][c]['TP']+result['categories'][c]['FP']+result['categories'][c]['FN'] else 0 for c in ['random','popular','adversarial'])/3;metric='POPE three-category mean F1 (%)'
        else:value=100*sum(r['correct'] if task=='gqa' else r['acc'] for r in result['per_question'])/protocol['n'];metric='GQA exact-match accuracy (%)' if task=='gqa' else 'VizWiz official leave-one-out accuracy (%)'
    else:
        tree=ast.parse(Path(protocol['origin']['score_entry']).read_text());helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract_letter');main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Import) and any(a.name=='pandas' for a in n.names));body=[]
        for node in main.body[start:]:
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='print':break
            body.append(node)
        ns={'json':json,'re':__import__('re'),'args':SimpleNamespace(question_file=qfile,result_file=str(prediction),out=None)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[helper]+body,type_ignores=[])),'frozen_mmbench_circular_cpu','exec'),ns);result=ns['res']
        if len(ns['df'])!=4876 or len(ns['preds'])!=4876 or result['n_groups']!=1292:raise RuntimeError('MMBench complete circular denominator failed')
        result['n_rows']=4876
        result['unparsed_rows_counted_wrong']=sum(ns['extract_letter'](text)is None for text in ns['preds'].values())
        value=100*result['circular_correct']/result['n_groups'];metric='MMBench circular accuracy (%)'
    return dict(result,value=value,metric=metric)

def validate_artifacts(protocol,p):
    verify_sources(protocol);verify_images(protocol)
    registered=job_specs()[protocol['job_id']]
    for key in ['model','task','method','budget','n','parameters','origin','declared_generate_command']:
        if protocol[key]!=registered[key]:raise RuntimeError('Registered recipe changed: '+key)
    ident=Path(protocol['identity_manifest'])
    if sha(ident)!=protocol['identity_manifest_sha256']:raise RuntimeError('Identity manifest changed')
    expected=json.loads(ident.read_text())['rows'];rows=[json.loads(l) for l in p['prediction'].read_text().splitlines()];traces=[json.loads(l) for l in p['runtime'].read_text().splitlines()]
    if len(rows)!=protocol['n'] or len(traces)!=protocol['n']:raise RuntimeError('Full raw/runtime counts required')
    if [(str(r['question_id']),r['prompt']) for r in rows]!=[(str(r['question_id']),r['prompt']) for r in expected]:raise RuntimeError('Full ordered native composite question/prompt identities differ')
    qsha=sha(protocol['parameters']['question_file']);budget=protocol['budget'];params=protocol['parameters']
    for position,(row,trace,q) in enumerate(zip(rows,traces,expected)):
        if not isinstance(row.get('text'),str) or row['text'].strip().upper().startswith('FAILED'):raise RuntimeError('Non-string/explicit FAILED output')
        if trace['question_position']!=position or str(trace['question_id'])!=str(q['question_id']) or trace['image']!=q['image'] or trace['prompt_sha256']!=hashlib.sha256(q['prompt'].encode()).hexdigest() or trace['input_sha256']!=qsha or trace['visual_token_budget_parameter']!=budget or trace['guidance_sha256']!=hashlib.sha256(guidance(protocol,q['prompt']).encode()).hexdigest():raise RuntimeError('Native runtime full ordered identity changed')
        shape=trace['image_tensor_shape'];count=trace['actual_visual_tokens_retained']
        if type(count)is not int or count<1 or (len(shape)!=4 and len(shape)!=5) or any(type(d)is not int or d<1 for d in shape) or protocol['model']=='next' and len(shape)!=5 or shape[0]!=1 or shape[-3:]!=[3,336,336]:raise RuntimeError('Actual image shape/token count invalid')
        crops=shape[1] if len(shape)==5 else 1
        if (budget==0 and count!=crops*576) or (budget>0 and (count>crops*budget or protocol['model']=='v15' and count!=budget)):raise RuntimeError('Runtime budget geometry mismatch')
        want=dict(temperature=params['temperature'],do_sample=False,top_p=params['top_p'],num_beams=params['num_beams'],max_new_tokens=params['max_new_tokens'],use_cache=True)
        if trace['native_generate_parameters']!=want:raise RuntimeError('Actual native generate parameters differ')
        if any(trace.get('actual_model_object',{}).get(k)!=params[k] for k in ['alpha','beta','visual_token_num']):raise RuntimeError('Actual loaded object differs')
        if trace['loaded_method_parameters']!={k:params[k] for k in ['model_path','model_base','alpha','beta','visual_token_num']}:raise RuntimeError('Actual native model/budget parameters differ')
        if protocol['method']=='AnchorZip' and (trace['anchorzip_lambda']!=.25 or trace['anchorzip_mode']!='rtg'):raise RuntimeError('Frozen AnchorZip math changed')
        if protocol['task'] in ['mmben','mmbcn'] and (row['options']!=q['options'] or row['round_id']!=0):raise RuntimeError('MMBench rotation row/options changed')
    native=json.loads(p['native_protocol'].read_text())
    if native['parameters']!=params or native['native_runtime_argv']!=protocol['native_runtime_argv'] or native['wrapper_sha256']!=sha(protocol['origin']['native_entry']) or native['question_file_sha256']!=qsha or native['AST_validation']!=protocol['AST_validation']:raise RuntimeError('Native control sidecar changed')
    result=scoring(protocol,p['prediction'])
    return dict(result,success=True,n=protocol['n'],model=protocol['model'],task=protocol['task'],method=protocol['method'],budget=budget,visual_token_num=budget,prediction_sha256=sha(p['prediction']),runtime_sha256=sha(p['runtime']),control_protocol_sha256=sha(p['control_protocol']),image_manifest_sha256=protocol['image_manifest_sha256'],actual_count_distribution={str(k):v for k,v in Counter(t['actual_visual_tokens_retained'] for t in traces).items()},integrity='Complete native ordered composite question/prompt identity, actual runtime geometry/generation parameters, sources/GT/images and legal original guidance')

def validate_finished(job_id_or_protocol,attempt=1):
    if isinstance(job_id_or_protocol,str):p=paths(job_id_or_protocol,attempt);protocol=json.loads(p['control_protocol'].read_text())
    else:protocol=job_id_or_protocol;p=paths(protocol['job_id'],protocol['attempt'])
    if protocol['output_paths']!={k:str(v) for k,v in p.items()}:raise RuntimeError('Output path differs from fixed registered paths')
    if json.loads(p['control_protocol'].read_text())!=protocol:raise RuntimeError('In-memory protocol differs from immutable saved protocol')
    state=json.loads(p['state'].read_text());marker=json.loads(p['finished'].read_text())
    if state.get('status')!='complete' or state.get('complete')is not True or type(state.get('generation_exit_code'))is not int or state.get('generation_exit_code')!=0 or marker.get('job_id')!=protocol['job_id'] or marker.get('success')is not True or marker.get('complete')is not True or marker.get('n')!=protocol['n'] or type(marker.get('generation_exit_code'))is not int or marker.get('generation_exit_code')!=0:raise RuntimeError('Full generation exit0/score gate absent')
    result=validate_artifacts(protocol,p);saved=json.loads(p['score'].read_text())
    if result!=saved or marker['score_sha256']!=sha(p['score']) or marker['prediction_sha256']!=result['prediction_sha256']:raise RuntimeError('Score/complete marker hash mismatch')
    files={str(p[k]):sha(p[k]) for k in ['prediction','runtime','native_protocol','control_protocol','score','finished']};files.update(protocol['source_and_input_sha256']);files[str(Path(__file__).resolve())]=protocol['launcher_sha256']
    files[protocol['identity_manifest']]=protocol['identity_manifest_sha256'];files[protocol['image_manifest']]=protocol['image_manifest_sha256']
    return dict(score=result,files=files,finished=marker,protocol=protocol)

def model_object_snapshot(model):
    core=model.get_model()
    return dict(alpha=getattr(model,'alpha',getattr(core,'alpha',None)),beta=getattr(model,'beta',getattr(core,'beta',None)),visual_token_num=getattr(model,'visual_token_num',getattr(core,'visual_token_num',None)),model_class=type(model).__module__+'.'+type(model).__name__,core_class=type(core).__name__,dtype=str(next(model.parameters()).dtype),attention_backend=getattr(model.config,'_attn_implementation',None),vision_class=type(model.get_vision_tower()).__name__)

def native_runtime(protocol,p):
    from llava.model import builder
    loader=builder.load_pretrained_model;position=0;rows=json.loads(Path(protocol['identity_manifest']).read_text())['rows'];qsha=sha(protocol['parameters']['question_file']);trace=p['runtime'].open('x')
    def traced_loader(*args,**kwargs):
        loaded_params=dict(model_path=args[0],model_base=args[1],**{k:kwargs[k] for k in ['alpha','beta','visual_token_num']});loaded=loader(*args,**kwargs);model=loaded[1];original=model.generate
        actual=model_object_snapshot(model)
        if actual['dtype']!='torch.float16' or actual['attention_backend']!='sdpa' or any(actual[k]!=protocol['parameters'][k] for k in ['alpha','beta','visual_token_num']):raise RuntimeError('Loaded object parameters disagree with protocol')
        dump(OUT/(protocol['job_id']+'.loaded_model.json'),actual)
        def generated(*gen_args,**gen_kwargs):
            nonlocal position
            returned=original(*gen_args,**gen_kwargs)
            if not isinstance(returned,tuple) or len(returned)!=2:raise RuntimeError('Native generate tuple required')
            row=rows[position];images=gen_kwargs['images'];texts=gen_kwargs['texts']
            # Everything below is observed after generation; no vision timing hook.
            if texts!=guidance(protocol,row['prompt']):raise RuntimeError('Actual native guidance changed')
            az=sys.modules.get('llava_arch_anchorzip');record=dict(question_id=row['question_id'],question_position=position,image=row['image'],prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),guidance_sha256=hashlib.sha256(texts.encode()).hexdigest(),input_sha256=qsha,actual_visual_tokens_retained=int(returned[1]),visual_token_budget_parameter=protocol['budget'],image_tensor_shape=list(images.shape),image_sizes=gen_kwargs.get('image_sizes'),native_generate_parameters={k:gen_kwargs[k] for k in ['temperature','do_sample','top_p','num_beams','max_new_tokens','use_cache']},loaded_method_parameters=loaded_params,anchorzip_lambda=getattr(az,'LAM',None),anchorzip_mode=getattr(az,'MODE',None),actual_model_object=actual,source='Original native generate return observed after generation')
            trace.write(json.dumps(record)+'\n');trace.flush();position+=1;return returned
        model.generate=generated;return loaded
    builder.load_pretrained_model=traced_loader
    try:
        runpy.run_path(protocol['origin']['native_entry'],run_name='__main__')
        if position!=protocol['n']:raise RuntimeError('Incomplete native runtime')
    finally:builder.load_pretrained_model=loader;trace.close()

def run(protocol,p):
    for k in ['prediction','runtime','native_protocol','state','score','finished']:
        if p[k].exists():raise FileExistsError('Preserve existing artifact: '+str(p[k]))
    slot=utility().safe_slot()
    if slot['free_memory_mib']<protocol['min_free_memory_mib'] or protocol['exclusive_gpu'] and slot['other_gpu_models']:raise RuntimeError('Conservative per-job GPU gate failed')
    verify_sources(protocol);verify_images(protocol)
    for name in ['USE_LLAVA_ARCH_CDPRUNER','USE_LLAVA_ARCH_DIVPRUNE','USE_LLAVA_ARCH_HIPRUNE','USE_LLAVA_ARCH_ABLATION']:
        if os.environ.get(name)=='1':raise RuntimeError('Unexpected alternate architecture')
    state=dict(status='generating',pid=os.getpid(),started_utc=now(),concurrency=slot,generation_exit_code=None);dump(p['state'],state)
    sys.path.insert(0,str(LLAVA));from llava.model.multimodal_encoder import clip_encoder
    original=clip_encoder.CLIPVisionTower.forward;module,validation=utility().patch_module()
    if validation!=protocol['AST_validation']:raise RuntimeError('Wait-only AST changed')
    ns=dict(clip_encoder.__dict__);exec(compile(module,'legacy_wait_only','exec'),ns);clip_encoder.CLIPVisionTower.forward=ns['forward']
    oldargv=sys.argv;oldcwd=os.getcwd();sys.argv=list(protocol['native_runtime_argv']);os.chdir(LLAVA)
    dump(p['native_protocol'],dict(protocol='legacy_native_recipe_wait_only',parameters=protocol['parameters'],native_runtime_argv=protocol['native_runtime_argv'],question_file_sha256=sha(protocol['parameters']['question_file']),wrapper_sha256=sha(protocol['origin']['native_entry']),AST_validation=validation,source='Original native entry executed with recovered archived recipe; original entry writes no sidecar'))
    try:native_runtime(protocol,p)
    except BaseException as exc:state.update(status='failed',generation_exit_code=1,error=repr(exc));dump(p['state'],state);raise
    finally:sys.argv=oldargv;os.chdir(oldcwd);clip_encoder.CLIPVisionTower.forward=original
    state.update(status='scoring',generation_exit_code=0,generation_finished_utc=now());dump(p['state'],state)
    result=validate_artifacts(protocol,p);dump(p['score'],result);dump(p['finished'],dict(success=True,complete=True,n=protocol['n'],generation_exit_code=0,finished_utc=now(),job_id=protocol['job_id'],score_sha256=sha(p['score']),prediction_sha256=result['prediction_sha256']))
    state.update(status='complete',complete=True,finished_utc=now());dump(p['state'],state);print(json.dumps(result,ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--validate-finished',action='store_true');parser.add_argument('--job-id',choices=job_specs(),required=True);parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    if args.run and args.validate_finished:parser.error('Choose GPU run or CPU validation')
    if args.validate_finished:print(json.dumps(validate_finished(args.job_id,args.attempt),ensure_ascii=False,indent=2));return
    protocol,p=prepare(args.job_id,args.attempt)
    if args.run:run(protocol,p)
    else:print(json.dumps(dict(CPU_only=True,job_id=args.job_id,n=protocol['n'],control_protocol=str(p['control_protocol']),generate_command=protocol['generate_command']),indent=2))
if __name__=='__main__':main()
