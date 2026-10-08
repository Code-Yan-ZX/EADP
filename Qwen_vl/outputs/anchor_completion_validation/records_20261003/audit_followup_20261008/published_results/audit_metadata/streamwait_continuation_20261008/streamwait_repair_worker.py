"""Registered NeXT continuation: frozen native jobs plus only CLIP stream waits.

Import, job_specs and --prepare are CPU-only. Only explicit --run uses a GPU.
Original model/wrapper/queue files and predictions remain untouched.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

OUT=Path(__file__).resolve().parent
ROOT=Path('/media/disk2/YZX/research/EADP_amp')
LLAVA=ROOT/'LLaVA'
AUDIT=ROOT/'Qwen_vl/outputs/audit_followup_20261008'
LANE= AUDIT/'rerun_batch/lane2_next_plan.json'
UTIL=AUDIT/'next_gap_diagnosis_20261008/stability/next_text_stream_panel.py'
PATCH_HELPER=AUDIT/'rerun_fast/run_pope_streamwait_control.py'
TEXT_RUNTIME=ROOT/'Qwen_vl/scripts/stage1_roundtrip_pilot/textvqa_runtime_launcher.py'
TEXT_WRAPPER=TEXT_RUNTIME.with_name('llava_eval_arm_model_vqa.py')
SCIENCE=TEXT_RUNTIME.with_name('llava_eval_arm_science.py')
SQA_BASE=AUDIT/'sqa_image_only_base'
TEXT_GT=LLAVA/'playground/data/eval/textvqa/TextVQA_0.5.1_val.json'
TEXT_DIR=LLAVA/'playground/data/eval/anchorzip_p3/next_textvqa_official_20261008'
PYTHON='/home/dell/miniconda3/envs/llava_pruner/bin/python'
IDS={
 'next_text_E128':('textvqa','EADP',128,'EGATHER'),
 'next_text_AZ128':('textvqa','AnchorZip',128,'LRMAIN025'),
 'next_text_E64':('textvqa','EADP',64,'EGATHER_K64'),
 'next_text_AZ64':('textvqa','AnchorZip',64,'LRMAIN0125'),
 'next_text_FULL':('textvqa','FULL',0,'FULL'),
 'next_sqa_E128':('sqa','EADP',128,'E_GATHER_K128'),
 'next_sqa_AZ128':('sqa','AnchorZip',128,'AZ_K128'),
 'next_sqa_E64':('sqa','EADP',64,'E_GATHER_K64'),
 'next_sqa_AZ64':('sqa','AnchorZip',64,'AZ_K64'),
 'next_sqa_E32':('sqa','EADP',32,'E_GATHER_K32'),
 'next_sqa_AZ32':('sqa','AnchorZip',32,'AZ_K32'),
 'next_sqa_FULL':('sqa','FULL',0,'FULL')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(tmp,path)
def utility():
    spec=importlib.util.spec_from_file_location('frozen_stream_util',UTIL)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def paths(job_id,attempt=1):
    if job_id not in IDS or attempt<1:raise ValueError('Unknown registered job or invalid attempt')
    stem=job_id+'_streamwait'+(f'_attempt{attempt}' if attempt!=1 else '')
    prediction=OUT/(stem+'.jsonl')
    return dict(prediction=prediction,runtime=Path(str(prediction)+'.runtime.jsonl'),
      native_protocol=Path(str(prediction)+'.protocol.json'),control_protocol=OUT/(stem+'.control.protocol.json'),
      score=OUT/(stem+'.official.score.json'),state=OUT/(stem+'.state.json'),
      finished=OUT/(stem+'.finished.json'),log=OUT/(stem+'.log'))
def option(argv,name):return argv[argv.index(name)+1]
def questions(spec):
    p=Path(spec['parameters']['question_file'])
    return json.loads(p.read_text()) if spec['task']=='sqa' else [json.loads(s) for s in p.open()]

def job_specs():
    """Read original frozen jobs, returning the twelve exact registered specs."""
    lane=json.loads(LANE.read_text())
    by_pair={(j['task'],j['arm']):j for j in lane['jobs']}
    specs={}
    for job_id,(task,method,budget,arm) in IDS.items():
        original=by_pair.get((task,arm))
        if original is None:
            # Initial NeXT K128 pair predates the lane jobs and lacks runtime.
            native=Path(str(TEXT_DIR/(arm+'.jsonl'))+'.protocol.json')
            base=json.loads(native.read_text())
            argv=[PYTHON,str(TEXT_RUNTIME),'--model-path',base['model_path'],'--question-file',base['question_file'],
              '--image-folder',base['image_folder'],'--answers-file',base['answers_file'],'--conv-mode',base['conv_mode'],
              '--temperature','0','--visual_token_num',str(budget),'--alpha',str(base['alpha']),'--beta',str(base['beta']),
              '--official-textvqa']
            if base['anchorzip']:argv.append('--anchorzip')
            origin=dict(native_protocol=str(native),native_protocol_sha256=sha(native),
              initial_native_settings={k:base[k] for k in ['alpha','beta','anchorzip','temperature','max_new_tokens','visual_token_num','conv_mode']})
            if origin['initial_native_settings']!={'alpha':.5,'beta':2.,'anchorzip':method=='AnchorZip','temperature':0.,'max_new_tokens':128,'visual_token_num':128,'conv_mode':'vicuna_v1'}:
                raise RuntimeError('Initial K128 native protocol changed')
        else:
            argv=list(original['generate_command']);origin=dict(original_job=original)
        params=dict(model_path=option(argv,'--model-path'),question_file=option(argv,'--question-file'),
          image_folder=option(argv,'--image-folder'),conv_mode=option(argv,'--conv-mode'),visual_token_num=int(option(argv,'--visual_token_num')),
          alpha=float(option(argv,'--alpha')),beta=float(option(argv,'--beta')),temperature=float(option(argv,'--temperature')),
          anchorzip='--anchorzip' in argv,lambda_completion=.25,max_new_tokens=128 if task=='textvqa' else 1024,
          top_p=None,num_beams=1,single_pred_prompt=task=='sqa')
        if params['alpha']!=.5 or params['beta']!=2. or params['visual_token_num']!=budget or params['anchorzip']!=(method=='AnchorZip') or params['temperature']!=0:
            raise RuntimeError('Original frozen method parameters differ')
        if task=='textvqa' and '--official-textvqa' not in argv:raise RuntimeError('Official TextVQA switch absent')
        if task=='sqa' and '--single-pred-prompt' not in argv:raise RuntimeError('Original SQA option guidance absent')
        specs[job_id]=dict(key=job_id,job_id=job_id,model='next',task=task,method=method,budget=budget,n=5000 if task=='textvqa' else 2017,
          parameters=params,original_generate_command=argv,source_and_input_sha256=dict(lane['source_sha256']),
          origin=origin,min_free_memory_mib=32000 if method=='FULL' else 22500,exclusive_gpu=method=='FULL')
    return specs

def image_manifest(spec,rows):
    path=OUT/('next_'+spec['task']+'_images.manifest.json')
    images=sorted(set(str(r['image']) for r in rows))
    if path.exists():
        manifest=json.loads(path.read_text())
        if manifest['image_names']!=images or manifest['question_file_sha256']!=sha(spec['parameters']['question_file']):raise RuntimeError('Existing image manifest differs')
    else:
        records=[]
        for image in images:
            image_path=Path(spec['parameters']['image_folder'])/image
            records.append(dict(image=image,path=str(image_path),sha256=sha(image_path)))
        manifest=dict(n=len(images),image_names=images,question_file_sha256=sha(spec['parameters']['question_file']),records=records)
        dump(path,manifest)
    return path
def verify_sources(protocol):
    if protocol['launcher_sha256']!=sha(__file__):raise RuntimeError('Continuation worker changed')
    for path,digest in protocol['source_and_input_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen source/input changed: '+path)
def verify_images(protocol):
    if sha(protocol['image_manifest'])!=protocol['image_manifest_sha256']:raise RuntimeError('Image manifest changed')
    for image in json.loads(Path(protocol['image_manifest']).read_text())['records']:
        if sha(image['path'])!=image['sha256']:raise RuntimeError('Image bytes changed: '+image['path'])

def prepare(job_id,attempt=1):
    OUT.mkdir(parents=True,exist_ok=True)
    spec=job_specs()[job_id];p=paths(job_id,attempt);rows=questions(spec)
    if len(rows)!=spec['n']:raise RuntimeError('Frozen question count changed')
    frozen=dict(spec['source_and_input_sha256'])
    for source in [LANE,UTIL,PATCH_HELPER,LLAVA/'llava/eval/m4c_evaluator.py',
      Path(spec['parameters']['model_path'])/'config.json',Path(spec['parameters']['model_path'])/'generation_config.json']:
        frozen[str(source)]=sha(source)
    if spec['origin'].get('native_protocol'):frozen[spec['origin']['native_protocol']]=spec['origin']['native_protocol_sha256']
    for path,digest in frozen.items():
        if sha(path)!=digest:raise RuntimeError('Original lane frozen identity changed: '+path)
    module,validation=utility().patch_module();compile(module,'continuation_wait_only','exec')
    image_path=image_manifest(spec,rows)
    argv=list(spec['original_generate_command'][1:])
    argv[argv.index('--answers-file')+1]=str(p['prediction'])
    protocol=dict(spec,protocol='NeXT_registered_wait_only_continuation',attempt=attempt,prepared_utc=now(),
      source_and_input_sha256=frozen,launcher_sha256=sha(__file__),AST_validation=validation,
      image_manifest=str(image_path),image_manifest_sha256=sha(image_path),native_runtime_argv=argv,
      output_paths={k:str(v) for k,v in p.items()},
      generate_command=[PYTHON,str(Path(__file__).resolve()),'--run','--job-id',job_id,'--attempt',str(attempt)],
      production_modified=False,pre_vision_GPU_hash_or_sync=False,index_trace=False,
      runtime_observation='Only original generate return/count/shape recorded after generation; native math/prompts unchanged')
    if p['control_protocol'].exists():
        existing=json.loads(p['control_protocol'].read_text());a=dict(existing);b=dict(protocol)
        a.pop('prepared_utc');b.pop('prepared_utc')
        if a!=b:raise RuntimeError('Preserve existing differing continuation protocol')
        return existing,p
    dump(p['control_protocol'],protocol);return protocol,p

def expected_sqa_prompt(row):
    text=row['conversations'][0]['value'].replace('<image>','').strip()
    if 'image' in row:text='<image>\n'+text
    return text+'\nAnswer with the option\'s letter from the given choices directly.'

def science_official_cpu(prediction):
    """Execute the original evaluator's parsing/count AST; omit its file writes."""
    evaluator=LLAVA/'llava/eval/eval_science_qa.py';tree=ast.parse(evaluator.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_pred_idx')
    main=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare))
    body=[]
    for node in main.body[1:]:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='print':break
        body.append(node)
    namespace={'json':json,'os':os,'re':__import__('re'),'random':__import__('random'),
      'args':SimpleNamespace(base_dir=str(SQA_BASE),split='test',result_file=str(prediction),options=list('ABCDE'))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn]+body,type_ignores=[])),'frozen_official_SQA_CPU','exec'),namespace)
    correct,total=namespace['correct'],namespace['total']
    image_correct,image_total=namespace['multimodal_correct'],namespace['multimodal_total']
    if total!=2017 or image_total!=2017:raise RuntimeError('SQA official image denominator changed')
    return dict(accuracy_percent=100*correct/total,n_correct=correct,n_total=total,n_image=image_total,
      image_accuracy_percent=100*image_correct/image_total,evaluator_sha256=sha(evaluator))

def text_official_cpu(rows):
    m4c=LLAVA/'llava/eval/m4c_evaluator.py'
    spec=importlib.util.spec_from_file_location('frozen_m4c_evaluator',m4c)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tree=ast.parse((LLAVA/'llava/eval/eval_textvqa.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prompt_processor')
    ns={'re':__import__('re')};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'frozen_official_TextVQA_processor','exec'),ns)
    annotations={(r['image_id'],r['question'].lower()):r for r in json.loads(TEXT_GT.read_text())['data']}
    pred=[dict(pred_answer=r['text'],gt_answers=annotations[(r['question_id'],ns['prompt_processor'](r['prompt']))]['answers']) for r in rows]
    return dict(accuracy_percent=100*module.TextVQAAccuracyEvaluator().eval_pred_list(pred),evaluator_sha256=sha(m4c))

def validate_artifacts(protocol,p):
    verify_sources(protocol);verify_images(protocol)
    inputs=questions(protocol)
    rows=[json.loads(s) for s in p['prediction'].open()];traces=[json.loads(s) for s in p['runtime'].open()]
    if len(rows)!=protocol['n'] or len(traces)!=protocol['n']:raise RuntimeError('Complete full question count required')
    expected=[(r['question_id'],r['text']) if protocol['task']=='textvqa' else (r['id'],expected_sqa_prompt(r)) for r in inputs]
    actual=[(r['question_id'],r['prompt']) for r in rows]
    if len(set(expected))!=protocol['n'] or actual!=expected:raise RuntimeError('Full ordered unique question/prompt identity changed')
    qdigest=sha(protocol['parameters']['question_file']);budget=protocol['budget']
    for position,(row,trace,q) in enumerate(zip(rows,traces,inputs)):
        if not isinstance(row['text'],str) or not row['text'].strip() or row['text'].strip().startswith('FAILED'):raise RuntimeError('Empty/FAILED output')
        qid,prompt=expected[position]
        if (trace['question_position']!=position or trace['question_id']!=qid or trace['image']!=q['image']
          or trace['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest() or trace['input_sha256']!=qdigest
          or trace['visual_token_budget_parameter']!=budget):raise RuntimeError('Runtime/input identity changed')
        shape=trace['image_tensor_shape'];count=trace['actual_visual_tokens_retained']
        if len(shape)!=5 or shape[0]!=1 or shape[2:]!=[3,336,336] or not isinstance(count,int) or count<1:
            raise RuntimeError('Invalid AnyRes runtime shape/count')
        if (budget==0 and count!=shape[1]*576) or (budget>0 and count>shape[1]*budget):raise RuntimeError('Actual budget exceeds nominal input geometry')
        if protocol['task']=='sqa':
            meta=row['metadata']
            if meta['actual_visual_tokens_retained']!=count or meta['visual_token_budget_parameter']!=budget or not meta['has_image']:
                raise RuntimeError('Native SQA returned count metadata mismatch')
    native=json.loads(p['native_protocol'].read_text())
    if protocol['task']=='textvqa':
        for name in ['model_path','question_file','image_folder','conv_mode','visual_token_num','alpha','beta','temperature','top_p','num_beams','max_new_tokens','anchorzip']:
            if native[name]!=protocol['parameters'][name]:raise RuntimeError('Native TextVQA parameter mismatch: '+name)
        if native['question_rows']!=5000 or native['question_file_sha256']!=qdigest or native['wrapper_sha256']!=sha(TEXT_WRAPPER):raise RuntimeError('Native TextVQA input/wrapper changed')
        calculated=text_official_cpu(rows)
    else:
        if native['parameters']!=protocol['parameters'] or native['question_file_sha256']!=qdigest or native['wrapper_sha256']!=sha(SCIENCE):raise RuntimeError('Native SQA sidecar identity mismatch')
        calculated=science_official_cpu(p['prediction'])
    return dict(calculated,success=True,n=protocol['n'],model='next',task=protocol['task'],method=protocol['method'],
      budget=budget,visual_token_num=budget,prediction_sha256=sha(p['prediction']),runtime_sha256=sha(p['runtime']),
      actual_count_distribution={str(k):v for k,v in Counter(t['actual_visual_tokens_retained'] for t in traces).items()},
      control_protocol_sha256=sha(p['control_protocol']),image_manifest_sha256=protocol['image_manifest_sha256'],
      integrity='Complete ordered unique full keys/prompts, runtime/count/geometry, source/input/images/native parameter identity, no empty/FAILED')

def validate_finished(job_id_or_protocol,attempt=1):
    """Read-only CPU validation returning score, frozen paths/hashes and marker."""
    if isinstance(job_id_or_protocol,str):
        p=paths(job_id_or_protocol,attempt);protocol=json.loads(p['control_protocol'].read_text())
    else:
        protocol=job_id_or_protocol;p=paths(protocol['job_id'],protocol['attempt'])
    if protocol['output_paths']!={k:str(v) for k,v in p.items()}:raise RuntimeError('Output path escapes fixed continuation directory')
    state=json.loads(p['state'].read_text());marker=json.loads(p['finished'].read_text())
    if (state.get('generation_exit_code')!=0 or not marker.get('success') or not marker.get('complete')
      or marker.get('n')!=protocol['n'] or marker.get('generation_exit_code')!=0):raise RuntimeError('Successful full generation/scoring gate absent')
    score=validate_artifacts(protocol,p);saved=json.loads(p['score'].read_text())
    if score!=saved or marker['score_sha256']!=sha(p['score']) or marker['prediction_sha256']!=score['prediction_sha256']:
        raise RuntimeError('Complete score/marker artifact hash mismatch')
    files={str(p[name]):sha(p[name]) for name in ['prediction','runtime','native_protocol','control_protocol','score','finished']}
    files.update(protocol['source_and_input_sha256']);files[str(Path(__file__).resolve())]=protocol['launcher_sha256']
    files[protocol['image_manifest']]=protocol['image_manifest_sha256']
    return dict(score=score,files=files,finished=marker,protocol=protocol)

def science_runtime_adapter(protocol,p):
    from llava.model import builder
    loader=builder.load_pretrained_model;rows=questions(protocol);position=0
    input_digest=sha(protocol['parameters']['question_file'])
    trace=p['runtime'].open('x')
    def traced_loader(*args,**kwargs):
        nonlocal position
        loaded=loader(*args,**kwargs);model=loaded[1];original=model.generate
        def generated(*gen_args,**gen_kwargs):
            nonlocal position
            returned=original(*gen_args,**gen_kwargs);row=rows[position];images=gen_kwargs.get('images')
            if not isinstance(returned,tuple) or len(returned)!=2:raise RuntimeError('Original generate tuple required')
            trace.write(json.dumps(dict(question_id=row['id'],image=row['image'],question_position=position,
              prompt_sha256=hashlib.sha256(expected_sqa_prompt(row).encode()).hexdigest(),input_sha256=input_digest,
              actual_visual_tokens_retained=int(returned[1]),visual_token_budget_parameter=protocol['budget'],
              image_tensor_shape=list(images.shape),image_sizes=gen_kwargs.get('image_sizes'),
              source='Original native science generate return observed after generation'))+'\n');trace.flush();position+=1
            return returned
        model.generate=generated;return loaded
    builder.load_pretrained_model=traced_loader
    try:
        runpy.run_path(str(SCIENCE),run_name='__main__')
        if position!=protocol['n']:raise RuntimeError('Incomplete native SQA runtime')
    finally:
        builder.load_pretrained_model=loader;trace.close()

def run(protocol,p):
    for name in ['prediction','runtime','native_protocol','state','score','finished']:
        if p[name].exists():raise FileExistsError('Preserve existing artifact: '+str(p[name]))
    util=utility();slot=util.safe_slot()
    if slot['free_memory_mib']<protocol['min_free_memory_mib'] or (protocol['exclusive_gpu'] and slot['other_gpu_models']):
        raise RuntimeError('Per-job exclusive/memory gate failed')
    verify_sources(protocol);verify_images(protocol)
    for name in ['USE_LLAVA_ARCH_CDPRUNER','USE_LLAVA_ARCH_DIVPRUNE','USE_LLAVA_ARCH_HIPRUNE','USE_LLAVA_ARCH_ABLATION']:
        if os.environ.get(name)=='1':raise RuntimeError('Unexpected alternate architecture')
    state=dict(status='generating',pid=os.getpid(),started_utc=now(),concurrency=slot,generation_exit_code=None)
    dump(p['state'],state);sys.path.insert(0,str(LLAVA))
    from llava.model.multimodal_encoder import clip_encoder
    original=clip_encoder.CLIPVisionTower.forward;module,validation=util.patch_module()
    if validation!=protocol['AST_validation']:raise RuntimeError('Registered wait-only AST changed')
    ns=dict(clip_encoder.__dict__);exec(compile(module,'continuation_wait_only','exec'),ns)
    clip_encoder.CLIPVisionTower.forward=ns['forward'];previous_argv=sys.argv;sys.argv=list(protocol['native_runtime_argv'])
    try:
        if protocol['task']=='textvqa':runpy.run_path(str(TEXT_RUNTIME),run_name='__main__')
        else:
            dump(p['native_protocol'],dict(protocol='shipped_CQM_A_image2017_control',parameters=protocol['parameters'],
              question_file_sha256=sha(protocol['parameters']['question_file']),wrapper_sha256=sha(SCIENCE),
              source='Frozen native SQA wrapper does not write a sidecar; launcher records its exact original argv/metadata',
              original_protocol_metadata=protocol['origin']['original_job']['protocol_metadata']))
            science_runtime_adapter(protocol,p)
    except BaseException as exc:
        state.update(status='failed',generation_exit_code=1,error=repr(exc));dump(p['state'],state);raise
    finally:
        sys.argv=previous_argv;clip_encoder.CLIPVisionTower.forward=original
    state.update(status='scoring',generation_exit_code=0,generation_finished_utc=now());dump(p['state'],state)
    score=validate_artifacts(protocol,p);dump(p['score'],score)
    dump(p['finished'],dict(success=True,complete=True,n=protocol['n'],generation_exit_code=0,finished_utc=now(),
      job_id=protocol['job_id'],score_sha256=sha(p['score']),prediction_sha256=score['prediction_sha256']))
    state.update(status='complete',complete=True,finished_utc=now());dump(p['state'],state)
    print(json.dumps(score,ensure_ascii=False,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--validate-finished',action='store_true');parser.add_argument('--job-id',choices=IDS,required=True);parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    if args.run and args.validate_finished:parser.error('Choose run or CPU validation')
    if args.validate_finished:print(json.dumps(validate_finished(args.job_id,args.attempt),ensure_ascii=False,indent=2));return
    protocol,p=prepare(args.job_id,args.attempt)
    if args.run:run(protocol,p)
    else:print(json.dumps(dict(CPU_only=True,job_id=args.job_id,control_protocol=str(p['control_protocol']),generate_command=protocol['generate_command'],n=protocol['n']),indent=2))

if __name__=='__main__':main()
