"""Prepare/run predefined full TextVQA stream-wait controls independently.

--prepare (default) is CPU-only. --run is an explicit coordinator operation.
The original runtime adapter and native wrapper are executed with runpy;
only the checked three-node CLIP forward patch is installed in memory.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

OUT=Path(__file__).resolve().parent
ROOT=Path('/media/disk2/YZX/research/EADP_amp')
LLAVA=ROOT/'LLaVA'
ARCHIVE=LLAVA/'playground/data/eval/anchorzip_p3/next_textvqa_official_20261008'
BASE=ARCHIVE/'EGATHER_K32.jsonl.protocol.json'
AZBASE=ARCHIVE/'LRMAIN00625.jsonl.protocol.json'
RUNTIME=ROOT/'Qwen_vl/scripts/stage1_roundtrip_pilot/textvqa_runtime_launcher.py'
WRAPPER=RUNTIME.with_name('llava_eval_arm_model_vqa.py')
PANEL=OUT/'next_text_stream_panel.py'
PYTHON='/home/dell/miniconda3/envs/llava_pruner/bin/python'
GT=LLAVA/'playground/data/eval/textvqa/TextVQA_0.5.1_val.json'
ARMS={'EADP_beta2':dict(anchorzip=False,beta=2.),
      'AZ_beta2':dict(anchorzip=True,beta=2.),
      'EADP_beta1':dict(anchorzip=False,beta=1.)}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(tmp,path)
def load_panel():
    spec=importlib.util.spec_from_file_location('frozen_next_panel_util',PANEL)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def key(row):return row['question_id'],row.get('prompt',row.get('text'))


def paths(arm,attempt):
    stem='next_text_K32_'+arm+'_streamwait'+(f'_attempt{attempt}' if attempt!=1 else '')
    pred=OUT/(stem+'.jsonl')
    return dict(prediction=pred,runtime=Path(str(pred)+'.runtime.jsonl'),
        native_protocol=Path(str(pred)+'.protocol.json'),control_protocol=OUT/(stem+'.control.protocol.json'),
        score=OUT/(stem+'.official.score.json'),state=OUT/(stem+'.state.json'),
        finished=OUT/(stem+'.finished.json'),log=OUT/(stem+'.log'))


def image_manifest(base,questions):
    path=OUT/'next_text_K32_streamwait_images.manifest.json'
    unique=sorted(set(row['image'] for row in questions))
    if path.exists():
        manifest=json.loads(path.read_text())
        if manifest['question_file_sha256']!=base['question_file_sha256'] or manifest['image_names']!=unique:
            raise RuntimeError('Existing image manifest differs')
        return path,manifest
    records=[]
    for image in unique:
        image_path=Path(base['image_folder'])/image
        records.append(dict(image=image,path=str(image_path),size=image_path.stat().st_size,sha256=sha(image_path)))
    manifest=dict(image_names=unique,n=len(unique),question_file_sha256=base['question_file_sha256'],records=records)
    dump(path,manifest);return path,manifest


def verify_images(protocol):
    path=Path(protocol['image_manifest'])
    if sha(path)!=protocol['image_manifest_sha256']:raise RuntimeError('Image manifest modified')
    for image in json.loads(path.read_text())['records']:
        if sha(image['path'])!=image['sha256']:raise RuntimeError('Image bytes changed: '+image['path'])


def prepare(arm,attempt):
    OUT.mkdir(parents=True,exist_ok=True)
    base=json.loads(BASE.read_text());azbase=json.loads(AZBASE.read_text())
    questions=[json.loads(line) for line in Path(base['question_file']).open()]
    if (len(questions)!=5000 or base['question_rows']!=5000 or base['visual_token_num']!=32
        or base['alpha']!=.5 or base['beta']!=2. or base['max_new_tokens']!=128
        or base['temperature']!=0 or sha(base['question_file'])!=base['question_file_sha256']):
        raise RuntimeError('Frozen base configuration changed')
    params={k:base[k] for k in ['model_path','question_file','image_folder','conv_mode','alpha','visual_token_num','temperature','top_p','num_beams','max_new_tokens']}
    params.update(ARMS[arm])
    frozen={str((LLAVA/rel).resolve()):digest for rel,digest in azbase['source_sha256'].items()}
    frozen[str(WRAPPER)]=base['wrapper_sha256']
    for source in [BASE,AZBASE,RUNTIME,PANEL,GT,LLAVA/'llava/eval/eval_textvqa.py',
        LLAVA/'llava/eval/m4c_evaluator.py',Path(base['question_file']),
        Path(base['model_path'])/'config.json',Path(base['model_path'])/'generation_config.json']:
        frozen[str(source)]=sha(source)
    for file,digest in frozen.items():
        if sha(file)!=digest:raise RuntimeError('Frozen source/input modified: '+file)
    images_path,images=image_manifest(base,questions)
    utility=load_panel();module,validation=utility.patch_module()
    compile(module,'NeXT_full_wait_control','exec')
    p=paths(arm,attempt)
    argv=[str(RUNTIME),'--model-path',params['model_path'],'--question-file',params['question_file'],
        '--image-folder',params['image_folder'],'--answers-file',str(p['prediction']),
        '--conv-mode',params['conv_mode'],'--temperature','0','--visual_token_num','32',
        '--alpha','0.5','--beta',str(params['beta']),'--num-chunks','1','--chunk-idx','0',
        '--num_beams','1','--official-textvqa']
    if params['anchorzip']:argv.append('--anchorzip')
    protocol=dict(protocol='full_NeXT_TextVQA_K32_wait_only_control',prepared_utc=now(),arm=arm,attempt=attempt,
        question_count=5000,question_file_sha256=base['question_file_sha256'],parameters=params,
        source_and_input_sha256=frozen,launcher_sha256=sha(__file__),AST_validation=validation,
        image_manifest=str(images_path),image_manifest_sha256=sha(images_path),
        native_runtime_argv=argv,output_paths={k:str(v) for k,v in p.items()},
        generate_command=[PYTHON,str(Path(__file__).resolve()),'--run','--arm',arm,'--attempt',str(attempt)],
        GPU_not_authorized_by_prepare=True,production_modified=False,index_trace=False,
        interpretation='E/AZ beta2 share the identical wait-only repair; beta1 is a predefined published-script default sensitivity control, not a verified paper configuration',
        forbidden='No question changes, no parameter sweep, no conditional high-score selection')
    if p['control_protocol'].exists():
        old=json.loads(p['control_protocol'].read_text())
        expected=dict(protocol);previous=dict(old)
        expected.pop('prepared_utc');previous.pop('prepared_utc')
        if expected!=previous:raise RuntimeError('Existing immutable control protocol differs')
        return old,p
    dump(p['control_protocol'],protocol);return protocol,p


def validate_and_score(prediction,runtime,questions,parameters):
    """Exact official m4c scoring after strict composite-key/trace validation."""
    rows=[json.loads(line) for line in Path(prediction).open()]
    traces=[json.loads(line) for line in Path(runtime).open()]
    expected=[key(q) for q in questions]
    question_digest=sha(parameters['question_file'])
    wrapper_digest=sha(WRAPPER)
    if len(rows)!=5000 or len(traces)!=5000 or len(questions)!=5000:raise RuntimeError('Full 5000 rows required')
    if len(set(expected))!=5000 or [key(r) for r in rows]!=expected:raise RuntimeError('Prediction composite keys/order are incomplete, duplicate or changed')
    for position,(row,trace,q) in enumerate(zip(rows,traces,questions)):
        text=row['text']
        if not isinstance(text,str) or not text.strip() or text.strip().startswith('FAILED'):raise RuntimeError('Empty/FAILED prediction')
        if (trace['question_position']!=position or trace['question_id']!=q['question_id'] or trace['image']!=q['image']
            or trace['prompt_sha256']!=hashlib.sha256(q['text'].encode()).hexdigest()
            or trace['input_sha256']!=question_digest or trace['source_wrapper_sha256']!=wrapper_digest
            or trace['visual_token_budget_parameter']!=32
            or trace['image_tensor_shape']!=[1,5,3,336,336]
            or not 1<=trace['actual_visual_tokens_retained']<=160):
            raise RuntimeError('Invalid or changed runtime identity/budget at '+str(position))
    native=json.loads(Path(str(prediction)+'.protocol.json').read_text())
    for name in ['model_path','question_file','image_folder','conv_mode','visual_token_num','alpha','beta','temperature','top_p','num_beams','max_new_tokens','anchorzip']:
        if native[name]!=parameters[name]:raise RuntimeError('Native wrapper protocol mismatch: '+name)
    if (native['question_rows']!=5000 or native['question_file_sha256']!=question_digest
        or native['wrapper_sha256']!=wrapper_digest
        or native['score_text_transform']!='remove exact single-word-or-phrase answer suffix'
        or native['llm_prompt_transform']!='unchanged full TextVQA prompt, including OCR and answer suffix'):
        raise RuntimeError('Native input/wrapper/guidance identity changed')
    # Load only official evaluator source directly, avoiding any model loader.
    spec=importlib.util.spec_from_file_location('official_m4c',LLAVA/'llava/eval/m4c_evaluator.py')
    evaluator_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator_module)
    # The exact prompt processor is taken from its frozen official source AST.
    import ast
    tree=ast.parse((LLAVA/'llava/eval/eval_textvqa.py').read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='prompt_processor')
    namespace={'re':__import__('re')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'official_prompt_processor','exec'),namespace)
    prompt_processor=namespace['prompt_processor']
    annotations={(r['image_id'],r['question'].lower()):r for r in json.loads(GT.read_text())['data']}
    predictions=[dict(pred_answer=r['text'],gt_answers=annotations[(r['question_id'],prompt_processor(r['prompt']))]['answers']) for r in rows]
    accuracy=100*evaluator_module.TextVQAAccuracyEvaluator().eval_pred_list(predictions)
    return dict(success=True,n=5000,accuracy_percent=accuracy,prediction_sha256=sha(prediction),runtime_sha256=sha(runtime),
        actual_count_distribution=dict(Counter(t['actual_visual_tokens_retained'] for t in traces)),
        official_evaluator_sha256=sha(LLAVA/'llava/eval/m4c_evaluator.py'),GT_sha256=sha(GT),
        evaluator='Exact frozen TextVQAAccuracyEvaluator + exact frozen prompt_processor',
        integrity='All 5000 ordered composite keys, full prompts, aligned runtime/input/shape/budget, native protocol, nonempty/nonFAILED verified')


def score(protocol,p):
    if not p['state'].exists() or json.loads(p['state'].read_text()).get('generation_exit_code')!=0:
        raise RuntimeError('Scoring gate needs a recorded successful full generation')
    for file,digest in protocol['source_and_input_sha256'].items():
        if sha(file)!=digest:raise RuntimeError('Frozen source/input modified before score: '+file)
    verify_images(protocol)
    questions=[json.loads(line) for line in Path(protocol['parameters']['question_file']).open()]
    report=validate_and_score(p['prediction'],p['runtime'],questions,protocol['parameters'])
    report.update(arm=protocol['arm'],control_protocol=str(p['control_protocol']),control_protocol_sha256=sha(p['control_protocol']),
        image_manifest_sha256=protocol['image_manifest_sha256'])
    if p['score'].exists():
        if json.loads(p['score'].read_text())!=report:raise RuntimeError('Preserve existing differing score')
    else:dump(p['score'],report)
    marker=dict(success=True,complete=True,n=5000,generation_exit_code=0,finished_utc=now(),
        arm=protocol['arm'],prediction_sha256=report['prediction_sha256'],score=str(p['score']),score_sha256=sha(p['score']))
    if not p['finished'].exists():dump(p['finished'],marker)
    return report


def run(protocol,p):
    for name in ['prediction','runtime','native_protocol','state','score','finished']:
        if p[name].exists():raise FileExistsError('Preserve existing artifact: '+str(p[name]))
    utility=load_panel();slot=utility.safe_slot()
    for file,digest in protocol['source_and_input_sha256'].items():
        if sha(file)!=digest:raise RuntimeError('Frozen source/input modified before run: '+file)
    verify_images(protocol)
    for name in ['USE_LLAVA_ARCH_CDPRUNER','USE_LLAVA_ARCH_DIVPRUNE','USE_LLAVA_ARCH_HIPRUNE','USE_LLAVA_ARCH_ABLATION']:
        if os.environ.get(name)=='1':raise RuntimeError('Unexpected alternate architecture: '+name)
    state=dict(status='generating',started_utc=now(),pid=os.getpid(),concurrency=slot,
        control_protocol_sha256=sha(p['control_protocol']),generation_exit_code=None)
    dump(p['state'],state)
    sys.path.insert(0,str(LLAVA))
    from llava.model.multimodal_encoder import clip_encoder
    original=clip_encoder.CLIPVisionTower.forward
    module,validation=utility.patch_module()
    if validation!=protocol['AST_validation']:raise RuntimeError('Wait-only AST changed')
    namespace=dict(clip_encoder.__dict__)
    exec(compile(module,'full_NeXT_wait_control','exec'),namespace)
    clip_encoder.CLIPVisionTower.forward=namespace['forward']
    previous_argv=sys.argv
    sys.argv=list(protocol['native_runtime_argv'])
    try:
        runpy.run_path(str(RUNTIME),run_name='__main__')
    except BaseException as exc:
        state.update(status='failed',generation_exit_code=1,error=repr(exc),finished_utc=now());dump(p['state'],state)
        raise
    finally:
        sys.argv=previous_argv
        clip_encoder.CLIPVisionTower.forward=original
    state.update(status='scoring',generation_exit_code=0,generation_finished_utc=now());dump(p['state'],state)
    report=score(protocol,p)
    state.update(status='complete',complete=True,finished_utc=now(),score=str(p['score']));dump(p['state'],state)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--score-only',action='store_true')
    parser.add_argument('--arm',choices=ARMS,default='EADP_beta2')
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    if args.attempt<1:parser.error('Positive attempt required')
    if args.run and args.score_only:parser.error('Choose --run or --score-only')
    protocol,p=prepare(args.arm,args.attempt)
    if args.run:run(protocol,p)
    elif args.score_only:print(json.dumps(score(protocol,p),ensure_ascii=False,indent=2))
    else:print(json.dumps(dict(CPU_only=True,arm=args.arm,protocol=str(p['control_protocol']),
        generate_command=protocol['generate_command'],source_sha256=protocol['launcher_sha256'],question_count=5000),indent=2))


if __name__=='__main__':main()
