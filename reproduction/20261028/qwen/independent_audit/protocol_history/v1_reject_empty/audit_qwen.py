"""CPU-only independent input/provenance and complete-result checks for Qwen.

No model loading, generation, API use or writes to the original run directories.
Author metric functions are AST-extracted unchanged; reference metrics are
implemented independently. All new output stays in this audit directory.
"""
import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import string
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
QWEN = HERE.parent
ROOT = HERE.parents[3]
AUTHOR = ROOT / '_reproduction/author_eadp/Qwen_vl'
COUNTS = {'DocVQA_VAL':5349, 'AI2D_TEST':3088}
DEFAULT_PLAN = QWEN / 'public_fa2_full_run02_plan.json'
NATIVE_NA = {'', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
             '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL',
             'NaN', 'None', 'n/a', 'nan', 'null'}
csv.field_size_limit(64 * 1024 * 1024)


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def normalized_id(value):
    text=str(value)
    return str(int(text[:-2])) if re.fullmatch(r'\d+\.0',text) else text


def native_cell(value):
    return None if value is None or str(value) in NATIVE_NA else str(value)


def scoring_prediction(raw):
    # Preserve literal raw text; reproduce pd.read_excel -> str() only for scoring.
    return 'nan' if native_cell(raw) is None else raw


def parse_answers(value):
    if isinstance(value,list):
        return [str(v) for v in value]
    text=str(value)
    parsed=ast.literal_eval(text) if text.strip().startswith('[') else text
    return [str(v) for v in parsed] if isinstance(parsed,list) else [str(parsed)]


def author_helpers():
    namespace=dict(cp=copy,os=types.SimpleNamespace(environ={}),string=string,
                   get_logger=logging.getLogger)
    for relative,names in [
        ('vlmeval/dataset/utils/vqa_eval.py',{'levenshtein_distance','anls_compute'}),
        ('vlmeval/utils/matching_util.py',{'can_infer_option','can_infer_text','can_infer'})]:
        path=AUTHOR/'VLMEvalKit'/relative
        module=ast.parse(path.read_text())
        selected=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in selected}==names
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def independent_distance(a,b):
    """Two-row Wagner-Fischer implementation, independent of author code."""
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def independent_anls(prediction,ground_truths):
    scores=[]
    for truth in ground_truths:
        left=' '.join(truth.lower().split());right=' '.join(prediction.lower().split())
        # Preserve this released scorer's raw uppercase-length denominator.
        denominator=max(len(truth.upper()),len(prediction.upper()))
        similarity=1.-independent_distance(left,right)/denominator if denominator else 1.
        scores.append(similarity if similarity>=.5 else 0.)
    assert scores
    return max(scores)


def independent_option(prediction,choices):
    """Deterministic released exact-matching semantics, with no judge call."""
    if 'Failed to obtain answer via API' in prediction:
        return None
    for refusal in ["Sorry, I can't help with images of people yet.","I can't process this file.",
                    "I'm sorry, but without the image provided",'Cannot determine the answer']:
        if refusal in prediction:
            return 'Z'
    words=re.sub(r'[.()\[\],:;!*#{}]',' ',prediction).split()
    present=[letter for letter in choices if letter in words]
    if len(present)==1 and words.index(present[0])>len(words)-5:
        return present[0]
    if not present and 'Z' in words:
        return 'Z'
    text=prediction.lower()
    if len(text)>2*sum(len(str(v)) for v in choices.values()):
        return None
    matching=[k for k,v in choices.items() if str(v).lower() in text]
    return matching[0] if len(matching)==1 else None


def canonical_dataset(name, specification):
    target=HERE/(name+'.canonical.jsonl')
    metadata=HERE/(name+'.canonical_manifest.json')
    source=Path(specification['data_path']);stat=identity(source)
    if metadata.exists() and target.exists():
        record=json.loads(metadata.read_text())
        if record['source_stat']==stat and record['source_sha256']==specification['data_sha256']:
            assert sha(target)==record['canonical_sha256']
            return jsonl(target),record
    digest=hashlib.sha256();md5=hashlib.md5();canonical=[]
    with source.open('rb') as f:
        def physical_lines():
            for raw in f:
                digest.update(raw);md5.update(raw)
                yield raw.decode('utf-8')
        for row in csv.DictReader(physical_lines(),delimiter='\t'):
            canonical.append({k:v for k,v in row.items() if k!='image'})
    assert digest.hexdigest()==specification['data_sha256']
    assert identity(source)==stat
    assert len(canonical)==COUNTS[name]
    assert len({normalized_id(r['index']) for r in canonical})==COUNTS[name]
    with target.open('w') as f:
        for row in canonical:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    record=dict(source_stat=stat,source_sha256=digest.hexdigest(),source_md5=md5.hexdigest(),
                canonical_sha256=sha(target),n=len(canonical),
                scope='All TSV records retained except encoded image payload; source entire-byte SHA verified.')
    save(metadata,record)
    return canonical,record


def prompt_for(row,dataset):
    """Execute original prompt-only methods without constructing a model."""
    path=AUTHOR/'VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py'
    namespace={}
    exec(compile(path.read_text(),str(path),'exec'),namespace)
    cls=namespace['Qwen3VLPromptMixin'];obj=object.__new__(cls)
    obj.dump_image_func=lambda line:[str(Path('/media/disk2/YZX/LMUData/images')/dataset/line['image_path'])]
    line={k:(float('nan') if native_cell(v) is None else v) for k,v in row.items()}
    method=obj._build_vqa_prompt if dataset=='DocVQA_VAL' else obj._build_mcq_prompt
    return method(line,dataset)


def source_checks(plan):
    checked=[]
    for field in ['source_sha256','frozen_evidence']:
        for path,expected in plan[field].items():
            assert sha(path)==expected,('frozen input changed',path)
            checked.append(dict(path=path,sha256=expected))
    assert plan['author_commit']=='e1a08801461c181871b9ff1cb0803b8e9966f083'
    assert plan['recipe']['visual_token_num']==256 and plan['recipe']['alpha']==.5 and plan['recipe']['beta']==2.
    assert plan['recipe']['min_pixels']==plan['recipe']['max_pixels']==1048576
    return checked


def overlay_checks(plan):
    hook=QWEN/'import_overlay/sitecustomize.py';helper=hook.with_name('result_transfer.py')
    ns=dict(__file__=str(hook))
    before=list(sys.meta_path)
    try:
        exec(compile(hook.read_text(),str(hook),'exec'),ns)
        finder=sys.meta_path[0]
        assert finder.find_spec('vlmeval.utils.result_transfer').origin==str(helper)
        for name in ['vlmeval.config','model.pruner','vlmeval.inference',
                     'vlmeval.dataset.image_vqa','vlmeval.dataset.image_mcq',
                     'vlmeval.vlm.qwen3_vl.model_fixed_res']:
            assert finder.find_spec(name) is None
    finally:
        sys.meta_path[:]=before
    text=helper.read_text();tree=ast.parse(text)
    assert not any(isinstance(n,(ast.Assign,ast.AugAssign)) and isinstance(getattr(n,'value',None),ast.Call)
                   and getattr(getattr(n,'value',None).func,'id',None)=='exec' for n in tree.body)
    assert sha(helper)==plan['packaging_repair']['provenance']['sha256']
    independently_downloaded=HERE/'official_result_transfer.py'
    if independently_downloaded.exists():
        assert sha(independently_downloaded)==sha(helper)
    run=AUTHOR/'VLMEvalKit/run.py';rtree=ast.parse(run.read_text())
    calls=[]
    def inspect_calls(node,guards=()):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ['MMMU_result_transfer','MMTBench_result_transfer']:
            calls.append(dict(function=node.func.id,line=node.lineno,guards=list(guards)))
        for child in ast.iter_child_nodes(node):
            inspect_calls(child,guards+(ast.unparse(node.test),) if isinstance(node,ast.If) else guards)
    inspect_calls(rtree)
    assert len(calls)==2
    assert any("dataset_name in ['MMMU_TEST']" in c['guards'] for c in calls)
    assert any("'MMT-Bench_ALL' in dataset_name" in c['guards'] for c in calls)
    absent=subprocess.run(['git','cat-file','-e','HEAD:Qwen_vl/VLMEvalKit/vlmeval/utils/result_transfer.py'],
                          cwd=AUTHOR,stdout=subprocess.PIPE,stderr=subprocess.PIPE).returncode!=0
    assert absent
    return dict(hook_sha256=sha(hook),helper_sha256=sha(helper),helper_bytes=helper.stat().st_size,
        helper_absent_from_author_commit=absent,only_import_target='vlmeval.utils.result_transfer',
        exported_call_sites=calls,selected_datasets_call_neither=True,
        source='Pinned official VLMEvalKit helper, imported only to satisfy missing packaging dependency; no selected-task math change.')


def prepare(plan_path):
    plan=json.loads(plan_path.read_text());checked=source_checks(plan)
    datasets={name:canonical_dataset(name,spec)[0] for name,spec in plan['datasets'].items()}
    smoke=jsonl(QWEN/'author_fa2_smoke/answers.jsonl')
    state=json.loads((QWEN/'author_fa2_smoke/state.json').read_text())
    runtime=json.loads((QWEN/'author_fa2_smoke/runtime.json').read_text())
    pandas_semantics=json.loads((HERE/'pandas_native_semantics.json').read_text())
    assert set(pandas_semantics['STR_NA_VALUES'])==NATIVE_NA
    assert state['status']=='complete' and state['n']==len(smoke)==4
    assert sha(QWEN/'author_fa2_smoke/answers.jsonl')==state['answers_sha256']
    assert runtime['attention_backend']=='flash_attention_2' and runtime['model_dtype']=='torch.bfloat16'
    audited=[]
    for record in smoke:
        name=record['dataset'];position=record['position'];row=datasets[name][position]
        assert position in plan['datasets'][name]['positions']
        assert record['index']==normalized_id(row['index'])
        assert record['message']==prompt_for(row,name)
        assert isinstance(record['prediction'],str) and record['prediction'].strip()
        for path,expected in record['image_sha256'].items():assert sha(path)==expected
        audited.append(dict(dataset=name,position=position,index=record['index'],
            message_matches_source=True,image_sha256=record['image_sha256'],prediction_present=True))
    assert {(r['dataset'],r['position']) for r in smoke}=={(name,p) for name,s in plan['datasets'].items() for p in s['positions']}
    metric_sources={}
    for relative in ['dataset/image_vqa.py','dataset/image_mcq.py','dataset/utils/vqa_eval.py',
                     'dataset/utils/multiple_choice.py','utils/matching_util.py','smp/file.py']:
        path=AUTHOR/'VLMEvalKit/vlmeval'/relative
        archived=subprocess.check_output(['git','show','HEAD:Qwen_vl/VLMEvalKit/vlmeval/'+relative],cwd=AUTHOR)
        assert path.read_bytes()==archived
        metric_sources[str(path)]=sha(path)
    image_manifest_path=HERE/'current_cached_images.json'
    if image_manifest_path.exists():
        image_manifest=json.loads(image_manifest_path.read_text())
        for record in image_manifest:
            assert identity(record['path'])==record['stat']
    else:
        cache_audit=json.loads((QWEN/'dataset_cache_audit.json').read_text())
        image_manifest=[]
        for name,metadata in cache_audit['datasets'].items():
            used={str(Path('/media/disk2/YZX/LMUData/images')/name/r['image_path']) for r in datasets[name]}
            assert used=={r['image_path'] for r in metadata['images']}
            for record in metadata['images']:
                path=Path(record['image_path']);stat=identity(path)
                actual=sha(path)
                assert actual==record['cached_sha256']==record['author_regenerated_sha256']
                assert identity(path)==stat
                image_manifest.append(dict(dataset=name,path=str(path),stat=stat,sha256=actual))
        save(image_manifest_path,image_manifest)
    failure=QWEN/'public_fa2_full/outputs/logs/eadp_qwen3-8b_t256_a0.5_b2.0_DocVQA_VAL.log'
    report=dict(plan=dict(path=str(plan_path),sha256=sha(plan_path)),frozen_files_checked=checked,
        canonical_counts={k:len(v) for k,v in datasets.items()},smoke_records=audited,
        smoke_runtime=runtime,smoke_accuracy_computed=False,overlay=overlay_checks(plan),
        metric_source_sha256=metric_sources,pandas_native_semantics=pandas_semantics,
        current_image_checks=dict(n=len(image_manifest),bytes=sum(r['stat']['bytes'] for r in image_manifest),
            manifest=dict(path=str(image_manifest_path),sha256=sha(image_manifest_path))),
        first_attempt=dict(log=str(failure),sha256=sha(failure),
            failure='ModuleNotFoundError: vlmeval.utils.result_transfer before model/generation',
            failure_preserved=True),GPU_loaded_by_this_audit=False,
        limits='Four smoke records validate structure/runtime, not benchmark accuracy. Full results require complete dataset-specific verification.')
    save(HERE/'preparation_audit.json',report)
    save(HERE/(plan_path.stem+'.preparation_audit.json'),report)
    assert 'torch' not in sys.modules, 'CPU auditor must not load torch/model'
    return report


def read_xlsx(path):
    """Read ordinary VLMEvalKit XLSX values using only Python's standard library."""
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings=[''.join(x.itertext()) for x in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        table=[]
        with archive.open('xl/worksheets/sheet1.xml') as stream:
            for _,element in ET.iterparse(stream,events=['end']):
                if element.tag!='{'+ns['m']+'}row':continue
                row={}
                for cell in element.findall('m:c',ns):
                    column=re.match('[A-Z]+',cell.attrib['r']).group()
                    value=cell.find('m:v',ns);kind=cell.attrib.get('t')
                    if kind=='inlineStr':text=''.join(cell.find('m:is',ns).itertext())
                    elif kind=='s':text=strings[int(value.text)]
                    else:text=value.text if value is not None and value.text is not None else ''
                    row[column]=text
                table.append(row);element.clear()
    if not table:return []
    header=table[0]
    return [{name:row.get(col,'') for col,name in header.items()} for row in table[1:]]


def read_table(path):
    path=Path(path)
    if path.suffix=='.xlsx':return read_xlsx(path)
    if path.suffix in ['.csv','.tsv']:
        with path.open(newline='') as f:return list(csv.DictReader(f,delimiter='\t' if path.suffix=='.tsv' else ','))
    if path.suffix=='.json':
        data=json.loads(path.read_text());assert isinstance(data,list);return data
    raise ValueError('Unsupported raw prediction format: '+str(path))


def validate_predictions(canonical,predictions,expected_n):
    assert len(canonical)==len(predictions)==expected_n, 'Incomplete full dataset'
    ids=[normalized_id(r['index']) for r in canonical]
    assert len(set(ids))==expected_n
    assert [normalized_id(r['index']) for r in predictions]==ids,'Missing/duplicate/reordered indices'
    for position,(truth,pred) in enumerate(zip(canonical,predictions)):
        assert str(pred['question'])==truth['question'],('question mismatch',position)
        assert parse_answers(pred['answer'])==parse_answers(truth['answer']),('ground truth mismatch',position)
        for option in string.ascii_uppercase:
            if option in truth:assert native_cell(pred[option])==native_cell(truth[option]),('option mismatch',position,option)
        value=pred['prediction']
        assert isinstance(value,str) and value.strip(),('empty prediction',position)
        assert 'Failed to obtain answer via API' not in value,('failed prediction',position)
    return True


def verify(plan_path,run_dir):
    plan=json.loads(plan_path.read_text());source_checks(plan);helpers=author_helpers()
    preparation=json.loads((HERE/(plan_path.stem+'.preparation_audit.json')).read_text())
    assert preparation['plan']['sha256']==sha(plan_path)
    for path,expected in preparation['metric_source_sha256'].items():assert sha(path)==expected
    image_manifest=preparation['current_image_checks']['manifest']
    assert sha(image_manifest['path'])==image_manifest['sha256']
    for image in json.loads(Path(image_manifest['path']).read_text()):
        assert identity(image['path'])==image['stat']
    state=json.loads((run_dir/'state.json').read_text())
    assert state['plan_sha256']==sha(plan_path)
    assert Path(plan['environment']['OUTPUT_ROOT']).parent==run_dir
    results={};pending=[]
    for name,specification in plan['datasets'].items():
        canonical,_=canonical_dataset(name,specification)
        paths={p.resolve() for ext in ['xlsx','tsv','json'] for p in run_dir.rglob(plan['model_name']+'_'+name+'.'+ext)}
        if not paths:pending.append(name);continue
        assert len(paths)==1,('ambiguous raw files',name,paths)
        path=paths.pop();predictions=read_table(path)
        validate_predictions(canonical,predictions,COUNTS[name])
        per_question=[]
        for truth,prediction in zip(canonical,predictions):
            raw_text=prediction['prediction'];text=scoring_prediction(raw_text)
            if name=='DocVQA_VAL':
                gts=parse_answers(truth['answer'])
                distance=min(helpers['anls_compute'](g,text) for g in gts)
                author=max(0.,1.-distance) if 1.-distance>=.5 else 0.
                independent=independent_anls(text,gts)
                entry=dict(index=truth['index'],author_score=author,independent_score=independent,
                           literal_prediction_score=independent_anls(raw_text,gts))
            else:
                choices={k:native_cell(truth[k]) for k in string.ascii_uppercase if k in truth and native_cell(truth[k]) is not None}
                inferred=helpers['can_infer'](text,copy.deepcopy(choices)) or None
                independent_inferred=independent_option(text,choices)
                assert inferred==independent_inferred,('MCQ extractor disagreement',truth['index'],inferred,independent_inferred)
                author=float(inferred==truth['answer']);independent=float(independent_inferred==truth['answer'])
                entry=dict(index=truth['index'],author_option=inferred,independent_option=independent_inferred,
                           author_score=author,independent_score=independent,
                           strict_letter_score=float(raw_text.strip()==truth['answer']))
            entry.update(raw_prediction=raw_text,scored_prediction=text,
                         prediction_changed_by_native_na_loader=raw_text!=text)
            assert abs(author-independent)<1e-12,('metric disagreement',truth['index'])
            per_question.append(entry)
        score=100*sum(r['independent_score'] for r in per_question)/COUNTS[name]
        score_artifacts=[p for suffix in ['csv','xlsx','json'] for p in path.parent.glob(path.stem+'_acc.'+suffix)]
        author_reported=[]
        for artifact in score_artifacts:
            score_rows=read_table(artifact)
            for row in score_rows:
                if 'Overall' in row:
                    raw=float(row['Overall']);percent=raw if name=='DocVQA_VAL' else raw*100
                    assert abs(percent-score)<1e-7,('author score artifact mismatch',artifact,percent,score)
                    author_reported.append(dict(path=str(artifact),sha256=sha(artifact),raw_value=raw,value_percent=percent))
        destination=HERE/(run_dir.name+'_'+name+'.per_question.jsonl')
        destination.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in per_question))
        results[name]=dict(n=COUNTS[name],raw_predictions=dict(path=str(path),sha256=sha(path)),
            value_percent=score,metric='ANLS (%)' if name=='DocVQA_VAL' else 'Released deterministic exact-matching accuracy (%)',
            native_na_prediction_conversions=sum(r['prediction_changed_by_native_na_loader'] for r in per_question),
            prediction_loader='Native pandas default-NA conversion followed by str(); raw prediction is retained separately.',
            pointwise_author_reference_matches=COUNTS[name],author_score_artifacts=author_reported,
            author_score_status='artifact_verified' if author_reported else 'author_artifact_missing; isolated unchanged author helpers and independent CPU metric agree',
            per_question=dict(path=str(destination),sha256=sha(destination)),
            scope='Complete dataset, public recipe. AI2D score is deterministic exact matching without an API judge.' )
        if name=='DocVQA_VAL':
            results[name]['auxiliary_literal_prediction_ANLS_percent']=100*sum(r['literal_prediction_score'] for r in per_question)/COUNTS[name]
    logs=[]
    for path in run_dir.rglob('*.log'):
        lines=[s for s in path.read_text(errors='replace').splitlines() if any(x in s for x in
              ['Traceback','ModuleNotFoundError','exact matching','OPENAI API','ERROR','Exception'])]
        if lines:logs.append(dict(path=str(path),lines=lines[-30:]))
    report=dict(run=str(run_dir),plan_sha256=sha(plan_path),datasets=results,pending_datasets=pending,
        complete=not pending and len(results)==2,launcher_state=state,scoring_or_packaging_logs=logs,
        author_score_artifacts_complete=not pending and len(results)==2 and all(r['author_score_artifacts'] for r in results.values()),
        public_launcher_exited_successfully=state.get('launcher_exit_code')==0,
        auditor=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        generated_by_this_audit=False,GPU_loaded_by_this_audit=False,
        limits='Author helper scoring is executed separately on preserved raw predictions. API scoring failures/fallbacks remain visible; no generation retry or score-based parameter selection.')
    save(HERE/(run_dir.name+'.verification.json'),report)
    assert 'torch' not in sys.modules, 'CPU auditor must not load torch/model'
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','verify'])
    parser.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    parser.add_argument('--run-dir',type=Path,default=QWEN/'public_fa2_full_run02')
    args=parser.parse_args()
    report=prepare(args.plan) if args.action=='prepare' else verify(args.plan,args.run_dir)
    print(json.dumps({k:report[k] for k in (['canonical_counts','smoke_accuracy_computed','GPU_loaded_by_this_audit']
        if args.action=='prepare' else ['complete','pending_datasets','datasets'])},ensure_ascii=False,indent=2))
