"""Full4377 CPU verification, preserving the old metric and adding public scoring."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import string
import sys

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def adapted_core():
    path=BASE/'verify_completed.py';original=ast.parse(path.read_text());tree=copy.deepcopy(original)
    out=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OUT' for t in n.targets))
    old_out=copy.deepcopy(out.value);out.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(OUT))],keywords=[])
    changes={"observation['returned_visual_token_num'] == 128":576,
             "loaded['visual_token_num'] == 128":0,
             "paper['method'] == 'EADP'":'FULL',
             "paper['budget_nominal'] == 128":576}
    edits=[]
    # The paper assertions use a chained `and`, each comparison remains its own
    # AST node. Only FULL expectations change; every metric operation is kept.
    for n in ast.walk(tree):
        if isinstance(n,ast.Compare) and ast.unparse(n) in changes:
            old=copy.deepcopy(n.comparators[0]);new=changes[ast.unparse(n)]
            edits.append((n,old));n.comparators[0]=ast.Constant(new)
    assert len(edits)==4
    restored=copy.deepcopy(tree)
    restored_out=next(n for n in restored.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OUT' for t in n.targets));restored_out.value=old_out
    inverse={"observation['returned_visual_token_num'] == 576":128,
             "loaded['visual_token_num'] == 0":128,
             "paper['method'] == 'FULL'":'EADP',
             "paper['budget_nominal'] == 576":128}
    count=0
    for n in ast.walk(restored):
        if isinstance(n,ast.Compare) and ast.unparse(n) in inverse:
            n.comparators[0]=ast.Constant(inverse[ast.unparse(n)]);count+=1
    assert count==4 and ast.dump(restored)==ast.dump(original)
    ast.fix_missing_locations(tree);compile(tree,str(path)+':FULL_expectations_only','exec')
    return path,tree


def main():
    path,tree=adapted_core()
    exec(compile(tree,str(path)+':FULL_expectations_only','exec'),{'__file__':str(path),'__name__':'__main__'})
    import pandas as pd
    from transformers import AutoTokenizer
    protocol=json.loads((OUT/'protocol.json').read_text())
    report=json.loads((OUT/'independent_verification.json').read_text())
    assert report['success'] and report['n_rows']==4377 and report['n_circular_groups']==1176
    loaded=json.loads((OUT/'observed/loaded_model.json').read_text())
    assert loaded['actual_encode_images_source']==protocol['native_FULL_bypass']['source']
    assert loaded['actual_encode_images_source_sha256']==protocol['native_FULL_bypass']['source_sha256']
    patch=json.loads((OUT/'runtime_patch_validation.json').read_text())
    assert patch['applied_in_this_process'] and patch['restored_ast_identical'] and patch['exact_added_statement_count']==3
    assert patch['author_source_after_patch_sha256']==patch['source_sha256']
    runtime=[json.loads(line) for line in (OUT/'observed/runtime.jsonl').read_text().splitlines()]
    references=json.loads((OUT/'cpu_llm_tokens_current.json').read_text())['records']
    prediction=Path(protocol['command'][protocol['command'].index('--answers-file')+1])
    answers=[json.loads(line) for line in prediction.read_text().splitlines()]
    assert len(runtime)==len(references)==len(answers)==4377
    model=protocol['command'][protocol['command'].index('--model-path')+1]
    tokenizer=AutoTokenizer.from_pretrained(model,use_fast=False,local_files_only=True)
    config=json.loads((Path(model)/'config.json').read_text())
    if config.get('mm_use_im_patch_token',True):tokenizer.add_tokens(['<im_patch>'],special_tokens=True)
    if config.get('mm_use_im_start_end',False):tokenizer.add_tokens(['<im_start>','<im_end>'],special_tokens=True)
    assert len(tokenizer)==loaded['tokenizer_length']
    for obs,ref,answer in zip(runtime,references,answers):
        assert obs['encode_calls']==1 and obs['actual_mask_shape']==[1,576]
        assert obs['actual_mask_all_true'] and obs['actual_retained_tokens']==576
        assert obs['projected_features_shape']==[1,576,4096] and obs['projected_features_finite']
        assert obs['input_ids_sha256']==ref['token_ids_sha256']
        decoded=tokenizer.batch_decode(obs['generated_token_ids'],skip_special_tokens=True)[0].strip()
        assert decoded==answer['text'],answer['question_id']
    spec=importlib.util.spec_from_file_location('frozen_public_MMB_helpers',BASE/'public_mcq_metric_check.py')
    public=importlib.util.module_from_spec(spec);spec.loader.exec_module(public)
    namespace=dict(cp=copy,os=type('OfflineOS',(),{'environ':{}})(),re=re,string=string,pd=pd)
    sources=[public.original_helpers(public.UTILS/'utils/matching_util.py',['can_infer_option','can_infer_text','can_infer'],namespace),
             public.original_helpers(public.UTILS/'dataset/utils/multiple_choice.py',['build_choices','prefetch_answer','prefetch_circular_group'],namespace)]
    data=pd.read_table(protocol['dataset']['path']);data['prediction']=[a['text'] for a in answers]
    data['GT']=data['answer'];data['g_index']=[int(i%1000000) for i in data['index']]
    public_changes=[]
    for _,row in data.iterrows():
        parsed=namespace['prefetch_answer'](row);local=re.findall(r'\b[A-E]\b',row['prediction'].strip())
        local=local[0] if local else None
        if parsed!=local:public_changes.append(dict(index=int(row['index']),public=parsed,local=local))
    hits={};unresolved=[]
    for idx,group in data.groupby('g_index',sort=True):
        result=namespace['prefetch_circular_group'](group)
        if result is None:unresolved.append(int(idx));hit=0
        else:hit=result['hit']
        assert hit in (0,1);hits[int(idx)]=hit
    assert len(hits)==1176
    value=100*sum(hits.values())/1176
    report.update(current_method='FULL',native_cli_visual_token_num=0,actual_retained_tokens_per_row=576,
                  mask_observation_checks=4377,generated_token_decoding_checks=4377,complete_input_token_SHA_checks=4377,
                  native_FULL_bypass=protocol['native_FULL_bypass'],controlled_stream_patch=patch,
                  local_exact_letter_accuracy_percent=report['circular_accuracy_percent'],
                  released_public_circular_accuracy_percent=value,public_circular_correct=sum(hits.values()),
                  public_parser_changed_rows=public_changes,public_unresolved_groups=unresolved,
                  planned_primary_metric='released_public_exact_matching_circular',primary_value_percent=value,
                  primary_minus_paper_FULL_pp=value-64.7,unchanged_public_helper_sources=sources,
                  verifier_adapter=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__),
                      only_changes=['OUT directory','Observed retained128->576','Loaded nativeK128->0','Paper method EADP->FULL and nominal128->576'],
                      restored_core_AST_identical=True,metric_math_changed=False),
                  comparison_boundary=protocol['limits'])
    (OUT/'independent_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    # CPU tokenizer/transformers imports may import torch, but never a model or CUDA.
    if 'torch' in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
    print(json.dumps(dict(success=True,n_rows=4377,n_groups=1176,primary_public=value,
                          preserved_local=report['circular_accuracy_percent'],actual_retained_tokens=576)))


if __name__=='__main__':main()
