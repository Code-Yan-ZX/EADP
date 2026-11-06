"""Independently rebuild DEV task prompts and audited runtime image identities on CPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import string
import sys
from contract import read_jsonl, sha, validate_manifest

ROOT=Path('/media/disk2/YZX/research/EADP_v2')
PROMPT=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py'
DATA=Path('/media/disk2/YZX/LMUData')
TSVS={'DocVQA_VAL':DATA/'DocVQA_VAL.tsv','TextVQA_VAL':DATA/'TextVQA_VAL_local.tsv','AI2D_TEST':DATA/'AI2D_TEST.tsv'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest-dir',type=Path,required=True); p.add_argument('--inputs',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    panel=validate_manifest(a.manifest_dir); records=read_jsonl(a.inputs)
    if len(panel)!=len(records): raise ValueError('input count')
    summary=json.loads((a.manifest_dir/'panel_runtime_paths_summary.json').read_text())
    map_path=a.manifest_dir/'panel_runtime_paths.jsonl'
    if sha(map_path)!=summary['runtime_paths_sha256'] or sha(summary['localized_tsv'])!=summary['localized_tsv_sha256']:
        raise ValueError('audited runtime image mapping or localized TSV changed')
    for path,digest in summary['author_sources_sha256'].items():
        if sha(path)!=digest: raise ValueError('author input source changed')
    mapping={}
    for r in read_jsonl(map_path):
        if r['split']!='DEV': continue
        key=(r['task'],str(r['qid']))
        if key in mapping: raise ValueError('duplicate DEV runtime path')
        mapping[key]=r
    needed={(r['task'],str(r['qid'])) for r in panel}
    if set(mapping)!=needed: raise ValueError('runtime path and panel qids differ')
    import pandas as pd
    source={}
    for task,path in TSVS.items():
        # Match author smp.file.load's pandas defaults, but avoid loading GT/images.
        cols={'index','question','hint',*string.ascii_uppercase}
        table=pd.read_csv(path,sep='\t',usecols=lambda name:name in cols)
        for row in table.to_dict('records'):
            key=(task,str(row['index']))
            if key not in needed: continue
            if key in source: raise ValueError('duplicate source qid')
            source[key]={k:v for k,v in row.items() if k!='index'}
    if set(source)!=needed: raise ValueError('missing source question')
    text_original=pd.read_csv(DATA/'TextVQA_VAL.tsv',sep='\t',usecols=['index','question'])
    original_questions={('TextVQA_VAL',str(row['index'])):row['question'] for row in text_original.to_dict('records') if ('TextVQA_VAL',str(row['index'])) in needed}
    if len(original_questions)!=sum(key[0]=='TextVQA_VAL' for key in needed): raise ValueError('original TextVQA qid missing')
    if any(source[key]['question']!=question for key,question in original_questions.items()): raise ValueError('localized TextVQA question differs from original TSV')
    sys.path[:0]=[str(ROOT/'_reproduction/author_eadp/Qwen_vl'),str(ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit'),str(ROOT/'reproduction/20261028/qwen/import_overlay')]
    from vlmeval.dataset import DATASET_TYPE
    types={task:DATASET_TYPE(task,default=None) for task in TSVS}
    if types!={'DocVQA_VAL':'VQA','TextVQA_VAL':'VQA','AI2D_TEST':'MCQ'}: raise ValueError('unexpected author dataset type')
    spec=importlib.util.spec_from_file_location('author_prompt_independent',PROMPT)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    prompt=mod.Qwen3VLPromptMixin(use_custom_prompt=True)
    prompt.set_dump_image(lambda line:[line['image_path']])
    for p_row, actual in zip(panel,records):
        key=(p_row['task'],str(p_row['qid'])); mapped=mapping[key]
        if any(str(actual[k])!=str(p_row[k]) for k in ('task','qid','group_id')): raise ValueError('panel order or identity mismatch')
        if mapped['status']!='found' or mapped['group_id']!=p_row['group_id'] or actual['image_path']!=mapped['expected_path']:
            raise ValueError('image mapping mismatch')
        if actual['image_sha256']!=mapped['cache_byte_sha256'] or sha(actual['image_path'])!=mapped['cache_byte_sha256']:
            raise ValueError('image SHA mismatch')
        line={**source[key],'image_path':mapped['expected_path']}
        expected=prompt.build_prompt(line, dataset=key[0])
        if expected!=actual['message']: raise ValueError('author task prompt mismatch')
        digest=hashlib.sha256(json.dumps(expected,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if digest!=actual['message_sha256']: raise ValueError('message digest mismatch')
        if any(k in actual for k in ('answer','answers','prediction','ground_truth')): raise ValueError('GT in inputs')
    proof={'status':'passed','n_verified':len(records),'inputs_sha256':sha(a.inputs),
           'panel_sha256':sha(a.manifest_dir/'panel_manifest.jsonl'),'mapping_sha256':sha(map_path),
           'path_summary_sha256':sha(a.manifest_dir/'panel_runtime_paths_summary.json'),
           'localized_tsv_sha256':sha(TSVS['TextVQA_VAL']),'author_prompt_sha256':sha(PROMPT),
           'original_textvqa_sha256':sha(DATA/'TextVQA_VAL.tsv'), 'original_textvqa_verified_questions':len(original_questions),
           'verifier_sha256':sha(__file__),'dataset_types':types,'gt_read':False}
    with a.out.open('x') as f: json.dump(proof,f,indent=2); f.write('\n')
    print(json.dumps(proof))

if __name__=='__main__': main()
