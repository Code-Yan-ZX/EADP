"""Build DEV messages from frozen panel, audited cache mapping and author prompt code."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import string
from contract import TASKS, read_jsonl, sha, validate_manifest

ROOT = Path('/media/disk2/YZX/research/EADP_v2')
PROMPT = ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/prompt.py'
DATA = Path('/media/disk2/YZX/LMUData')


def load_author_prompt():
    spec=importlib.util.spec_from_file_location('author_qwen3_prompt', PROMPT)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    obj=mod.Qwen3VLPromptMixin(use_custom_prompt=True)
    obj.set_dump_image(lambda row: [row['image_path']])
    return obj


def load_selected_rows(panel):
    import pandas as pd
    wanted = {task:{str(r['qid']) for r in panel if r['task']==task} for task in TASKS}
    result = {}
    paths = {'DocVQA_VAL': DATA/'DocVQA_VAL.tsv', 'TextVQA_VAL': DATA/'TextVQA_VAL_local.tsv', 'AI2D_TEST': DATA/'AI2D_TEST.tsv'}
    for task, path in paths.items():
        # Same pandas NA/type semantics as author smp.file.load; omit image/GT columns.
        cols={'index','question','hint',*string.ascii_uppercase}
        table=pd.read_csv(path, sep='\t', usecols=lambda name:name in cols)
        for row in table.to_dict('records'):
            qid=str(row['index'])
            if qid not in wanted[task]: continue
            if (task,qid) in result: raise ValueError('duplicate source qid')
            clean={k:v for k,v in row.items() if k!='index'}
            result[task,qid]=clean
        if sum(t==task for t,_ in result) != len(wanted[task]): raise ValueError(f'missing source rows for {task}')
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest-dir',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); panel=validate_manifest(a.manifest_dir)
    mapping=a.manifest_dir/'panel_runtime_paths.jsonl'; summ=a.manifest_dir/'panel_runtime_paths_summary.json'
    m=json.loads(summ.read_text())
    if sha(mapping)!=m['runtime_paths_sha256'] or sha(m['localized_tsv'])!=m['localized_tsv_sha256']:
        raise ValueError('runtime path mapping or localized TSV changed')
    for path,digest in m['author_sources_sha256'].items():
        if sha(path)!=digest: raise ValueError('author path source changed: '+path)
    mapped={(r['task'],str(r['qid'])):r for r in read_jsonl(mapping) if r['split']=='DEV'}
    if len(mapped)!=len(panel): raise ValueError('missing or repeated DEV runtime paths')
    rows=load_selected_rows(panel); prompt=load_author_prompt(); result=[]
    for x in panel:
        key=(x['task'],str(x['qid'])); entry=mapped[key]
        if entry['status']!='found' or entry['group_id']!=x['group_id'] or sha(entry['expected_path'])!=entry['cache_byte_sha256']:
            raise ValueError('runtime path or image identity mismatch: '+str(key))
        line=dict(rows[key],image_path=entry['expected_path'])
        msg=(prompt._build_mcq_prompt(line,x['task']) if x['task']=='AI2D_TEST' else prompt._build_vqa_prompt(line,x['task']))
        result.append(dict(task=x['task'],qid=str(x['qid']),group_id=x['group_id'], image_path=entry['expected_path'],
                           image_sha256=entry['cache_byte_sha256'], message=msg,
                           message_sha256=hashlib.sha256(json.dumps(msg,ensure_ascii=False,sort_keys=True).encode()).hexdigest()))
    a.out.mkdir(parents=True,exist_ok=False)
    out=a.out/'inputs.jsonl'
    with out.open('x',encoding='utf8') as f:
        for r in result: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    identity={'inputs_sha256':sha(out),'panel_sha256':sha(a.manifest_dir/'panel_manifest.jsonl'),
              'path_mapping_sha256':sha(mapping),'path_summary_sha256':sha(summ),'prompt_sha256':sha(PROMPT),
              'prepare_sha256':sha(__file__),'n':len(result), 'gt_read':False}
    (a.out/'prepare.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity))

if __name__=='__main__': main()
