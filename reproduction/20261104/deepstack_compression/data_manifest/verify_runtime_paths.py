#!/usr/bin/env python3
"""Independent byte-level verification of frozen panel runtime paths."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/media/disk2/YZX/LMUData')
csv.field_size_limit(100_000_000)
def hfile(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def rows(name):
    with (HERE/name).open(encoding='utf8') as f:return [json.loads(line) for line in f]
def main():
    summary=json.loads((HERE/'panel_runtime_paths_summary.json').read_text())
    assert hfile(HERE/'panel_runtime_paths.jsonl')==summary['runtime_paths_sha256']
    assert hfile(ROOT/'TextVQA_VAL_local.tsv')==summary['localized_tsv_sha256']
    assert hfile(ROOT/'TextVQA_VAL.tsv')==summary['source_tsv_sha256']
    panel=rows('panel_manifest.jsonl'); paths=rows('panel_runtime_paths.jsonl')
    assert len(paths)==len(panel)==1152
    source={}
    with (ROOT/'TextVQA_VAL.tsv').open(encoding='utf8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            value=row['image']; source[row['index']]=value if len(value)<=64 and value.isdecimal() else row['index']
    with (ROOT/'TextVQA_VAL_local.tsv').open(encoding='utf8',newline='') as f:
        local={r['index']:r['image_path'] for r in csv.DictReader(f,delimiter='\t')}
    assert len(source)==len(local)==5000 and set(source)==set(local)
    for qid,reference in source.items():
        assert Path(local[qid])==ROOT/'images'/'TextVQA_VAL'/(reference+'.jpg')
    for row,p in zip(paths,panel):
        assert all(row[k]==p[k] for k in ('task','qid','group_id','split'))
        assert row['status']=='found'
        if row['task']=='TextVQA_VAL':assert row['expected_path']==local[row['qid']]
        assert hfile(Path(row['expected_path']))==row['cache_byte_sha256']
    result={'status':'pass','source_reference_rules_checked':len(source), 'panel_byte_hashes_checked':len(paths),
            'runtime_paths_sha256':summary['runtime_paths_sha256'], 'verifier_sha256':hfile(Path(__file__)),
            'limits':['Checks cached file bytes against the path manifest; does not re-decode pixels or run the model.']}
    (HERE/'panel_runtime_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
