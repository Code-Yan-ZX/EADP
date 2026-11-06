#!/usr/bin/env python3
"""Independent split/inventory checks; no answers, predictions or GPU."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = '20261104'
TASKS = ['DocVQA_VAL', 'TextVQA_VAL', 'AI2D_TEST']

def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def load(name):
    with (HERE/name).open(encoding='utf8') as f: return [json.loads(line) for line in f]

def main():
    meta = json.loads((HERE/'manifest_summary.json').read_text())
    assert hashlib.sha256((HERE/'build.py').read_bytes()).hexdigest() == meta['builder_sha256']
    for name, expected in meta['outputs_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected
    rows, panel, images, edges = map(load, ['full_manifest.jsonl', 'panel_manifest.jsonl', 'images.jsonl', 'near_pairs.jsonl'])
    assert all(i['index'] == n for n,i in enumerate(images))
    groups = defaultdict(set)
    for row in rows:
        assert row['group_id'] == images[row['image_index']]['group_id']
        groups[row['group_id']].add(row['split'])
    assert all(len(v)==1 for v in groups.values())
    qids = defaultdict(set)
    for row in rows:
        assert row['qid'] not in qids[row['task']]
        qids[row['task']].add(row['qid'])
    csv.field_size_limit(100_000_000)
    source_counts, document_missing = {}, Counter()
    for task in TASKS:
        path = Path(meta['sources'][task]['path'])
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
        assert h.hexdigest() == meta['sources'][task]['sha256']
        with path.open(encoding='utf8',newline='') as f:
            source_ids = []
            for row in csv.DictReader(f,delimiter='\t'):
                source_ids.append(row['index'])
                if task == 'DocVQA_VAL':
                    for field in ('ucsf_document_id','ucsf_document_page_no','docId','image_path'):
                        if row.get(field,'').strip().lower() in ('','nan','na','none','null'):
                            document_missing[field] += 1
        assert len(source_ids)==len(set(source_ids)) and set(source_ids)==qids[task]
        source_counts[task] = len(source_ids)
    selected = set()
    expected_panel = []
    for task in TASKS:
        choices = defaultdict(list)
        for row in rows:
            if row['task'] == task and row['group_id'] not in selected:
                choices[row['group_id']].append(row)
        ordered = sorted(choices, key=lambda g:(digest(f'{SEED}|{g}'),g))
        for name, subset in [('DEV',ordered[:128]),('CONFIRM',ordered[128:384])]:
            for gid in subset:
                row = min(choices[gid],key=lambda r:(digest(f"{SEED}|{r['qid']}"),r['qid']))
                expected_panel.append({'task':task,'qid':row['qid'],'group_id':gid,'split':name})
                assert groups[gid] == {name}
        selected.update(ordered[:384])
    assert expected_panel == panel
    assert all(v == {'FINAL'} for gid,v in groups.items() if gid not in selected)
    near = {(e['a'],e['b']) for e in edges}
    assert len(near)==len(edges)
    for e in edges:
        a,b = images[e['a']],images[e['b']]
        assert (int(a['phash'],16)^int(b['phash'],16)).bit_count()==e['distance']<=4
        assert a['group_id']==b['group_id']
    # Exhaustive comparison for a deterministic spread of 256 reference images.
    examined = sorted(range(len(images)),key=lambda i:digest(images[i]['byte_sha256']))[:256]
    found = 0
    for i in examined:
        v = int(images[i]['phash'],16)
        for j, image in enumerate(images):
            if i!=j and (v^int(image['phash'],16)).bit_count()<=4:
                found += 1
                assert tuple(sorted((i,j))) in near
    result = {'status':'pass','builder_sha256':meta['builder_sha256'],
              'verifier_sha256':hashlib.sha256((HERE/'verify.py').read_bytes()).hexdigest(), 'source_question_counts':source_counts,
              'document_missing_identifier_counts':{k:document_missing[k] for k in ('ucsf_document_id','ucsf_document_page_no','docId','image_path')},
              'panel_counts':dict(Counter(f"{p['task']}:{p['split']}" for p in panel)),
              'groups':len(groups),'audited_phash_queries':len(examined),'audited_close_pairs_directed':found,
              'checked': ['source and output SHA256','source qid inventory','no group split leakage',
                          'deterministic task-priority panel reconstruction','near-edge distances',
                          'brute force pHash neighbor completeness for deterministic 256-image sample']}
    (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
