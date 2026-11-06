#!/usr/bin/env python3
"""Independent reconstruction of conservative anchor-group exclusions."""
import collections
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=HERE.parent
ANCHOR=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/full1024_two_path_panel/plan.json')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def rows(p):
    with p.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def main():
    m=json.loads((HERE/'summary.json').read_text())
    assert sha(ANCHOR)==m['anchor_plan_sha256']
    assert sha(DATA/'full_manifest.jsonl')==m['full_manifest_sha256']
    assert sha(DATA/'panel_manifest.jsonl')==m['panel_manifest_sha256']
    assert sha(HERE/'build.py')==m['builder_sha256']
    for name,digest in m['outputs_sha256'].items():assert sha(HERE/name)==digest
    source=rows(DATA/'full_manifest.jsonl'); panel=rows(DATA/'panel_manifest.jsonl')
    anchor_ids=[str(x['index']) for x in json.loads(ANCHOR.read_text())['panel']]
    matches=[r for r in source if r['task']=='DocVQA_VAL' and r['qid'] in anchor_ids]
    assert len(matches)==len(set(anchor_ids))==16
    groups={r['group_id'] for r in matches}
    assert {(r['task'],r['qid'],r['group_id'],r['split']) for r in matches}=={(r['task'],r['qid'],r['group_id'],r['split']) for r in rows(HERE/'anchor_groups.jsonl')}
    expected_dropped=[r for r in source if r['split'] in ('CONFIRM','FINAL') and r['group_id'] in groups]
    expected_kept=[r for r in source if r['split'] in ('CONFIRM','FINAL') and r['group_id'] not in groups]
    assert rows(HERE/'excluded_questions.jsonl')==expected_dropped
    assert rows(HERE/'inference_questions.jsonl')==expected_kept
    assert rows(HERE/'confirm_panel_excluded.jsonl')==[r for r in panel if r['split']=='CONFIRM' and r['group_id'] in groups]
    assert rows(HERE/'confirm_panel_supplement.jsonl')==[r for r in panel if r['split']=='CONFIRM' and r['group_id'] not in groups]
    assert not ({r['group_id'] for r in expected_kept}&groups)
    # Recount groups without relying on builder aggregation.
    pairs={(r['task'],r['split'],r['group_id']) for r in expected_dropped}
    independent_counts=collections.Counter(f'{task}:{split}' for task,split,_ in pairs)
    assert dict(independent_counts)==m['excluded_group_counts']
    result={'status':'pass','anchor_distinct_groups':len(groups),'excluded_group_counts':dict(independent_counts),
            'excluded_questions':len(expected_dropped),'retained_questions':len(expected_kept),
            'summary_sha256':sha(HERE/'summary.json'),'verifier_sha256':sha(Path(__file__)),
            'checks':['independent original-source SHA and anchor qid reconstruction','full ordered retained/dropped question lists',
                      'all anchor-linked groups excluded globally','CONFIRM panel partition','independent group count'],
            'predictions_or_answers_read':False}
    (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
