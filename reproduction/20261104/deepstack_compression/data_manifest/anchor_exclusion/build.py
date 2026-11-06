#!/usr/bin/env python3
"""Produce additional inference lists excluding all fixed E0 anchor groups."""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
DATA=HERE.parent
ANCHOR=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261028/qwen/full1024_two_path_panel/plan.json')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(n):
    with (DATA/n).open(encoding='utf8') as f:return [json.loads(line) for line in f]
def write(n, rows):
    with (HERE/n).open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')

def main():
    anchors=json.loads(ANCHOR.read_text())['panel']
    assert len(anchors)==16
    full=read('full_manifest.jsonl'); panel=read('panel_manifest.jsonl')
    index={(r['task'],r['qid']):r for r in full}
    assert len(index)==len(full)
    anchor_rows=[]
    for a in anchors:
        r=index[('DocVQA_VAL',str(a['index']))]
        anchor_rows.append({'task':r['task'],'qid':r['qid'],'group_id':r['group_id'],'split':r['split']})
    assert len({x['qid'] for x in anchor_rows})==16
    excluded={r['group_id'] for r in anchor_rows}
    affected=[r for r in full if r['split'] in ('CONFIRM','FINAL') and r['group_id'] in excluded]
    kept=[r for r in full if r['split'] in ('CONFIRM','FINAL') and r['group_id'] not in excluded]
    panel_affected=[r for r in panel if r['split']=='CONFIRM' and r['group_id'] in excluded]
    panel_kept=[r for r in panel if r['split']=='CONFIRM' and r['group_id'] not in excluded]
    write('anchor_groups.jsonl',anchor_rows)
    write('excluded_questions.jsonl',affected)
    write('inference_questions.jsonl',kept)
    write('confirm_panel_excluded.jsonl',panel_affected)
    write('confirm_panel_supplement.jsonl',panel_kept)
    counts=lambda seq:dict(sorted(Counter(f"{r['task']}:{r['split']}" for r in seq).items()))
    groups=lambda seq:dict(sorted(Counter(f'{task}:{split}' for task,split,gid in {(r['task'],r['split'],r['group_id']) for r in seq}).items()))
    outputs=['anchor_groups.jsonl','excluded_questions.jsonl','inference_questions.jsonl','confirm_panel_excluded.jsonl','confirm_panel_supplement.jsonl']
    summary={'status':'completed','anchor_plan_sha256':sha(ANCHOR),'full_manifest_sha256':sha(DATA/'full_manifest.jsonl'),
             'panel_manifest_sha256':sha(DATA/'panel_manifest.jsonl'),'builder_sha256':sha(Path(__file__)),
             'anchor_question_counts':counts(anchor_rows),'anchor_distinct_groups':len(excluded),
             'excluded_question_counts':counts(affected),'excluded_group_counts':groups(affected),
             'retained_question_counts':counts(kept),'retained_group_counts':groups(kept),
             'confirm_panel_excluded_counts':counts(panel_affected),'confirm_panel_retained_counts':counts(panel_kept),
             'outputs_sha256':{n:sha(HERE/n) for n in outputs},
             'policy':'Original split untouched; exclude all anchor-linked groups globally from CONFIRM/FINAL supplemental lists. Anchor outputs were used for identity diagnostics without GT/score; do not call affected groups never generated.',
             'limitations':['This is a conservative sensitivity analysis alongside original split; datasets were historically exposed.',
                            'CONFIRM requires frozen candidate/controls before screening; FINAL requires final candidate/reference lock and one formal evaluation.',
                            'No predictions or scores read or generated.']}
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'excluded_questions':summary['excluded_question_counts'],'excluded_groups':summary['excluded_group_counts'],
                      'retained_questions':summary['retained_question_counts'],'confirm_panel':summary['confirm_panel_retained_counts']},ensure_ascii=False))

if __name__=='__main__':main()
