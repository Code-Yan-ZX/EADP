"""Formal no-GT acceptance: full denominator, not an equality success gate."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import admission
import baseline
import compare
import finish_guard

D=Path(__file__).resolve().parent;B=D.parents[1];O=D/'run_run001/observations'
MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')

def snapshot_check(plan):
    baseline.supervised(D/'monitor_snapshot001',D/'snapshot.py')
    manifest=baseline.read(D/'snapshot001/manifest.json')
    if manifest.get('status')!='passed' or manifest.get('plan_sha256')!=baseline.sha(D/'plan002.json') or set(manifest['files'])!=set(plan['source_sha256']):
        raise ValueError('snapshot manifest mismatch')
    for source,record in manifest['files'].items():
        if record['sha256']!=plan['source_sha256'][source] or baseline.sha(record['path'])!=record['sha256']:
            raise ValueError('source snapshot mismatch')
    return baseline.sha(D/'snapshot001/manifest.json')

def validate_events(events,anchors,rows):
    expected=[(a['task'],str(a['qid'])) for a in anchors]
    start=[(e['task'],str(e['qid'])) for e in events if e.get('event')=='arm_started']
    end=[(e['task'],str(e['qid'])) for e in events if e.get('event')=='arm_recorded']
    if start!=expected or end!=expected or any(e.get('status')!='completed' for e in events if e.get('event')=='arm_recorded'):
        raise ValueError('journal incomplete, failed, or reordered')
    for a,r in zip(anchors,rows):
        steps=[e['index'] for e in events if e.get('event')=='forward_complete' and e.get('task')==a['task'] and str(e.get('qid'))==str(a['qid'])]
        if steps!=list(range(r['generated_count'])):raise ValueError('forward count')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    plan=admission.check_plan();admission.check_review('preflight');admission.check_review('final')
    run=finish_guard.validate();snapshot_sha=snapshot_check(plan)
    oldplan=baseline.read(baseline.ORIGINAL/'plan001.json');anchors=oldplan['anchors']
    complete=baseline.read(O/'worker_complete.json')
    if (complete.get('status')!='diagnostic_completed' or complete.get('wavefront_selector_restored') is not True or complete.get('n_expected')!=384 or complete.get('n_recorded')!=384
        or complete.get('GT_opened') is not False or complete.get('scoring_performed') is not False):
        raise ValueError('worker did not complete full panel')
    oldverifier=B/'e6/pixelprune_t020_accept/verify.py'
    spec=importlib.util.spec_from_file_location('_pixel_original_validate',oldverifier);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    comparisons=[];records=[];files={}
    for anchor in anchors:
        key=anchor['task']+'_'+str(anchor['qid']);old=baseline.ORIGINAL/'run_smoke001/observations'
        op=old/(key+'.json');og=old/(key+'_generated.json');np=O/(key+'.json');ng=O/(key+'_generated.json')
        row=baseline.read(np);raw=baseline.read(ng);previous=baseline.read(op);previous_raw=baseline.read(og)
        original.validate_item(row,raw,anchor,tokenizer)
        comparisons.append(compare.compare_one(anchor,previous,previous_raw,row,raw));records.append(row)
        for path in (np,ng):files[str(path)]=baseline.sha(path)
    events=[json.loads(line) for line in (O/'progress.jsonl').read_text().splitlines()]
    validate_events(events,anchors,records);summary=compare.summarize(anchors,comparisons)
    for path in (O/'progress.jsonl',O/'worker_complete.json',D/'run_run001/launch.json',D/'run_run001/exit.json'):
        files[str(path)]=baseline.sha(path)
    out={'status':'passed','questions':384,'physical_generations':384,'GT_opened':False,'predictions_opened':True,'scoring_performed':False,
         'comparisons':comparisons,'summary':summary,'files_sha256':files,'source_sha256':plan['source_sha256'],
         'plan_sha256':baseline.sha(D/'plan002.json'),'snapshot_manifest_sha256':snapshot_sha,'run':run,'verifier_sha256':baseline.sha(__file__),
         'limitations':['post-DEV exploratory comparison','full historical visual/DS/KV tensors and first logits not archived',
                        'existing DEV predictions parsed for comparison; no GT opened','not a timing comparison'], 'errors':[]}
    with args.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print('full no-GT verification complete')

if __name__=='__main__':main()
