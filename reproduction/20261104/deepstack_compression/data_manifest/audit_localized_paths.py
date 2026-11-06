#!/usr/bin/env python3
"""Verify actual author localized TextVQA path map and panel image fingerprints."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import build

HERE = Path(__file__).resolve().parent
ROOT = build.ROOT
AUTHOR = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval')
csv.field_size_limit(100_000_000)
def sha(b): return hashlib.sha256(b).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def load(n):
    with (HERE/n).open(encoding='utf8') as f: return [json.loads(x) for x in f]

def main():
    source=ROOT/'TextVQA_VAL.tsv'; local=ROOT/'TextVQA_VAL_local.tsv'
    # LOCALIZE maps reference IDs to their source ID filename, not the current qid.
    source_refs={}
    with source.open(encoding='utf8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            qid, value=r['index'],r['image']
            source_refs[qid]=value if value.isdecimal() and len(value)<=64 else qid
    local_rows={}
    with local.open(encoding='utf8',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            assert r['index'] not in local_rows
            local_rows[r['index']]=r['image_path']
    assert set(local_rows)==set(source_refs)
    mismatches=[]
    for qid, source_id in source_refs.items():
        expected=ROOT/'images'/'TextVQA_VAL'/(source_id+'.jpg')
        if Path(local_rows[qid]) != expected: mismatches.append({'qid':qid,'source_id':source_id,'local_path':local_rows[qid],'expected':str(expected)})
    full={(r['task'],r['qid']):r for r in load('full_manifest.jsonl')}
    images=load('images.jsonl')
    previous={(r['task'],r['qid']):r for r in load('panel_cache_audit.jsonl')}
    panel=load('panel_manifest.jsonl')
    result=[]; stats=Counter()
    for n,p in enumerate(panel):
        if p['task']!='TextVQA_VAL':
            old=previous[(p['task'],p['qid'])].copy()
            old['path_provenance']='explicit source image_path'; result.append(old); continue
        ref=images[full[(p['task'],p['qid'])]['image_index']]
        path=Path(local_rows[p['qid']]); key=p['task']+':'+p['split']
        x={**p,'expected_path':str(path),'path_provenance':'TextVQA_VAL_local.tsv',
           'source_image_qid':source_refs[p['qid']],'tsv_byte_sha256':ref['byte_sha256'],
           'tsv_rgb_sha256':ref['rgb_sha256'],'tsv_phash':ref['phash']}
        stats[key+':total']+=1
        try:
            raw=path.read_bytes(); wh,pix,ph=build.describe(raw)
            x.update({'status':'found','cache_byte_sha256':sha(raw),'cache_rgb_sha256':pix,
                      'cache_phash':f'{ph:016x}','cache_width':wh[0],'cache_height':wh[1],
                      'byte_match':sha(raw)==ref['byte_sha256'],'rgb_match':pix==ref['rgb_sha256'],
                      'phash_hamming_to_tsv':(ph^int(ref['phash'],16)).bit_count()})
            stats[key+':found']+=1; stats[key+':byte_equal']+=int(x['byte_match'])
            stats[key+':rgb_equal']+=int(x['rgb_match']); stats[key+':phash_distance_gt4']+=int(x['phash_hamming_to_tsv']>4)
        except FileNotFoundError:
            x['status']='missing_localized_path'; stats[key+':missing']+=1
        except (OSError,ValueError) as e:
            x['status']='unreadable'; x['error']=str(e); stats[key+':unreadable']+=1
        result.append(x)
        if stats[key+':total']%64==0: print(key,stats[key+':total'],flush=True)
    out=HERE/'panel_runtime_paths.jsonl'
    with out.open('w',encoding='utf8') as f:
        for x in result: f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
    metadata={'localized_tsv':str(local),'localized_tsv_sha256':filehash(local),
              'source_tsv_sha256':filehash(source), 'original_path_audit_sha256':filehash(HERE/'panel_cache_audit.jsonl'),
              'author_sources_sha256':{str(AUTHOR/p):filehash(AUTHOR/p) for p in ['dataset/image_base.py','smp/file.py','smp/vlm.py']},
              'script_sha256':filehash(Path(__file__)),'runtime_paths_sha256':filehash(out),
              'source_local_id_count':len(source_refs),'localized_path_mismatches':mismatches,'text_counts':dict(sorted(stats.items())),
              'limitations':['Actual E1 runner must explicitly consume these paths and verify hashes.',
                             'TSV-derived groups remain unchanged; no model inputs or old caches were overwritten.',
                             'Original qid.jpg audit used the non-localized branch and is not the author default on this installation.']}
    (HERE/'panel_runtime_paths_summary.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps({'path_mismatches':len(mismatches),'counts':dict(stats)}),flush=True)
    if mismatches or any(x['status']!='found' for x in result): raise SystemExit('missing or mismatched runtime paths')

if __name__=='__main__':main()
