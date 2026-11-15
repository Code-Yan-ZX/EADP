"""Expand binding pairs from already-local DocVQA TRAIN parquet + provided OCR.

No model imports or predictions. No downloads. VAL parents (including prior
cf-mining parents) are excluded. The original v2 records are copied unchanged
as the prefix of combined_manifest.json. Null provided-OCR confidence remains
null; the provider's explicit Low flag is rejected.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
import time

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import data_build as db

TRAIN = Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/m3_train_data')
REVISION = '33136ef456fa5a3fe68568d6e31dda4eeff95b9b'


def polygon_bbox(polygon):
    xs, ys = polygon[::2], polygon[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def angle(polygon):
    value = abs(math.degrees(math.atan2(polygon[3] - polygon[1], polygon[2] - polygon[0])))
    return min(value, abs(180 - value))


def provided_ocr(raw, source):
    lines, words = {}, []
    for index, item in enumerate(raw['lines']):
        lid = str(index)
        lines[lid] = dict(bbox=polygon_bbox(item['bounding_box']), words=[w['text'] for w in item['words']],
                          polygon=item['bounding_box'], text=item['text'])
        for word in item['words']:
            # An axis-aligned patch must not erase substantially tilted text.
            if angle(word['bounding_box']) > 5:
                continue
            words.append(dict(text=word['text'].strip(), bbox=polygon_bbox(word['bounding_box']),
                              polygon=word['bounding_box'], line_id=lid, confidence=None,
                              raw_confidence=word['confidence'], source_engine='provided DocVQA OCR'))
    return dict(source=str(source), words=words, lines=lines, engine='provided DocVQA OCR',
                metadata={k:v for k,v in raw.items() if k!='lines'})


def summarize(records):
    return dict(pairs=len(records), unique_pages=len({r['image_id'] for r in records}),
                unique_documents=len({r['source_document_id'] for r in records}),
                splits=dict(Counter(r['split'] for r in records)),
                line_disjoint_pairs=sum(r['eligible_for_line_oracle'] for r in records),
                value_token_masks_disjoint=sum(r['value_overlap_masks_disjoint'] for r in records))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-root', default=str(TRAIN))
    ap.add_argument('--val-tsv', default='/media/disk2/YZX/LMUData/DocVQA_VAL.tsv')
    ap.add_argument('--base-manifest', default=str(db.OUT/'v2/manifest.json'))
    ap.add_argument('--out', default=str(db.OUT/'extra'))
    ap.add_argument('--target', type=int, default=40)
    ap.add_argument('--preview', type=int, default=12)
    ap.add_argument('--seed', type=int, default=20261109)
    ap.add_argument('--font', default=db.FONT)
    args=ap.parse_args()
    out=Path(args.out);root=Path(args.train_root)
    for directory in [out,out/'images',out/'source_images',out/'ocr']:
        directory.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path(args.base_manifest).read_text());base_records=base['records']
    val=pd.read_csv(args.val_tsv,sep='\t',usecols=['ucsf_document_id'])
    excluded=set(val.ucsf_document_id.astype(str))|{r['source_document_id'] for r in base_records}
    shards=sorted((root/'train_fetch_001/shards').glob('*.parquet'))
    rows=[]
    for si,path in enumerate(shards):
        raw=pq.read_table(path,columns=['question_id','question','answers','data_split','other_metadata']).to_pylist()
        for ri,item in enumerate(raw):
            m=item['other_metadata']
            assert item['data_split']=='train'
            rows.append(dict(index=item['question_id'],question=item['question'],answer=repr(item['answers']),
                             docId=m['doc_id'],ucsf_document_id=m['ucsf_document_id'],image_path=m['image'],
                             shard=si,local_row=ri,data_split=item['data_split']))
    frame=pd.DataFrame(rows)
    eligible=frame[~frame.ucsf_document_id.astype(str).isin(excluded)]
    questions=db.numeric_questions(eligible)
    questions={p:v for p,v in questions.items() if len({x['numeric_key'] for x in v})>=2}
    first_rows={image:group.iloc[0] for image,group in frame[frame.image_path.isin(questions)].groupby('image_path',sort=False)}
    print(f'TRAIN annotations={len(frame)} excluded VAL parents={len(excluded)} numeric candidate pages={len(questions)}',flush=True)
    records=[];failures=[];quality_rejections=[];used_documents=set();attempted=0;started=time.time()
    source_inventory=root/'source_inventory_001.json'
    provenance=dict(source_split='DocVQA TRAIN; internal source-document dev/confirm partitions, not official evaluation',
                    local_source=str(root),repository='https://huggingface.co/datasets/pixparse/docvqa-single-page-questions',
                    repository_revision=REVISION,source_inventory=str(source_inventory),
                    source_inventory_sha256=hashlib.sha256(source_inventory.read_bytes()).hexdigest(),
                    excluded_all_VAL_source_documents=sorted(excluded),excluded_VAL_parent_count=len(excluded),
                    selection_uses_model_outputs=False,extra_target_fixed_before_compressed_predictions=args.target,
                    confidence_policy='Provided OCR has null or Low flags: reject Low, preserve null without numeric imputation. Tesseract confidence thresholds do not apply to this distinct source.',
                    unchanged_rules='digit-count ratio <=2; shared font height retention >=0.6; no narrow vertical short values; matching unit categories; two distinct unique annotated numerical answers; word angle <=5 degrees',
                    region_warning='QA-grounded provided-OCR value boxes / provided-OCR line proposals; never gold semantic field boundaries',config=vars(args))
    provenance['geometry_warning'] = ('source_to_letterbox1024/value_region_1024/context_line_region_1024 are ideal direct square-pad mappings only. '
        'Native wrapper first fetch_image smart_resize(factor=32,min=max=1024**2), then square-pad and resize1024. '
        'Runner must project raw source pixel boxes through that actual pipeline; do not use ideal token masks for native oracles.')
    def save(complete=False):
        for r in records:
            assert r['source_document_id'] not in excluded
            assert r['original_numeric_multiset']==r['swapped_numeric_multiset']
        payload=dict(schema_version=2,complete=complete,provenance=provenance,counts=summarize(records),records=records,
                     exclusions=failures,candidate_quality_rejections=quality_rejections,
                     scan_counts=dict(annotation_rows=len(frame),numeric_candidate_pages=len(questions),pages_attempted=attempted,wall_seconds=time.time()-started))
        db.atomic_json(out/'manifest.json',payload)
        combined=base_records+records
        cp=dict(schema_version=2,complete=complete,provenance=dict(base_manifest=str(Path(args.base_manifest).resolve()),
                extra_manifest=str((out/'manifest.json').resolve()),base_prefix_unchanged=True,
                selection_uses_model_outputs=False,geometry_warning=provenance['geometry_warning'],
                source_split='DocVQA VAL base + TRAIN extra; internal source-document dev/confirm partitions'),
                counts=summarize(combined),records=combined)
        assert cp['records'][:len(base_records)]==base_records
        assert len({r['source_document_id'] for r in combined})==len(combined)
        db.atomic_json(out.parent/'combined_manifest.json',cp)
    save(False)
    rng=random.Random(args.seed)
    for si,path in enumerate(shards):
        candidates=sorted(p for p,r in first_rows.items() if int(r.shard)==si)
        rng.shuffle(candidates)
        if not candidates:
            continue
        ocr_rows=pq.read_table(path,columns=['ocr_results']).to_pylist()
        image_column=None
        for image_id in candidates:
            if len(records)>=args.target:
                break
            row=first_rows[image_id];document=str(row.ucsf_document_id)
            if document in used_documents:
                continue
            attempted+=1
            raw=ocr_rows[int(row.local_row)]['ocr_results']
            orientation=float(raw.get('clockwise_orientation') or 0)%360
            if min(orientation,360-orientation)>5 or raw['unit']!='pixel':
                failures.append(dict(image_id=image_id,reason='nonhorizontal page or nonpixel OCR coordinates'))
                continue
            source=out/'source_images'/image_id
            payload=provided_ocr(raw,source)
            size=[int(raw['width']),int(raw['height'])]
            if not db.choose_pair(questions[image_id],payload,size,quality_rejections):
                failures.append(dict(image_id=image_id,reason='no eligible two-value swap under deterministic OCR/render/unit rules'))
                continue
            if image_column is None:
                image_column=pq.read_table(path,columns=['image'])['image']
            image=image_column[int(row.local_row)].as_py()
            source.write_bytes(image['bytes'])
            with Image.open(source) as im:
                actual=list(im.size)
            if actual!=size:
                failures.append(dict(image_id=image_id,reason='source image and provided OCR size differ',image_size=actual,ocr_size=size))
                continue
            db.atomic_json(out/'ocr'/(image_id+'.json'),payload)
            ra=SimpleNamespace(image_dir=str(out/'source_images'),out=str(out),font=args.font,seed=args.seed,tsv=str(path))
            record=db.record_for(image_id,questions[image_id],payload,frame,ra)
            assert record is not None
            record['dataset']='DocVQA_TRAIN'
            record['provenance'].update(localization='unique provided-OCR numerical token matching existing TRAIN QA answers',
                context_region_status='provided-OCR line proposal; not a gold semantic field boundary',
                qa_source=str(path),qa_repository_revision=REVISION,
                qa_annotation_row_indices=[int(frame.loc[f['row_idx'],'local_row']) for f in record['fields']],
                qa_annotation_shards=[str(shards[int(frame.loc[f['row_idx'],'shard'])]) for f in record['fields']],
                provided_ocr_orientation=orientation,provided_ocr_image_size=size,
                confidence_policy=provenance['confidence_policy'])
            bybox={tuple(w['bbox']):w for w in payload['words']}
            for f in record['fields']:
                w=bybox[tuple(f['value_bbox'])]
                f['provided_ocr_value_polygon']=w['polygon'];f['provided_ocr_raw_confidence']=w['raw_confidence']
                f['provided_ocr_line_polygon']=payload['lines'][w['line_id']]['polygon']
                assert f['provided_ocr_raw_confidence']!='Low'
            records.append(record);used_documents.add(document)
            save(False)
            print(f'[extra {len(records)}/{args.target}] {record["pair_id"]} {image_id} doc={document}',flush=True)
            if len(records)==args.preview:
                db.atomic_json(out/'manifest_preview.json',json.loads((out/'manifest.json').read_text()))
                db.montage(records,out/'montage_preview.png')
                print('PREVIEW READY '+str(out/'manifest_preview.json'),flush=True)
        del ocr_rows,image_column
        if len(records)>=args.target:
            break
    save(True)
    db.montage(records,out/'montage_preview.png')
    print('FINAL '+str(out/'manifest.json')+' '+str(summarize(records)),flush=True)

if __name__=='__main__':
    main()
