"""CPU-only fresh DocVQA numeric-evidence panel, independent QA-answer labels.

Provided OCR is used for length filtering and silver localization only. It is
never promoted to human OCR transcription. No model outputs enter selection.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import pyarrow.parquet as pq
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'binding_oracle'))
import data_build as db
import data_train as dt


def literal_numeric(value):
    return re.sub(r'[\s,$£€%]','',str(value)).rstrip('.')


def hashfile(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(db.QWEN/'outputs/local_reward_diagnostic/data'))
    ap.add_argument('--primary',type=int,default=32);ap.add_argument('--reserve',type=int,default=16)
    ap.add_argument('--seed',type=int,default=2026110817);ap.add_argument('--minimum-ocr-words',type=int,default=40);ap.add_argument('--maximum-ocr-words',type=int,default=450)
    args=ap.parse_args();out=Path(args.out)
    for p in [out,out/'images',out/'provided_ocr']:p.mkdir(parents=True,exist_ok=True)
    val_tsv=Path('/media/disk2/YZX/LMUData/DocVQA_VAL.tsv')
    val=pd.read_csv(val_tsv,sep='\t',usecols=['image_path','ucsf_document_id'])
    val_docs=set(val.ucsf_document_id.astype(str))
    cf_path=db.QWEN/'outputs/cf_mining/manifest.json';cf=json.loads(cf_path.read_text());cf_images={r['image_id'] for r in cf['records']}
    cf_docs=set(val[val.image_path.isin(cf_images)].ucsf_document_id.astype(str))
    binding_path=db.QWEN/'outputs/binding_oracle/data/combined_manifest.json';binding=json.loads(binding_path.read_text());binding_docs={r['source_document_id'] for r in binding['records']}
    # Conservative: exclude every parent in the historical EADP_v2 fit roster,
    # even if the audit does not prove that it entered an actual training run.
    audit_rows=dt.TRAIN/'train_audit_001/rows.jsonl';historical_fit=set()
    for line in audit_rows.open():
        row=json.loads(line)
        if row['split']=='fit':historical_fit.add(str(row['document_id']))
    excluded=val_docs|cf_docs|binding_docs|historical_fit
    shards=sorted((dt.TRAIN/'train_fetch_001/shards').glob('*.parquet'))
    frame_rows=[];page_loc={}
    for si,path in enumerate(shards):
        rows=pq.read_table(path,columns=['question_id','question','answers','other_metadata','data_split']).to_pylist()
        for ri,row in enumerate(rows):
            md=row['other_metadata'];assert row['data_split']=='train'
            frame_rows.append(dict(index=row['question_id'],question=row['question'],answer=repr(row['answers']),image_path=md['image'],docId=md['doc_id'],ucsf_document_id=md['ucsf_document_id'],shard_index=si,local_row_index=ri))
            page_loc.setdefault(md['image'],(si,ri))
    frame=pd.DataFrame(frame_rows);eligible=frame[~frame.ucsf_document_id.astype(str).isin(excluded)]
    qas=db.numeric_questions(eligible)
    qas={image:fields for image,fields in qas.items() if len({f['numeric_key'] for f in fields if sum(c.isdigit() for c in f['answer'])>=2})>=2}
    rng=random.Random(args.seed);order=list(range(len(shards)));rng.shuffle(order)
    records=[];rejections=[];used_docs=set();started=time.time();shard_hashes={}
    print(f'annotations={len(frame)} excluded_docs={len(excluded)} numeric_candidate_pages={len(qas)} historical_fit_roster={len(historical_fit)}',flush=True)
    provenance=dict(dataset='pixparse/docvqa-single-page-questions',repository='https://huggingface.co/datasets/pixparse/docvqa-single-page-questions',repository_revision=dt.REVISION,
        original_split='TRAIN; this panel is not official held-out benchmark or proof of unseen layouts',source_root=str(dt.TRAIN),seed=args.seed,
        evidence_ground_truth='Original human DocVQA QA-answer variants; literal coverage in a generated OCR transcript is an evidence proxy, not actual QA accuracy or full-page OCR gold.',
        localization='Unique format-tolerant provided-OCR numeric token per annotated answer: matching removes currency/comma/percent/edge whitespace and terminal periods after both strings pass the numeric regex. OCR polygons/line proposals are silver, not human gold boxes; original QA aliases stay unchanged for independent strict coverage.',
        selection='Annotations/OCR length/geometry and fixed seeded random ordering only; no model prediction or compression-failure selection.',
        compact_numeric_document_scope=dict(min_provided_ocr_words=args.minimum_ocr_words,max_provided_ocr_words=args.maximum_ocr_words,min_distinct_QA_numeric_fields=2,min_answer_digits=2),
        candidate_order='Seeded shuffled parquet-shard order, then seeded shuffled candidate image paths within each shard. One page per source parent.',
        exclusion_policy='All VAL parents, CF parents, previous binding parents, and conservatively all historical fit-roster parents, whether or not actual training exposure is proven.',
        exclusion_sources=dict(VAL=str(val_tsv),CF=str(cf_path),binding=str(binding_path),historical_fit_roster=str(audit_rows)),
        excluded_parent_counts=dict(VAL=len(val_docs),CF=len(cf_docs),binding=len(binding_docs),historical_fit_roster=len(historical_fit),union=len(excluded)),
        excluded_parent_documents=sorted(excluded),primary_target=args.primary,reserve_cap=args.reserve,provided_ocr_role='length filtering and silver localization only; not independent transcript truth')
    def save(complete):
        splitcounts=dict(Counter(r['role'] for r in records));words=[r['provided_ocr_word_count'] for r in records]
        summary=dict(schema_version=1,complete=complete,provenance=provenance,counts=dict(pages=len(records),independent_source_documents=len({r['source_document_id'] for r in records}),roles=splitcounts,
                original_QA_fields=sum(len(r['qa_fields']) for r in records),provided_ocr_word_min=min(words) if words else None,provided_ocr_word_max=max(words) if words else None),
                items=records,records=records,rejections=rejections,scanned=dict(annotations=len(frame),eligible_numeric_pages=len(qas),seconds=time.time()-started),source_parquet_sha256=shard_hashes)
        db.atomic_json(out/'manifest.json',summary)
        inv={k:v for k,v in summary.items() if k not in ['records','items','rejections']};inv['manifest']=str((out/'manifest.json').resolve());inv['selected_page_ids']=[r['page_id'] for r in records]
        db.atomic_json(out/'data_inventory.json',inv)
    save(False)
    for si in order:
        path=shards[si];pages=sorted(p for p in qas if page_loc[p][0]==si);rng.shuffle(pages)
        if not pages:continue
        ocrs=pq.read_table(path,columns=['ocr_results']).to_pylist();images=None
        for image_id in pages:
            document=qas[image_id][0]['source_document_id']
            if document in used_docs:continue
            _,ri=page_loc[image_id];raw=ocrs[ri]['ocr_results'];word_count=sum(len(line['words']) for line in raw['lines'])
            if not args.minimum_ocr_words<=word_count<=args.maximum_ocr_words:
                rejections.append(dict(image_id=image_id,reason='provided_OCR_word_count_outside_fixed_range',word_count=word_count));continue
            orientation=float(raw.get('clockwise_orientation') or 0)%360
            if min(orientation,360-orientation)>5 or raw.get('unit')!='pixel':
                rejections.append(dict(image_id=image_id,reason='nonhorizontal_or_nonpixel_provided_OCR'));continue
            payload=dt.provided_ocr(raw,out/'images'/image_id);matched=[];keys=set()
            for field in qas[image_id]:
                aliases=[a for a in field['answers'] if db.NUMERIC.fullmatch(a) and sum(c.isdigit() for c in a)>=2]
                literal_keys={literal_numeric(a) for a in aliases}
                matches=[w for w in payload['words'] if db.NUMERIC.fullmatch(w['text']) and literal_numeric(w['text']) in literal_keys]
                if len(matches)!=1 or matches[0].get('raw_confidence')=='Low' or min(matches[0]['bbox'][2:])<6:continue
                word=matches[0]
                if field['numeric_key'] in keys:continue
                keys.add(field['numeric_key']);line=payload['lines'][word['line_id']]
                matched.append(dict(field_id=f'f{len(matched)}',question_id=field['question_id'],question=field['question'],answers=field['answers'],answers_original=field['answers'],
                    preferred_answer=field['answer'],annotation_source_split='train',annotation_row_index=field['row_idx'],
                    value_bbox=word['bbox'],value_polygon=word['polygon'],provided_ocr_text=word['text'],provided_ocr_confidence=word['raw_confidence'],
                    context_line_bbox=line['bbox'],context_line_text=line['text'],context_line_polygon=line['polygon'],localization_status='QA-grounded provided-OCR silver region; no human bounding-box annotation'))
            if len(matched)<2:
                rejections.append(dict(image_id=image_id,reason='fewer_than_two_unique_nonLow_QA_grounded_literal_numeric_tokens',eligible_fields=len(matched)));continue
            matched=matched[:4]
            if images is None:images=pq.read_table(path,columns=['image'])['image']
            image=images[ri].as_py();target=out/'images'/image_id;target.write_bytes(image['bytes'])
            with Image.open(target) as im:size=list(im.size)
            if size!=[raw['width'],raw['height']]:
                rejections.append(dict(image_id=image_id,reason='image_OCR_size_mismatch'));target.unlink();continue
            ocrpath=out/'provided_ocr'/(image_id+'.json');db.atomic_json(ocrpath,payload)
            all_qa=[dict(question_id=int(row['index']),question=row.question,answers=db.answers(row.answer),shard=str(shards[int(row.shard_index)]),local_row_index=int(row.local_row_index)) for _,row in frame[frame.image_path==image_id].iterrows()]
            role='primary' if len(records)<args.primary else 'reserve'
            if str(path) not in shard_hashes:shard_hashes[str(path)]=hashfile(path)
            record=dict(page_id='docvqa_train_'+image_id.removesuffix('.png'),record_id='docvqa_train_'+image_id.removesuffix('.png'),role=role,selection_rank=len(records),dataset='DocVQA_TRAIN',source_document_id=document,source_parent=document,
                image_id=image_id,image=str(target.resolve()),image_path=str(target.resolve()),source_image=str(target.resolve()),source_image_size=size,source_sha256=hashlib.sha256(image['bytes']).hexdigest(),
                provided_ocr_word_count=word_count,provided_ocr_lines=len(raw['lines']),provided_ocr_path=str(ocrpath.resolve()),qa_fields=matched,qa=matched,questions=matched,source_qa=all_qa,
                doc_id='docvqa_train_'+image_id.removesuffix('.png'),parent_id=document,
                qa_repository_revision=dt.REVISION,source_parquet=str(path),source_parquet_row=ri,source_parquet_sha256=shard_hashes[str(path)],
                region_coordinate_convention='source image pixel-edge xywh; polygons x1,y1,...; native preprocessing projection must be computed by runner',
                evidence_warning=provenance['evidence_ground_truth'])
            records.append(record);used_docs.add(document);assert document not in excluded
            save(False);print(f'[{len(records)}] {role} {record["page_id"]} fields={len(matched)} words={word_count} image={target.resolve()}',flush=True)
            if len(records)>=args.primary+args.reserve:break
        del ocrs,images
        if len(records)>=args.primary+args.reserve:break
    # Hash only source shards actually used; preserve source bytes untouched.
    assert len({r['source_document_id'] for r in records})==len(records)
    assert len({r['source_sha256'] for r in records})==len(records)
    save(True)
    print('FINAL '+str(out/'manifest.json')+' '+str(Counter(r['role'] for r in records)),flush=True)

if __name__=='__main__':main()
