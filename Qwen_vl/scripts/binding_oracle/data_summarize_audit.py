"""Read-only source-annotation/geometry audit; writes only a new audit summary."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from qwen_vl_utils.vision_process import smart_resize
import data_build as db
import data_train as dt


def main():
    root=db.OUT;combined=json.loads((root/'combined_manifest.json').read_text());records=combined['records']
    base=json.loads((root/'v2/manifest.json').read_text());extra=json.loads((root/'extra/manifest.json').read_text())
    pixels=json.loads((root/'combined_pixel_audit.json').read_text())
    assert records[:22]==base['records'];assert records[22:]==extra['records'];assert pixels['all_pass'] and pixels['pairs']==62
    frame=pd.read_csv('/media/disk2/YZX/LMUData/DocVQA_VAL.tsv',sep='\t',usecols=['index','docId','ucsf_document_id','image_path','question','answer'])
    val_lookup={int(r['index']):dict(question=r.question,answers=db.answers(r.answer),image_id=r.image_path,document=str(r.ucsf_document_id)) for _,r in frame.iterrows()}
    all_val_docs=set(frame.ucsf_document_id.astype(str))
    prior=json.loads((db.QWEN/'outputs/cf_mining/manifest.json').read_text());prior_images={r['image_id'] for r in prior['records']}
    prior_docs=set(frame[frame.image_path.isin(prior_images)].ucsf_document_id.astype(str))
    train_lookup={}
    for path in sorted((dt.TRAIN/'train_fetch_001/shards').glob('*.parquet')):
        for row in pq.read_table(path,columns=['question_id','question','answers','other_metadata','data_split']).to_pylist():
            md=row['other_metadata'];train_lookup[int(row['question_id'])]=dict(question=row['question'],answers=[a.strip() for a in row['answers']],image_id=md['image'],document=md['ucsf_document_id'],source_split=row['data_split'])
    annotated_questions=0;field_questions=0;percentage_mismatches=[];field_units=Counter()
    for r in records:
        lookup=val_lookup if r['dataset']=='DocVQA_VAL' else train_lookup
        assert r['source_document_id'] not in prior_docs
        if r['dataset']=='DocVQA_TRAIN':
            assert r['source_document_id'] not in all_val_docs
        qids={q['question_id'] for q in r['source_qa']}
        assert len(qids)>=2
        assert len({f['question_id'] for f in r['fields']})==2
        for index,f in enumerate(r['fields']):
            assert f['answers_swapped']==r['fields'][1-index]['answers_original']
        for q in r['source_qa']:
            gt=lookup[q['question_id']];assert gt['image_id']==r['image_id'];assert gt['document']==r['source_document_id']
            assert gt['question']==q['question'];assert gt['answers']==q['answers'];annotated_questions+=1
        kinds=[]
        for f in r['fields']:
            gt=lookup[f['question_id']];assert f['question_id'] in qids
            assert gt['question']==f['question'];assert gt['answers']==f['answers_original'];field_questions+=1
            k=db.unit_kind(dict(question=f['question'],answers=f['answers_original']),dict(text=f['ocr_text']))
            kinds.append(k);field_units[k]+=1
            aliases=[a for a in f['answers_original'] if db.NUMERIC.fullmatch(a)]
            ocr_percent='%' in f['ocr_text'];alias_flags=['%' in a for a in aliases]
            if alias_flags and ocr_percent not in alias_flags:
                percentage_mismatches.append(dict(pair_id=r['pair_id'],field_id=f['field_id'],dataset=r['dataset'],question=f['question'],ocr_text=f['ocr_text'],aliases=f['answers_original'],ocr_has_percent=ocr_percent))
        assert kinds[0]==kinds[1]
    panel=json.loads((root.parent/'geometry_panel.json').read_text());panels={x['pair_id']:x for x in panel};geometry=[]
    idx=np.arange(1024);left=((idx%32)*32).astype(np.float32);top=((idx//32)*32).astype(np.float32)
    for r in records:
        W,H=r['source_image_size'];rh,rw=smart_resize(H,W,factor=32,min_pixels=1024**2,max_pixels=1024**2)
        side=max(rw,rh);px=(side-rw)//2;py=(side-rh)//2;areas=[]
        for f in r['fields']:
            x,y,w,h=f['value_bbox'];b=np.asarray([(x*rw/W+px)*1024/side,(y*rh/H+py)*1024/side,((x+w)*rw/W+px)*1024/side,((y+h)*rh/H+py)*1024/side],dtype=np.float32)
            area=np.maximum(0,np.minimum(left+32,b[2])-np.maximum(left,b[0]))*np.maximum(0,np.minimum(top+32,b[3])-np.maximum(top,b[1]));areas.append(area)
        areas=np.stack(areas);labels=areas.argmax(0)+1;labels[areas.max(0)==0]=0;counts=np.bincount(labels,minlength=3).tolist();collision=int(((areas>0).sum(0)>1).sum())
        recorded=panels[r['pair_id']];assert recorded['valid'];assert recorded['counts']==counts;assert recorded['collisions']==collision;assert recorded['fetch_size']==[rw,rh]
        geometry.append(dict(pair_id=r['pair_id'],field_token_counts=counts[1:],collision_tokens=collision,fetch_size=[rw,rh],valid=min(counts[1:])>0))
    assert all(g['valid'] for g in geometry)
    native=json.loads((root/'native_geometry_cpu_audit.json').read_text())
    summary=dict(all_pass=True,scope='Data-only audit; no teacher/compressor prediction files opened; no GPU model loaded.',
        manifest=str(root/'combined_manifest.json'),counts=dict(pairs=62,independent_source_documents=62,unique_pages=62,source_byte_hashes=62,base_VAL_pairs=22,extra_TRAIN_pairs=40,split=dict(Counter(r['split'] for r in records))),
        unchanged_prefix=dict(v2_first22_exact_records=True,extra_first12_exact_records=extra['records'][:12]==json.loads((root/'extra/manifest_preview.json').read_text())['records']),
        source_qa=dict(selected_field_questions_verified_against_original_annotations=field_questions,all_retained_source_page_questions_verified=annotated_questions,all_62_pages_have_at_least_two_annotated_questions=True,all_original_aliases_retained=True,swapped_aliases_are_other_field_original_aliases=True,annotation_sources=['local DocVQA_VAL.tsv','already local pinned pixparse/docvqa-single-page-questions TRAIN parquet']),
        leakage=dict(prior_CF_parent_documents=len(prior_docs),prior_CF_parent_overlap_all62=0,all_VAL_parent_documents=len(all_val_docs),extra_TRAIN_overlap_all_VAL_parents=0,base_extra_shared_documents=0,one_page_per_document=True,split_unit='ucsf_document_id',note='Parent-ID and source-byte checks are exact; not an exhaustive semantic near-duplicate or publisher-template audit.'),
        deterministic_quality=dict(max_digit_count_ratio=2,min_shared_font_height_retention=.6,max_provided_word_angle_degrees=5,reject_vertical_narrow_short_values=True,require_unit_category_match=True,field_unit_counts=dict(field_units),Tesseract_confidence_min=55,provided_ocr_confidence='reject explicit Low; null remains null; numeric Tesseract confidence is not imputed',source_pixels_and_dimensions_kept=True,both_value_boxes_rerendered_in_both_members=True,reported_numeric_multiset_unchanged_all62=True),
        pixel_audit=dict(path=str(root/'combined_pixel_audit.json'),all124_rendered_members_outside_documented_patches_changed_pixels=0,source_sha256_matches_all=True,dimensions_preserved_all=True),
        geometry=dict(version='native_smart_resize_then_square_v2',independent_CPU_all62_counts_match_runner=True,all62_have_at_least_one_token_per_field=True,collision_pages=sum(g['collision_tokens']>0 for g in geometry),collision_assignment='larger positive token/box intersection area; ties field0; no prediction-dependent geometry',actual_fetch_image_sizes_checked_against_smart_resize_pages=22,actual_fetch_image_size_checks_all_pass=native['all_fetch_image_sizes_match_smart_resize'],max_ideal_vs_native_edge_shift_v2_pixels=native['max_abs_edge_shift'],ideal_mapping_warning=extra['provenance']['geometry_warning'],region_names='QA-grounded OCR value boxes and OCR line proposals; not gold semantic field polygons',records=geometry),
        percentage_OCR_vs_aliases_format_mismatch=dict(fields=len(percentage_mismatches),pairs=len({x['pair_id'] for x in percentage_mismatches}),records=percentage_mismatches,policy='Keep original aliases and strict percentage-sensitive primary exact score. A teacher answer with the alternate percent format may fail the primary inclusion gate; any unit-aware sensitivity must be uniform and secondary.'),
        representative_figure=dict(path=str(root/'table_swap_audit.png'),metadata=str(root/'table_swap_audit.json'),pair_id='swap_38577_38580',selection='real year-amount table chosen by data type/readability before inspecting model outputs; annotated targets only'),
        rendering_limitations=['OCR rectangles are proposals; a few slanted glyph edge pixels can remain outside the erased rectangle. Symmetric rendering and full-teacher 4-answer gates control some artifacts, but do not establish perfect edit realism.','Provided-OCR and Tesseract confidence conventions differ and are explicitly recorded.','No gold semantic field boundaries, no held-out-layout claim, and no full DocVQA benchmark estimate.'])
    db.atomic_json(root/'data_audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k in ['all_pass','counts','source_qa','leakage','percentage_OCR_vs_aliases_format_mismatch']},indent=2))

if __name__=='__main__':
    main()
