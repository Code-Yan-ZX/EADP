"""Wait for both complete real CPU processor passes; compare without a model."""
import argparse
from array import array
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=HERE/'processor_input_identity'
DIRECTORIES=[BASE/'current_qwen3vl_clean_fullDoc',BASE/'author_qwen_fullDoc']


class Pending(Exception):pass


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def current_ticks(pid):
    try:return int(Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return None


def gates():
    records=[]
    for directory in DIRECTORIES:
        assert not (directory/'failure.json').exists(), 'Preserved CPU worker failure: '+str(directory/'failure.json')
        try:state=read(directory/'state.json')
        except (FileNotFoundError,json.JSONDecodeError) as error:raise Pending('CPU worker control record missing/being written') from error
        identity=state['identity'];ticks=current_ticks(identity['pid'])
        if state['status']!='complete':
            assert ticks==identity['start_ticks'], 'CPU worker disappeared before complete proof: '+str(directory)
            raise Pending(str(directory.name)+': '+str(state['rows'])+'/5349')
        if ticks==identity['start_ticks']:raise Pending('Completed CPU worker is finishing its last print/exit')
        report=read(directory/'report.json');preparation=read(directory/'preparation.json')
        assert sha(directory/'report.json')==state['report_sha256']
        assert report['status']=='complete' and report['rows']==5349 and report['unique_images']==1286
        assert report['real_full_processor_calls']==5349 and not report['image_output_cache_reused']
        assert report['identity']==identity and report['preparation_sha256']==sha(directory/'preparation.json')
        assert report['source_input_processor_hashes_unchanged'] and sha(HERE/'processor_doc_full.py')==report['script_sha256']
        assert not report['model_loaded'] and not report['CUDA_initialized'] and not report['FA2_modules_imported']
        assert report['current_dependency_stack_only'] and not report['historical_environment_reconstruction']
        for path,expected in preparation['frozen_small_files'].items():assert sha(path)==expected,path
        for item in preparation['processor_source'].values():assert sha(item['path'])==item['sha256']
        for name in ('per_question','unique_image_fingerprints'):
            assert Path(report[name]['path']).resolve()==(directory/('per_question.jsonl' if name=='per_question' else 'unique_images.json')).resolve()
            assert sha(report[name]['path'])==report[name]['sha256']
        records.append((state,report,preparation))
    return records


def validate_integer_tensor(values,record):
    assert sys.byteorder=='little' and record['dtype']=='torch.int64'
    assert isinstance(values,list) and len(values)==1
    assert record['shape']==[1,len(values[0])]
    assert hashlib.sha256(array('q',values[0]).tobytes()).hexdigest()==record['sha256']


def compare(records):
    canonical_path=HERE/'independent_audit/DocVQA_VAL.canonical.jsonl'
    canonical=[json.loads(line)['index'] for line in canonical_path.read_text().splitlines()]
    assert len(canonical)==len(set(canonical))==5349
    images=[read(folder/'unique_images.json') for folder in DIRECTORIES]
    assert set(images[0])==set(images[1]) and len(images[0])==1286
    image_differences=[]
    for path in sorted(images[0]):
        left,right=[m[path] for m in images]
        assert left['question_count']==right['question_count'] and left['first_position']==right['first_position']
        if left['fingerprint']!=right['fingerprint']:
            image_differences.append(dict(image_path=path,current_stack=left,author_stack=right))
    counts=Counter();differences=[];seen=set()
    with (DIRECTORIES[0]/'per_question.jsonl').open() as a,(DIRECTORIES[1]/'per_question.jsonl').open() as b:
        for position,pair in enumerate(zip_longest(a,b)):
            assert None not in pair and position<5349
            rows=[json.loads(line) for line in pair]
            assert all(row['position']==position and row['index']==canonical[position] for row in rows)
            assert rows[0]['image_path']==rows[1]['image_path'];seen.add(rows[0]['index'])
            for row,image_map in zip(rows,images):
                assert row['image_grid_thw']==[[1,64,64]]
                for name in ('input_ids','attention_mask','image_grid_thw'):
                    validate_integer_tensor(row[name],row['raw_processor_tensors'][name])
                    assert row['raw_processor_tensors'][name]==row['native_CPU_BF16_tensors'][name]
                fingerprint=image_map[row['image_path']]['fingerprint']
                assert row['raw_processor_tensors']['pixel_values']==fingerprint['raw_pixel_values']
                assert row['native_CPU_BF16_tensors']['pixel_values']==fingerprint['native_BF16_pixel_values']
            changed=[key for key in rows[0] if rows[0][key]!=rows[1][key]]
            if changed:
                counts.update(changed);differences.append(dict(position=position,index=rows[0]['index'],changed_fields=changed))
    assert position+1==len(seen)==5349
    assert sum(item['question_count'] for item in images[0].values())==5349
    output=dict(status='complete',CPU_only=True,model_loaded=False,CUDA_initialized=False,FA2_imported=False,
                completed_utc=datetime.now(timezone.utc).isoformat(),n_prompts=5349,n_unique_images=1286,
                real_full_processor_calls_per_environment=5349,image_output_cache_reused=False,
                current_dependency_stack_only=True,historical_environment_reconstructed=False,
                all_source_processor_ASTs_equal=records[0][2]['source_AST_proof']['original_processor_AST_sha256']==records[1][2]['source_AST_proof']['original_processor_AST_sha256'],
                environment_versions=[r[2]['versions'] for r in records],
                all_image_fingerprints_equal=not image_differences,all_prompt_records_equal=not differences,
                n_different_unique_images=len(image_differences),n_different_prompt_records=len(differences),
                changed_prompt_field_counts=dict(counts),image_differences=image_differences,prompt_differences=differences,
                actual_CPU_workers_exited=[dict(identity=r[0]['identity'],original_exited=current_ticks(r[0]['identity']['pid'])!=r[0]['identity']['start_ticks']) for r in records],
                sources=[dict(path=str(folder/name),sha256=sha(folder/name)) for folder in DIRECTORIES for name in ('state.json','process_identity.json','preparation.json','report.json','per_question.jsonl','unique_images.json')],
                comparer=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
                limits=['This compares current qwen3vl_clean and current author CPU processor stacks, not the unrecorded effective September22 package/backend environment.',
                        'Both execute all5349 complete real source processor paths;1286 unique images are deduplicated only in saved evidence, never supplied from a fake/cache processor.',
                        'Input identity does not establish historical model-input bytes, paper argv, attention-backend equivalence, or a causal explanation for answer/ANLS changes.'])
    save(BASE/'fullDoc_comparison.json',output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    claim=BASE/'fullDoc_comparer_identity.json'
    if args.wait:
        with claim.open('x') as stream:json.dump(dict(pid=os.getpid(),start_ticks=current_ticks(os.getpid()),CPU_only=True,model_loaded=False,automatic_retry=False),stream,indent=2)
    while True:
        try:
            complete=gates()
            if args.check:print(json.dumps(dict(ready=True,per_question_read=False)));return
            result=compare(complete)
            print(json.dumps({k:result[k] for k in ('status','n_prompts','n_unique_images','all_image_fingerprints_equal','all_prompt_records_equal','n_different_unique_images','n_different_prompt_records')}));return
        except Pending as error:
            if not args.wait:print(json.dumps(dict(status='pending',reason=str(error),per_question_read=False)));return
            time.sleep(10)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(BASE/'fullDoc_comparison_failure.json',dict(status='terminal_failure',error=repr(error),automatic_retry=False,model_loaded=False))
        raise
