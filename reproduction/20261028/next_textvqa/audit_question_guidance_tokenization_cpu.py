"""Exact offline native CLIPTokenizerFast audit; no model or scoring calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SNAPSHOT = Path('/home/dell/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1')
SUFFIX = '\nAnswer the question using a single word or phrase.'
SEPARATOR = '\nReference OCR token'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1'
    import numpy as np
    from transformers import CLIPTokenizerFast
    import transformers
    assert transformers.__version__ == '4.37.2'
    tokenizer = CLIPTokenizerFast.from_pretrained(SNAPSHOT, local_files_only=True)
    assert type(tokenizer) is CLIPTokenizerFast
    assert tokenizer.bos_token_id == 49406 and tokenizer.eos_token_id == 49407
    questions_path = HERE / 'author_linked_textvqa_questions.jsonl'
    assert sha(questions_path) == '9f60db6fb89e369263ed676dc26b48d00d4a462e73f0930c09d23394e9dcf3a3'
    questions = [json.loads(s) for s in questions_path.read_text().splitlines()]
    references = [json.loads((HERE / n).read_text())['CLIP_text_rows'] for n in
                  ['preprocess_current_all_images.json', 'preprocess_author_pins_all_images.json']]
    assert len(questions) == len(references[0]) == len(references[1]) == 5000 and references[0] == references[1]
    rows, full_lengths, question_lengths = [], [], []
    for i,(question, reference) in enumerate(zip(questions,references[0])):
        assert reference['position'] == i and str(reference['question_id']) == str(question['question_id'])
        text = question['text']
        assert text.count(SEPARATOR) == 1 and len(text.split('\n')) == 3 and text.endswith(SUFFIX)
        full = text.replace(SUFFIX, '')
        short = text.split(SEPARATOR)[0].replace(SUFFIX, '')
        assert short.strip()
        outputs = {}
        for name, guidance in [('full_guidance', full), ('question_only_guidance', short)]:
            encoded = tokenizer(text=guidance, return_tensors='np')
            ids, mask = encoded.input_ids, encoded.attention_mask
            assert ids.dtype == mask.dtype == np.int64 and ids.ndim == 2 and ids.shape[0] == 1
            assert ids.shape == mask.shape and int(ids[0,0]) == 49406 and int(ids[0,-1]) == 49407
            assert np.all(mask == 1)
            outputs[name] = dict(guidance_sha256=hashlib.sha256(guidance.encode()).hexdigest(),
                ids_length=int(ids.shape[1]),input_ids_sha256=array_sha(ids),attention_mask_sha256=array_sha(mask),
                native_num_segments=(int(ids.shape[1])-1)//77+1)
        full_record = outputs['full_guidance']
        assert all(full_record[k] == reference[k] for k in ('ids_length','input_ids_sha256','attention_mask_sha256'))
        full_lengths.append(full_record['ids_length'])
        question_lengths.append(outputs['question_only_guidance']['ids_length'])
        rows.append(dict(position=i,question_id=question['question_id'],
            unchanged_full_LLM_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
            full_guidance_matches_both_existing_reference_stacks=True,**outputs))
    assert len(rows) == 5000 and sha(questions_path) == '9f60db6fb89e369263ed676dc26b48d00d4a462e73f0930c09d23394e9dcf3a3'
    if 'torch' in sys.modules:
        assert not sys.modules['torch'].cuda.is_initialized()
    mapping = HERE / 'guidance_question_boundary_exact_token_rows.jsonl'
    with mapping.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    files = [questions_path, HERE/'preprocess_current_all_images.json', HERE/'preprocess_author_pins_all_images.json',
             Path(__file__).resolve(), SNAPSHOT/'tokenizer.json', SNAPSHOT/'tokenizer_config.json',
             SNAPSHOT/'special_tokens_map.json', SNAPSHOT/'vocab.json', SNAPSHOT/'merges.txt']
    describe = lambda p:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
    report = dict(success=True,checked_utc=datetime.now(timezone.utc).isoformat(),n5000=5000,
        tokenizer_class=type(tokenizer).__module__+'.'+type(tokenizer).__name__,transformers=transformers.__version__,
        official_revision=SNAPSHOT.name,local_files_only=True,tokenization_device='CPU numpy int64',
        full_guidance_transform='text.replace(exact_answer_suffix,empty)',
        question_only_transform='text.split(newline_Reference_OCR_token)[0].replace(exact_answer_suffix,empty)',
        full_input_ID_and_attention_hash_matches_existing_current_and_author_refs=5000,
        separator_exactly_once=5000,question_after_split_nonempty=5000,
        full_guidance=dict(min_tokens=min(full_lengths),max_tokens=max(full_lengths),longer_than77=sum(x>77 for x in full_lengths),
            length_histogram=dict(sorted(Counter(full_lengths).items()))),
        question_only_guidance=dict(min_tokens=min(question_lengths),max_tokens=max(question_lengths),
            at_most77=sum(x<=77 for x in question_lengths),longer_than77=sum(x>77 for x in question_lengths),
            all_native_single_segment=all((x-1)//77+1==1 for x in question_lengths),
            length_histogram=dict(sorted(Counter(question_lengths).items()))),
        exact_lengths_measured=True,byte_upper_bounds_used_as_measurements=False,
        supersedes_length_bounds_in=str(HERE/'guidance_question_boundary_audit.json'),
        full_LLM_input_IDs_or_images_modified=False,model_loaded=False,CUDA_initialized=False,GT_or_answers_read=False,
        scorer_invoked=False,GPU_generation=False,external_requests=False,
        perquestion_mapping=describe(mapping),sources=[describe(p) for p in files])
    output = HERE / 'guidance_question_boundary_exact_tokenization.json'
    with output.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(success=True,n=5000,full_hash_matches=5000,full_max=max(full_lengths),full_long=sum(x>77 for x in full_lengths),
        question_min=min(question_lengths),question_max=max(question_lengths),question_long=sum(x>77 for x in question_lengths),
        report_sha256=sha(output),mapping_sha256=sha(mapping))))


if __name__ == '__main__':
    main()
