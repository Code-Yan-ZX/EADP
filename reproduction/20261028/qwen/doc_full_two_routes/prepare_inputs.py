"""Create fixed, ground-truth-free full DocVQA input and processor manifest."""
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
Q=HERE.parent
CANON=Q/'independent_audit/DocVQA_VAL.canonical.jsonl'
PROCESSOR=Q/'processor_input_identity/author_qwen_fullDoc/per_question.jsonl'
IMAGES=Q/'independent_audit/current_cached_images.json'
EXPECTED_CANON='79091c0178d228237519c87a50f74f766ca8b2ea6c593d99790d3d0034a45ebc'
EXPECTED_PROCESSOR='51c52ca77f8869623db64515795dd7ce1b22133c730d206f86a513d5b41207bd'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def lines(path):
    with Path(path).open() as f:return [json.loads(s) for s in f]
def main():
    target=HERE/'input_manifest.jsonl';assert not target.exists()
    assert sha(CANON)==EXPECTED_CANON and sha(PROCESSOR)==EXPECTED_PROCESSOR
    canonical=lines(CANON);processor=lines(PROCESSOR)
    images={r['path']:r for r in lines_as_json(IMAGES) if r['dataset']=='DocVQA_VAL'}
    assert len(canonical)==len(processor)==5349 and len(images)==1286
    assert len({str(r['index']) for r in canonical})==5349
    rows=[]
    for pos,(a,b) in enumerate(zip(canonical,processor)):
        assert str(a['index'])==str(b['index']) and b['position']==pos
        image=str(Path('/media/disk2/YZX/LMUData/images/DocVQA_VAL')/a['image_path'])
        assert b['image_path']==image and image in images
        rows.append(dict(position=pos,index=str(a['index']),question=a['question'],image_path=image,
            image_sha256=images[image]['sha256'],message_sha256=b['message_sha256'],
            processor_expected=b['native_CPU_BF16_tensors']))
    with target.open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    proof=dict(rows=5349,images=1286,canonical_sha256=sha(CANON),processor_sha256=sha(PROCESSOR),
        image_metadata_sha256=sha(IMAGES),input_manifest_sha256=sha(target),
        fields=['position','index','question','image_path','image_sha256','message_sha256','processor_expected'],
        selection='all 5349 canonical positions in original order',answers_copied=False,prediction_or_score_used=False,
        CPU_only=True,model_loaded=False,GPU_started=False)
    (HERE/'input_preparation.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))
def lines_as_json(path):return json.loads(Path(path).read_text())
if __name__=='__main__':main()
