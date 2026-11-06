"""Select a unique accepted E1 record from supplied bytes; opens no files.

The caller must first independently validate the formal E1 verifier and
reviewer supervisor/exit chain, bind the accepted model/source recipe and
obtain the explicit future oracle execution admission. This function hashes
both complete byte streams before parsing any record.
"""
import hashlib,json
import e1_admission

def select(proof_bytes,journal_bytes,inputs_bytes,sample,receipt):
    if not isinstance(receipt,e1_admission.Receipt):raise ValueError('verified E1 upstream receipt required')
    proof=json.loads(proof_bytes)
    if hashlib.sha256(proof_bytes).hexdigest()!=receipt.proof_sha:raise ValueError('E1 acceptance changed')
    if (proof.get('status')!='passed' or proof.get('stage')!='full' or proof.get('natural_questions')!=384 or
        proof.get('natural_records')!=1536 or proof.get('tokenizer_decode_checked')!=1536 or
        proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False):
        raise ValueError('E1 completion proof incomplete')
    files=proof['files'];jhash=hashlib.sha256(journal_bytes).hexdigest();ihash=hashlib.sha256(inputs_bytes).hexdigest()
    if files['journal']['sha256']!=jhash or files['inputs']['sha256']!=ihash or receipt.journal_sha!=jhash or receipt.inputs_sha!=ihash:raise ValueError('accepted E1 source hash mismatch')
    key=(sample['task'],str(sample['qid']),sample['group_id']);lines=journal_bytes.splitlines()
    if len(lines)!=1536:raise ValueError('E1 natural inventory incomplete')
    found=[]
    for raw in lines:
        r=json.loads(raw)
        if (r.get('task'),str(r.get('qid')),r.get('group_id'))==key and r.get('arm')=='P3-D1' and r.get('forced') is False:
            found.append(raw)
    if len(found)!=1:raise ValueError('missing or duplicate E1 baseline')
    inputs=[json.loads(line) for line in inputs_bytes.splitlines()]
    if len(inputs)!=384:raise ValueError('E1 input inventory incomplete')
    matched=[r for r in inputs if (r.get('task'),str(r.get('qid')),r.get('group_id'))==key]
    if len(matched)!=1 or any(matched[0].get(k)!=sample.get(k) for k in ('image_path','image_sha256','message_sha256')):
        raise ValueError('E1 sample input mismatch')
    return found[0],matched[0],{'accepted_unique_line':True,'journal_sha256':jhash,'inputs_sha256':ihash,
                                 'acceptance_sha256':receipt.proof_sha,'review_sha256':receipt.review_sha,
                                 'weights_identity_sha256':receipt.weights_sha,'plan_sha256':receipt.plan_sha}
