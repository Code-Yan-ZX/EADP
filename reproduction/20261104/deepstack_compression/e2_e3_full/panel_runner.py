"""Execution core for an E2/E3 DEV panel; no launch or admission authority.

A caller must establish complete smoke/B proofs and all source, input, GPU,
container, deadline, and resource gates before loading or invoking this core.
No GT, scoring, resumption, or performance measurement is implemented here.
"""
import hashlib,json,os
from pathlib import Path
from full_imports import local
_contract=local('contract');E2=_contract.E2;E3=_contract.E3;record_digest=_contract.record_digest

ARMS=tuple([('E2',x) for x in E2]+[('E3',x) for x in E3])
JOURNAL_FIELDS=('task','qid','group_id','budget','ds_on','output_ids','eos_ids','generation_scalars','forced','family','arm')
E1_INPUT_FIELDS={'processor':'processor','ordered_ids':'ordered_ids','spatial_ids':'spatial_ids',
 'retained_ids':'retained_ids','main_visual':'visual_sha256','pruned_visual':'pruned_sha256',
 'full_deepstack':'deepstack_full_sha256','prefill_position_ids':'prefill_position_ids',
 'adjusted_delta':'rope_delta','prefill_embeds':'prefill_embed_sha256','prefill_mask':'attention_mask_sha256'}

def append(path,record):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def validate_against_e1(record,row,e1_reference):
    for key in ('task','qid','group_id','image_sha256','message_sha256'):
        if str(record[key])!=str(row[key]):raise ValueError('question identity mismatch: '+key)
    if record['forced'] is not False or record['budget']!=256 or record['ds_on'] is not True:
        raise ValueError('incorrect full panel generation mode')
    for key,ref_key in E1_INPUT_FIELDS.items():
        if record[key]!=e1_reference[ref_key]:raise ValueError('E1 input/prefill mismatch: '+key)
    if record['actual_injection_verified'] is not True:raise ValueError('actual injection not verified')
    ids=record['output_ids']; eos=record['eos_ids']; end=bool(ids and ids[-1] in eos)
    if not ids or len(ids)>2048 or record['eos']!=end or record['truncated']!=bool(len(ids)==2048 and not end) or not (end or record['truncated']):
        raise ValueError('natural output incomplete')
    if record['generation_scalars']!=e1_reference['generation_scalars']:raise ValueError('recipe differs from E1')

def validate_endpoint(record,e1_reference):
    for key in ('output_ids','answer','eos_ids','eos','truncated'):
        if record[key]!=e1_reference[key]:raise ValueError('E1 endpoint differs: '+key)
    logits=e1_reference.get('logits')
    if not isinstance(logits,list) or len(logits)!=1 or record['prefill_logits']!=logits[0]:
        raise ValueError('E1 first-logit differs')

def run_panel(*,rows,e1_refs,anchors,out,invoke,archive,summary):
    """Generate ordered records, writing predictions immediately on return.

    ``invoke(row,family,arm,on_generated,archive_arrays,need_logits)`` invokes
    the separately reviewed model core. ``archive`` saves complete tensor
    archives only for predetermined anchors; ``summary`` returns fixed tensor
    metadata for all other questions. The out directory must be new and empty.
    """
    out=Path(out);out.mkdir(exist_ok=False);(out/'arrays').mkdir()
    anchor_keys={(r['task'],str(r['qid'])) for r in anchors}
    if len(anchor_keys)!=3 or not anchor_keys.issubset({(r['task'],str(r['qid'])) for r in rows}):raise ValueError('bad fixed anchors')
    records_count=0;question_count=0
    for row in rows:
        key=(row['task'],str(row['qid'])); d0=e1_refs[(*key,'P3-D0')];d1=e1_refs[(*key,'P3-D1')];byarm={}
        for family,arm in ARMS:
            written=[]
            def journal(item):
                if written:raise ValueError('multiple generated events')
                event=dict(item,family=family,arm=arm)
                if set(event)!=set(JOURNAL_FIELDS):raise ValueError('generation event fields differ')
                append(out/'generated.jsonl',event);written.append(event)
            def capture(tensors):
                if key in anchor_keys:
                    token=hashlib.sha256(json.dumps([*key,family,arm],separators=(',',':')).encode()).hexdigest()
                    return archive(tensors,out/'arrays'/(token+'.npz'))
                return {'kind':'summary_only','tensors':summary(tensors)}
            rec=invoke(row,family,arm,journal,capture,family=='E2' and arm in ('000','111'))
            if len(written)!=1 or written[0]!={k:rec[k] for k in JOURNAL_FIELDS}:raise ValueError('generation journal does not match observation')
            if (rec['family'],rec['arm'])!=(family,arm):raise ValueError('arm identity mismatch')
            validate_against_e1(rec,row,d1)
            if family=='E2' and arm in ('000','111'):validate_endpoint(rec,d0 if arm=='000' else d1)
            expected_kind='full_arrays' if key in anchor_keys else 'summary_only'
            if rec['arrays']['kind']!=expected_kind:raise ValueError('wrong archive policy')
            append(out/'records.jsonl',rec);byarm[(family,arm)]=rec;records_count+=1
        aliases={}
        for mode,pair in (('OFF',('E2','000')),('MATCH',('E2','111')),('PERM',('E3','PERM')),('MEAN',('E3','MEAN'))):
            aliases[mode]={'source_family':pair[0],'source_arm':pair[1],'record_sha256':record_digest(byarm[pair])}
        append(out/'e3_views.jsonl',{'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'arms':aliases});question_count+=1
    if question_count!=384 or records_count!=3840:raise ValueError('wrong full panel denominator')
    result={'questions':question_count,'physical_generations':records_count,'GT_opened':False,'scoring_performed':False}
    with (out/'worker_complete.json').open('x') as f:json.dump(result,f);f.write('\n');f.flush();os.fsync(f.fileno())
    return result
