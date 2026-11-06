"""64-case CPU-testable panel logic. No model loader, GPU admission or CLI."""
import json,os,time,traceback
from pathlib import Path

ARMS=('minimax','main_minimax')

def save(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())

def event(path,value):
    with Path(path).open('a',encoding='utf8') as f:
        json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())

def run(cohort,rows,references,out,preflight_one,run_one,complete):
    """Run callbacks after external admission; caller supplies all model dependencies.

    run_one(row, arm, ids, reference, on_generated, on_raw) calls on_raw exactly
    once immediately after model.generate returns, before any decode/validation.
    """
    out=Path(out)
    if not out.is_dir() or any((out/name).exists() for name in ('progress.jsonl','worker_complete.json','failure.json','observations')):
        raise ValueError('new exclusive output directory required')
    if len(cohort)!=32 or len(rows)!=32 or len(references)!=32:
        raise ValueError('32 input/reference denominator')
    by_key={(r['task'],str(r['qid'])):r for r in rows}
    if len(by_key)!=32 or len({(x['task'],str(x['qid'])) for x in cohort})!=32:
        raise ValueError('duplicate input')
    for item in cohort:
        k=(item['task'],str(item['qid']));r=by_key.get(k);ref=references.get(k)
        if r is None or ref is None or any(str(r.get(x))!=str(item.get(x)) for x in ('task','qid','group_id','image_sha256','message_sha256')):
            raise ValueError('cohort/row identity')
        if any(str(ref.get(x))!=str(item.get(x)) for x in ('task','qid','group_id','message_sha256')):
            raise ValueError('E1 reference identity')
    observations=out/'observations';observations.mkdir(exist_ok=False)
    completed=0
    try:
        for item in cohort:
            k=(item['task'],str(item['qid']));r=by_key[k];ref=references[k]
            preflight_one(r,ref)
            for arm in ARMS:
                tag=f"{item['index']:03d}_{arm}";raw_path=observations/(tag+'_raw.json');record_path=observations/(tag+'.json')
                event(out/'progress.jsonl',{'event':'started','index':item['index'],'task':item['task'],'qid':item['qid'],'arm':arm,'unix':time.time()})
                exposures=[];generated=[]
                def on_raw(payload):
                    # The full model.generate result has already been converted
                    # to CPU; durable persistence precedes all validation.
                    save(raw_path,payload);exposures.append(payload)
                def on_generated(payload):
                    if not raw_path.is_file():raise ValueError('raw return not persisted first')
                    generated.append(payload)
                result=run_one(r,arm,item['ids'][arm],ref,on_generated,on_raw)
                if len(exposures)!=1 or len(generated)!=1:
                    raise ValueError('missing or repeated actual generation')
                raw=exposures[0];batch=raw.get('returned_batch')
                if (not isinstance(batch,list) or len(batch)!=1 or not isinstance(batch[0],list) or
                    batch[0]!=result.get('output_ids') or batch[0]!=generated[0].get('output_ids') or
                    raw.get('processor_input_length')!=ref['processor']['input_ids']['shape'][1] or
                    raw.get('generate_input_ids_shape') is not None or
                    not isinstance(raw.get('generate_inputs_embeds_shape'),list) or
                    raw['generate_inputs_embeds_shape'][:2]!=[1,raw['processor_input_length']-768]):
                    raise ValueError('raw batch or actual generate input shape')
                if any(str(result.get(x))!=str(item.get(x)) for x in ('task','qid','group_id','image_sha256','message_sha256')):
                    raise ValueError('output identity')
                save(record_path,result);completed+=1
                event(out/'progress.jsonl',{'event':'recorded','index':item['index'],'task':item['task'],'qid':item['qid'],'arm':arm,'unix':time.time()})
        if completed!=64:raise ValueError('incomplete panel')
        complete()
        save(out/'worker_complete.json',{'status':'completed','questions':32,'generations':64,'GT_opened':False,'scoring_performed':False,'errors':[]})
        return completed
    except BaseException as exc:
        event(out/'progress.jsonl',{'event':'failed','completed':completed,'error_type':type(exc).__name__,'unix':time.time()})
        if not (out/'failure.json').exists():
            save(out/'failure.json',{'status':'failed','completed':completed,'intended':64,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'GT_opened':False})
        raise
