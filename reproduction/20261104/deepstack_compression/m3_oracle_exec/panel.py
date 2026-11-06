"""Panel core: 128 frozen DEV images, fail-stop, no automatic resume or scoring.

The host launcher and independent runtime approval must have been completed
before this callable is used. The caller loads the reviewed frozen model and
passes its lazy reference loader. It must impose a process/container hard limit.
"""
import hashlib,json,os
from pathlib import Path
from collections import Counter
import e1_admission,e1_baseline,extract,one_question,run_control,search

def run(sample_bytes,expected_sample_sha,proof_bytes,journal_bytes,inputs_bytes,receipt,
        out,budget_end,question_seconds,wrapper,pruner_module,torch,teacher,saliency,target_contract,
        position_contract,truth_loader,author_generate,expected_scalars,expected_eos,validate_input):
    if not isinstance(receipt,e1_admission.Receipt):raise ValueError('E1 evidence gate receipt required')
    if hashlib.sha256(sample_bytes).hexdigest()!=expected_sample_sha:raise ValueError('frozen sample changed')
    sample=json.loads(sample_bytes);rows=sample['examples']
    if (sample.get('GT_opened') is not False or sample.get('predictions_opened') is not False or len(rows)!=128 or
        Counter(x['task'] for x in rows)!={'DocVQA_VAL':64,'TextVQA_VAL':32,'AI2D_TEST':32} or
        len({(x['task'],x['group_id']) for x in rows})!=128):raise ValueError('invalid frozen sample')
    out=Path(out);out.mkdir(exist_ok=False);events=search.Journal(out/'panel_events.jsonl')
    events.append('panel_started',sample_sha256=expected_sample_sha,n_expected=128,baseline='accepted E1 P3-D1',
                  maximum_new_generations=768,GT_opened=False,scoring=False)
    try:
        for index,sample_row in enumerate(rows):
            control=run_control.Deadline(budget_end,question_seconds);control.check()
            raw,input_row,provenance=e1_baseline.select(proof_bytes,journal_bytes,inputs_bytes,sample_row,receipt)
            # E1's verified inputs omit split; the independently frozen sample
            # is DEV-only and matched by task/qid/group/image/message above.
            input_row=dict(input_row,split='DEV')
            key=[sample_row['task'],str(sample_row['qid']),sample_row['group_id']]
            name=hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()+'.jsonl'
            q=search.Journal(out/name)
            try:
                events.append('question_started',index=index,task=key[0],qid=key[1],group_id=key[2],journal=name)
                if e1_admission.sha(input_row['image_path'])!=input_row['image_sha256']:raise ValueError('frozen image changed')
                control.check();full=extract.extract(torch,wrapper,pruner_module,input_row,validate_input)
                result=one_question.run(input_row,truth_loader,wrapper.processor.tokenizer,full,teacher,saliency,target_contract,
                    position_contract,wrapper,pruner_module,author_generate,expected_scalars,expected_eos,q,torch,raw,provenance,control.check)
                if len(result['outputs'])!=6:raise ValueError('expected six new outputs')
                events.append('question_completed',index=index,task=key[0],qid=key[1],group_id=key[2],journal=name,new_generations=6)
            except BaseException as exc:
                q.append('question_failed',error_type=type(exc).__name__,message=str(exc))
                events.append('panel_failed',index=index,error_type=type(exc).__name__,message=str(exc));raise
            finally:q.close()
        events.append('panel_completed',questions=128,new_generations=768,scoring=False)
        done={'status':'completed','questions':128,'new_generations':768,'scoring_performed':False,
              'sample_sha256':expected_sample_sha,'e1_completion_sha256':receipt.proof_sha}
        with (out/'worker_complete.json').open('x') as f:json.dump(done,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        return done
    finally:events.close()
