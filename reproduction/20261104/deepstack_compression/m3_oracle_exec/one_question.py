"""One admitted M3 question. Data/model loading and run admission are external.

No file/dataset is read here; this function cannot by itself authorize access
to true labels or GPU. Caller must supply the validated author greedy order,
full verified visual streams, frozen model and preselected reference. All
caller callbacks are exercised under a bounded, separately supervised process.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import search,teacher_backend,generate_backend,baseline

def run(row,truth_loader,tokenizer,full,teacher,saliency,target_contract,position_contract,
        wrapper,pruner_module,author_generate,expected_scalars,expected_eos,journal,torch,baseline_record,baseline_provenance,checkpoint):
    if row.get('split')!='DEV' or len(full['streams'])!=4 or full['input_ids'].shape[0]!=1:
        raise ValueError('unverified DEV input')
    checkpoint()
    try:
        baseline_binding=baseline.validate(torch,wrapper.model,row,full,baseline_record,baseline_provenance,expected_scalars,expected_eos,
            lambda details:journal.append('e1_baseline_first_logits',**details))
    except Exception as exc:
        journal.append('e1_baseline_rejected',error_type=type(exc).__name__,message=str(exc));raise
    journal.append('e1_baseline_verified',**baseline_binding)
    # Caller supplies the first reference; target_contract forbids selecting an
    # alternative by loss or prediction and records truncation explicitly.
    checkpoint()
    journal.append('reference_access_started',task=row['task'],qid=str(row['qid']))
    truth=truth_loader(row)
    target=target_contract.encode_first_reference(row['task'],truth,tokenizer)
    target_ids=torch.tensor([target['token_ids']],device=full['input_ids'].device,dtype=torch.long)
    journal.append('target_encoded',token_ids=target['token_ids'],original_token_count=target['original_token_count'],
                   truncated=target['truncated'],convention=target['convention'])
    checkpoint()
    loss,util,gradient_evidence=teacher_backend.utilities(teacher,saliency,wrapper.model,full['input_ids'],full['start'],
        full['streams'],full['positions'],full['delta'],target_ids)
    journal.append('full_gradient_completed',mean_nll=loss,gradient_streams=4,utility=util,evidence=gradient_evidence)
    evaluate=teacher_backend.scorer(teacher,position_contract,wrapper.model,full['input_ids'],full['start'],
        full['streams'],full['positions'],full['delta'],target_ids)
    def generate(spatial,on_generated):
        return generate_backend.generate(wrapper,pruner_module,row,spatial,full['greedy_order'],on_generated,
                                         author_generate,torch,expected_scalars,expected_eos,full)
    return search.execute(str(row['qid']),full['greedy_order'],util,evaluate,generate,journal,baseline_binding,checkpoint)
