"""Execute the frozen, answer-conditioned four-stream diagnostic."""
from __future__ import annotations
import json
import time
import torch
from experiment_common import (HERE, ROOT, OUT, K, MAX_TOKENS, SEED, NativeEngine,
    common, message, resolution, save_json, sha256, convert, prepare_sample)
from oracle import score_full_streams, generate_selection, tensor_sha256

ARMS = ('full','full_no_ds','rres512','b2','b2_no_ds','mag_v_on','mag_v_off',
        'mag_joint_on','mag_joint_off','signed_v_on','signed_v_off',
        'signed_joint_on','signed_joint_off','random_on','random_off')

def contrasts():
    pairs = [('mag_v_on','mag_joint_on'),('mag_v_off','mag_joint_off'),
             ('signed_v_on','signed_joint_on'),('signed_v_off','signed_joint_off'),
             ('full_no_ds','full'),('b2_no_ds','b2'),('random_off','random_on')]
    pairs += [(reference, teacher) for teacher in ('mag_v_on','mag_joint_on','signed_v_on','signed_joint_on')
              for reference in ('b2','rres512','random_on')]
    pairs += [(reference, teacher) for teacher in ('mag_v_off','mag_joint_off','signed_v_off','signed_joint_off')
              for reference in ('b2_no_ds','random_off')]
    return [list(pair) for pair in pairs]

def source_paths():
    local = [HERE / name for name in ('multistream_run.py','experiment_common.py','oracle.py',
        'test_oracle.py','gates.py','data.py','analysis.py','freeze.py','multistream_benchmark.py')]
    external = [ROOT / name for name in ('docs/multistream_oracle_prereg.md',
        'Qwen_vl/model/native_qwen3.py','Qwen_vl/model/e0_selectors.py','Qwen_vl/model/pruner.py',
        'Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py',
        'Qwen_vl/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py',
        'Qwen_vl/scripts/discovery/common.py','Qwen_vl/scripts/discovery/instrumented.py',
        'Qwen_vl/scripts/cf_mining/cf_mine.py','Qwen_vl/scripts/catchup_rerank/analyze.py')]
    return local + external

def verify_freeze(manifest_path):
    freeze = json.loads((OUT / 'run_manifest.json').read_text())
    if freeze['manifest_sha256'] != sha256(manifest_path):
        raise RuntimeError('frozen data manifest changed')
    if freeze['gates_sha256'] != sha256(OUT / 'gates.json'):
        raise RuntimeError('gates changed after freeze')
    gate = json.loads((OUT / 'gates.json').read_text())
    if not gate.get('pass_all') or gate.get('cpu_only') or gate.get('checkpointing') != freeze['checkpointing']:
        raise RuntimeError('gates have not passed')
    for path in source_paths():
        if freeze['source_sha256'].get(str(path)) != sha256(path):
            raise RuntimeError('frozen source changed: '+str(path))
    for path, digest in gate.get('source_sha256', {}).items():
        if sha256(path) != digest:
            raise RuntimeError('gated source changed: '+path)
    return freeze

def main(eng=None):
    from analysis import score_exact
    path = OUT / 'data/manifest.json'
    freeze = verify_freeze(path)
    items = json.loads(path.read_text())['items']
    if len(items) != 32 or len({x['parent_id'] for x in items}) != 32:
        raise ValueError('expected 32 unique frozen parents')
    config = dict(manifest_sha256=sha256(path), planned_arms=list(ARMS), contrasts=contrasts(),
                  ds_interactions=['mag','signed'], seed=SEED, K=K,
                  run_manifest_sha256=sha256(OUT/'run_manifest.json'))
    output = OUT / 'predictions.json'
    if output.exists():
        result = json.loads(output.read_text())
        if result['config'] != config:
            raise ValueError('output config differs from freeze')
    else:
        result = dict(config=config, records=[], complete=False)
    done = {row['item_id'] for row in result['records']}
    if done - {item['item_id'] for item in items}:
        raise ValueError('existing output contains unknown documents')
    if eng is None:
        eng = NativeEngine(common.load_model(common.BASELINE_MODEL,max_new_tokens=MAX_TOKENS))
    if eng.text.config._attn_implementation != 'sdpa':
        raise RuntimeError('attention backend differs from freeze')
    started = time.time()
    for i,item in enumerate(items):
        if item['item_id'] in done: continue
        torch.cuda.synchronize(); feature_started=time.perf_counter()
        sample = prepare_sample(eng,item)
        torch.cuda.synchronize(); feature_and_b2_seconds=time.perf_counter()-feature_started
        prep,V,DS=sample['prep'],sample['V'],sample['DS']
        expected=freeze['rendered_inputs'][item['item_id']]
        if prep['ids'][0].cpu().tolist()!=expected['input_ids'] or sample['random'].tolist()!=expected['random_keep']:
            raise RuntimeError('input or random selection changed')
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated()
        teacher = score_full_streams(eng,prep,V,DS,item['answers'][0],K=K,
                                     checkpointing=freeze['checkpointing'])
        if teacher['gold_token_ids'].tolist()!=expected['target_ids']:
            raise RuntimeError('teacher target token IDs changed')
        teacher['peak_memory'] = dict(baseline_allocated=baseline_mem,
            allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
        mapping={'mag_v':'magnitude_v','mag_joint':'magnitude_joint',
                 'signed_v':'signed_v','signed_joint':'signed_joint'}
        keep={key:teacher['keep_indices'][src] for key,src in mapping.items()}
        keep['b2']=sample['b2']; keep['random']=sample['random']
        teacher['target_text']=item['answers'][0]
        teacher['target_ids']=teacher['gold_token_ids']
        teacher['loss']=teacher['mean_answer_nll']
        teacher['stream_scores']={'mag':teacher['magnitude_per_stream'],'signed':teacher['signed_per_stream']}
        teacher['scores']={key:teacher['scores'][src] for key,src in mapping.items()}
        teacher['keep']=keep
        teacher['random_seed']=sample['random_seed']
        teacher['feature_extraction_and_b2_seconds']=feature_and_b2_seconds
        teacher['rank_order']={key:torch.argsort(teacher['scores'][key],descending=True,stable=True)[:K]
                               for key in mapping}
        positions=eng.positions_full(prep)[0]
        teacher['selected_inputs']={}
        for key,indices in keep.items():
            idx=indices.to(V.device); seq=eng.keep_sequence(prep,idx)
            teacher['selected_inputs'][key]=dict(V=tensor_sha256(V[idx]),
                DS=[tensor_sha256(d[idx]) for d in DS],
                positions=tensor_sha256(positions[:,:,seq]))
        teacher['random_overlap']={key:int(torch.isin(indices.cpu(),keep['random'].cpu()).sum())
                                   for key,indices in keep.items() if key!='random'}
        predictions={}
        for name in ARMS:
            if name=='rres512':
                with resolution(eng,512):
                    p=eng.prepare(message(item),'DocVQA_VAL'); v,d=eng.encode(p)
                    if p['n_vis']!=K: raise AssertionError('R-res must use 256 tokens')
                    pred=generate_selection(eng,p,v,d,max_tokens=MAX_TOKENS)
            else:
                if name in ('full','full_no_ds'): chosen=None
                elif name in ('b2','b2_no_ds'): chosen=keep['b2']
                else: chosen=keep[name.rsplit('_',1)[0]]
                ds_on=not (name.endswith('_off') or name.endswith('_no_ds'))
                pred=generate_selection(eng,prep,V,DS,chosen,deepstack=ds_on,max_tokens=MAX_TOKENS)
            pred['correct']=score_exact(pred['text'],item['answers'])
            pred['eos']=not pred['truncated']  # NativeEngine.decode stops before appending EOS.
            predictions[name]=pred
        for family in ('mag_v','mag_joint','signed_v','signed_joint','random'):
            a,b=predictions[family+'_on'],predictions[family+'_off']
            if a['positions_sha256']!=b['positions_sha256'] or not torch.equal(a['keep_idx'],b['keep_idx']):
                raise AssertionError('on/off pair changed positions or retained rows')
        row=dict(item_id=item['item_id'],parent_id=item['parent_id'],image_sha256=item['image_sha256'],
                 question=item['question'],answers=item['answers'],teacher=teacher,predictions=predictions)
        result['records'].append(convert(row));save_json(output,result)
        print(f"[{i+1}/32] {item['item_id']} " + ' '.join(f"{name}={int(predictions[name]['correct'])}" for name in ARMS)
              + f" elapsed={time.time()-started:.1f}s",flush=True)
    result['complete']=len(result['records'])==len(items)
    result['elapsed_seconds_this_process']=time.time()-started
    save_json(output,result)

if __name__=='__main__': main()
