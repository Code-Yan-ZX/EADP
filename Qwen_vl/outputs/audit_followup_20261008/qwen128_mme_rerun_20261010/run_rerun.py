"""Isolated full MME repeat of the saved Qwen AnchorZip K128 recipe.

Uses the original generation and scoring functions, same frozen bank and
manual greedy decoder. Adds CPU provenance records after completed calls.
The original shard, scorer cache, source and comparison table are preserved.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = OUT / 'run01'
STAGE = ROOT / 'Qwen_vl/scripts/stage1_roundtrip_pilot'
ORIGINAL = ROOT / 'Qwen_vl/outputs/stage1_roundtrip_pilot_k128'
ARM = 'L_R_MAIN0125'
BANK = ORIGINAL / 'full/bank_full_rtg_MME.json.gz'
OLD = ORIGINAL / 'legacy_full/acc' / ARM / 'MME.json'
STATE = OUT / 'state.json'
LOCK = Path('/tmp/llava_full_driver.lock')
TOKEN = f'qwen128-mme-{os.getpid()}'
START = time.time()


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')
    tmp.replace(path)


state = dict(status='preparing', pid=os.getpid(), started_beijing=now(),
             expected=2374, completed=0, output_root=str(RUN))
write(STATE, state)
locked = False
try:
    assert not (RUN / 'legacy_full/acc' / ARM / 'MME.json').exists(), 'fresh output required'
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free',
                                  '--format=csv,noheader,nounits'],text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'],text=True).strip()
    assert not apps, f'Exclusive GPU required: {apps}'
    assert int(gpu.rsplit(',', 1)[1].strip()) >= 32000, gpu
    LOCK.mkdir()
    (LOCK / 'qwen128_mme_owner').write_text(TOKEN)
    locked = True
    old = json.loads(OLD.read_text())
    assert old['meta']['K'] == 128 and len(old['records']) == 2374
    assert digest(BANK) == old['meta']['bank_sha256']
    (RUN / 'full').mkdir(parents=True)
    (RUN / 'full/bank_full_rtg_MME.json.gz').symlink_to(BANK)
    sources = [Path(__file__), OLD, BANK, Path('/media/disk2/YZX/LMUData/MME.tsv'),
        STAGE / 'rtg_accuracy_legacy.py', STAGE / 'rtg_common.py',
        ROOT / 'Qwen_vl/scripts/anchor_completion_validation/acu_common.py',
        ROOT / 'Qwen_vl/scripts/anchor_completion_validation/acu_accuracy.py',
        ROOT / 'Qwen_vl/scripts/anchor_merge_pilot/amp_common.py',
        ROOT / 'Qwen_vl/scripts/discovery/common.py',
        ROOT / 'Qwen_vl/model/native_qwen3.py',
        ROOT / 'Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py',
        ROOT / 'Qwen_vl/VLMEvalKit/vlmeval/dataset/image_yorn.py',
        ROOT / 'Qwen_vl/VLMEvalKit/vlmeval/dataset/utils/yorn.py']
    hashes = {str(p): digest(p) for p in sources}
    protocol = dict(created_beijing=now(), task='Qwen3-VL-8B AnchorZip K128 MME repeat',
        user_authorized=True, gpu_mode='exclusive', gpu_at_launch=gpu, process_args=sys.argv,
        cwd=os.getcwd(), model='Qwen3-VL-8B-Instruct-1024', budget=128,
        recipe=dict(method='RTG + official facility + Completion',
                    lambda_completion=0.25, alpha=0.5, beta=2.0,
                    deepstack=False, position='1d', decoding='manual greedy',
                    max_new_tokens=2048, dataset='MME', n=2374),
        bank_reuse=str(BANK),source_sha256=hashes,
        old_shard=str(OLD),fresh_output_root=str(RUN),
        instrumentation='CPU identity/runtime logs after each completed generation; model math unchanged')
    write(OUT / 'protocol.json', protocol)
    sys.path.insert(0, str(STAGE))
    import rtg_common as RC
    RC.K = 128
    RC.OUT_DIR = str(RUN)
    original_cfg = RC.arm_cfg

    def cfg128(name):
        cfg = dict(original_cfg(name))
        cfg['K'] = 128
        return cfg

    RC.arm_cfg = cfg128
    import rtg_accuracy_legacy as RAL
    RAL.ARM = ARM
    RAL.FULL = str(RUN / 'legacy_full/acc' / ARM)
    RAL.FAIL_LOG = str(RUN / 'score_failures.jsonl')
    Path(RAL.FULL).mkdir(parents=True)
    original_load = RAL.AC.common.load_model

    def load_model(*args, **kwargs):
        state.update(status='loading', updated_beijing=now())
        write(STATE,state)
        wrapper = original_load(*args,**kwargs)
        import torch, transformers
        param = next(wrapper.model.parameters())
        snapshot = dict(created_beijing=now(),wrapper=type(wrapper).__name__,
            model_class=type(wrapper.model).__name__,model_name=args[0],
            dtype=str(param.dtype),device=str(param.device),
            attn=getattr(wrapper.model.config.text_config,'_attn_implementation',None),
            torch_version=torch.__version__,transformers_version=transformers.__version__,
            cfg=RC.arm_cfg('R_MAIN025'),legacy_flags=RAL.LEGACY_FLAGS,
            generate_kwargs=getattr(wrapper,'generate_kwargs',None),
            fixed_image_size=getattr(wrapper,'image_size',None),
            model_name_or_path=getattr(wrapper.model.config,'_name_or_path',None),
            text_config_eos=wrapper.model.config.text_config.eos_token_id,
            tokenizer_eos=wrapper.processor.tokenizer.eos_token_id,
            generation_config_eos=wrapper.model.generation_config.eos_token_id)
        assert snapshot['dtype'] == 'torch.bfloat16'
        assert snapshot['attn'] == 'sdpa'
        assert snapshot['cfg']['K'] == 128 and snapshot['cfg']['lam'] == 0.25
        assert snapshot['legacy_flags'] == dict(deepstack=False,pos='1d')
        write(OUT / 'loaded_runtime.json',snapshot)
        state.update(status='generating', updated_beijing=now(), generation_started=time.time())
        write(STATE,state)
        return wrapper

    RAL.AC.common.load_model = load_model
    original_run_one = RAL.acu_run_one
    trace = OUT / 'runtime.jsonl'
    image_hashes = {}

    def run_one(engine, message, ds, bank_record, cfg, *args, **kwargs):
        result = original_run_one(engine, message, ds, bank_record, cfg, *args, **kwargs)
        index = int(bank_record['idx'])
        assert int(result['state'].keep_idx.numel()) == 128
        input_images = []
        for item in message:
            if item['type'] == 'image':
                path = str(Path(item['value']).resolve())
                if path not in image_hashes:
                    image_hashes[path] = digest(path)
                input_images.append(dict(path=path,sha256=image_hashes[path]))
        record = dict(index=index, completed_beijing=now(),
            message=message, message_sha256=hashlib.sha256(
                json.dumps(message,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            input_images=input_images, cfg=cfg, legacy_flags=RAL.LEGACY_FLAGS,
            n_vis_kept=128, gen_tokens=len(result['gen_ids']),
            gen_ids=result['gen_ids'],engine_eos_ids=sorted(engine.eos_ids),
            truncated=len(result['gen_ids']) >= 2048,
            prediction_sha256=hashlib.sha256(result['text'].encode()).hexdigest(),
            invariants=result['meta'],timings=result['timings'])
        with trace.open('a') as stream:
            stream.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
        state.update(completed=index+1, updated_beijing=now(),
                     elapsed_generation_seconds=time.time()-state['generation_started'])
        if index%25 == 0 or index == 2373:
            write(STATE,state)
        return result

    RAL.acu_run_one = run_one
    RAL.run_gen(['MME'])
    state.update(status='scoring',updated_beijing=now())
    write(STATE,state)
    RAL.run_score(['MME'])
    import pandas as pd
    from vlmeval.dataset.utils.yorn import YOrN_Extraction
    fresh_path = Path(RAL.FULL) / 'MME.json'
    fresh = json.loads(fresh_path.read_text())
    assert len(fresh['records']) == 2374 and set(fresh['records']) == set(old['records'])
    gt = pd.read_csv('/media/disk2/YZX/LMUData/MME.tsv',sep='\t',keep_default_na=False,
                     usecols=['index','category','image_path','answer','question'])
    pred = pd.read_csv(Path(RAL.FULL) / 'MME_pred.tsv',sep='\t',keep_default_na=False)
    aux = pd.read_excel(Path(RAL.FULL) / 'MME_pred_auxmatch.xlsx',keep_default_na=False)
    assert len(gt) == len(pred) == len(aux) == 2374
    for column in gt.columns:
        assert gt[column].tolist() == pred[column].tolist()
    for column in pred.columns:
        assert pred[column].tolist() == aux[column].tolist()
    detail = []
    for i,row in gt.iterrows():
        a,b=old['records'][str(i)],fresh['records'][str(i)]
        assert b['n_vis_kept'] == 128
        old_yn,new_yn = YOrN_Extraction(a['prediction']),YOrN_Extraction(b['prediction'])
        assert new_yn == aux.iloc[i]['extracted']
        old_hit,new_hit = int(old_yn == row['answer']),int(new_yn == row['answer'])
        assert new_hit == int(aux.iloc[i]['score'])
        detail.append(dict(index=i,category=row['category'],image_path=row['image_path'],
            question=row['question'],gt=row['answer'],old_extracted=old_yn,new_extracted=new_yn,
            old_correct=old_hit,new_correct=new_hit,
            text_equal=a['prediction']==b['prediction'],old_truncated=a['truncated'],
            new_truncated=b['truncated']))
    comparison = pd.DataFrame(detail)
    comparison.to_csv(OUT / 'per_question_comparison.csv',index=False)
    comparison[~comparison['text_equal']].to_csv(OUT / 'changed_answers.csv',index=False)
    category_scores = {}
    for cat, group in comparison.groupby('category'):
        scores={}
        for label in ['old','new']:
            vals=group[f'{label}_correct']
            paired=group.groupby('image_path')[f'{label}_correct']
            assert paired.size().eq(2).all()
            scores[label]=float(vals.mean()*100 + paired.min().mean()*100)
        category_scores[cat]=dict(**scores,delta=scores['new']-scores['old'],
            rescues=int(((group.old_correct==0)&(group.new_correct==1)).sum()),
            breaks=int(((group.old_correct==1)&(group.new_correct==0)).sum()))
    perception=['OCR','artwork','celebrity','color','count','existence','landmark','position','posters','scene']
    reasoning=['code_reasoning','commonsense_reasoning','numerical_calculation','text_translation']
    summary={}
    for label in ['old','new']:
        p=sum(category_scores[c][label] for c in perception)
        c=sum(category_scores[c][label] for c in reasoning)
        summary[label]=dict(perception=p,reasoning=c,total=p+c)
    official=pd.read_csv(Path(RAL.FULL) / 'MME_pred_score.csv').iloc[0]
    assert abs(float(official.perception)-summary['new']['perception']) < 1e-9
    assert abs(float(official.reasoning)-summary['new']['reasoning']) < 1e-9
    for path,expected in hashes.items():
        assert digest(path) == expected, f'Frozen source changed: {path}'
    traces=[json.loads(line) for line in trace.read_text().splitlines()]
    assert len(traces)==2374 and [r['index'] for r in traces]==list(range(2374))
    assert all(r['n_vis_kept']==128 for r in traces)
    result=dict(complete=True,n=2374,finished_beijing=now(),elapsed_seconds=time.time()-START,
        summary=summary,total_delta=summary['new']['total']-summary['old']['total'],
        categories=category_scores,text_changes=int((~comparison.text_equal).sum()),
        extracted_changes=int((comparison.old_extracted!=comparison.new_extracted).sum()),
        rescues=int(((comparison.old_correct==0)&(comparison.new_correct==1)).sum()),
        breaks=int(((comparison.old_correct==1)&(comparison.new_correct==0)).sum()),
        truncated_new_ids=[i for i,r in fresh['records'].items() if r['truncated']],
        source_recheck='all_pass',fresh_shard_sha256=digest(fresh_path),
        runtime_sha256=digest(trace),old_shard_preserved=True,
        comparison_boundary='Same frozen bank and recipe; independent repeat of generation, not bank construction.')
    write(OUT / 'result.json',result)
    state.update(status='complete', completed=2374,updated_beijing=now(),exit_code=0,
                 result=str(OUT / 'result.json'),elapsed_seconds=time.time()-START)
    write(STATE,state)
    print('[RESULT] '+json.dumps(result,ensure_ascii=False),flush=True)
except BaseException as exc:
    state.update(status='failed',updated_beijing=now(),error=f'{type(exc).__name__}: {exc}')
    write(STATE,state)
    raise
finally:
    if locked and (LOCK / 'qwen128_mme_owner').read_text() == TOKEN:
        (LOCK / 'qwen128_mme_owner').unlink()
        LOCK.rmdir()
