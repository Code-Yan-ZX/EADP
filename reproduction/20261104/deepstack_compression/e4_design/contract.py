"""CPU-only rejection rules for proposed E4 correctness and timing records.

These validate reported evidence, not the GPU implementation itself.
"""
import math

MODES = ('C-ON', 'C-DROP', 'C-SKIP')
KS = (128, 256, 512, 1024)


def _require(ok, why):
    if not ok: raise ValueError(why)


def validate_arm(r):
    mode, k = r['mode'], r['k']
    _require(mode in MODES and k in KS, 'unsupported mode or budget')
    _require(r['n_input_visual']==1024 and r['n_retained_visual']==k, 'wrong visual budget')
    order, spatial = r['ordered_ids'], r['spatial_ids']
    _require(len(order)==k and len(set(order))==k and all(isinstance(i,int) and 0<=i<1024 for i in order), 'invalid selected IDs')
    _require(sorted(order)==spatial, 'spatial selection not sorted greedy IDs')
    if k==1024:
        _require(r['selection_mode']=='keep_all' and order==list(range(1024)), 'invalid keep-all path')
        _require(r['scorer_calls']==0 and r['greedy_calls']==0, 'keep-all unexpectedly computed selector')
    else:
        _require(r['selection_mode']=='eadp_greedy' and r['scorer_calls']==1 and r['greedy_calls']==1, 'missing K-specific original selection')
    _require(r['main_merger_calls']==1 and r['vision_blocks_executed']==r['vision_block_count']>0, 'visual backbone was skipped')
    _require(r['ds_original_merger_calls']==([0,0,0] if mode=='C-SKIP' else [1,1,1]), 'incorrect DS merger executions')
    _require(r['ds_injected_rows']==([k,k,k] if mode=='C-ON' else [0,0,0]), 'incorrect DS injection')
    _require(r['position_mode']=='P3' and r['decode_delta']==r['full_delta']+1024-k, 'incorrect P3 delta')
    _require(r['vision_mask_count']==k and r['prefill_length']==r['original_length']-1024+k, 'incorrect prefill span')
    _require(bool(r['output_ids']) and all(isinstance(x,int) for x in r['output_ids']), 'missing generated token IDs')
    _require(isinstance(r['eos'],bool) and isinstance(r['truncated'],bool), 'missing terminal flags')
    return True


def validate_modes(records):
    _require([r['mode'] for r in records]==list(MODES), 'three modes required in fixed order')
    for r in records: validate_arm(r)
    first=records[0]
    common=('task','qid','group_id','image_sha256','message_sha256','processor_sha256',
            'k','ordered_ids','spatial_ids','visual_sha256','prefill_embed_sha256',
            'attention_mask_sha256','position_sha256','decode_delta')
    _require(all(all(r[x]==first[x] for x in common) for r in records), 'common inputs/selection differ')
    off,skip=records[1:]
    _require(off['output_ids']==skip['output_ids'] and off['first_logits_argmax']==skip['first_logits_argmax'], 'drop/skip outputs diverge')
    _require(off['logits_comparison_passed'] and skip['logits_comparison_passed'], 'predeclared logits tolerance failed')
    return True


def validate_timing(r):
    """Require true first output rather than prompt or logits and a no-stream total."""
    _require(r['batch']==1 and r['gpu_exclusive'] is True, 'not the agreed exclusive single-item run')
    _require(r['prompt_callback_index']==0 and r['first_token_callback_index']==1, 'prompt confused with first token')
    _require(r['callback_kind']=='generated_token_after_cpu_transfer', 'not an actual emitted first token')
    _require(r['stream_output_ids'] and r['stream_output_ids']==r['plain_output_ids'], 'stream/no-stream generated IDs differ')
    _require(r['callback_token_id']==r['stream_output_ids'][0], 'callback did not observe first output token')
    _require(r['ttft_scope']=='before_image_read_to_first_token_cpu', 'TTFT scope mismatch')
    _require(r['total_scope']=='before_image_read_to_end_no_streamer', 'total scope mismatch')
    _require(r['observer_sync_inside_ttft']=='first_token_transfer_only', 'profile synchronization contaminated TTFT')
    _require(all(isinstance(r[x],(int,float)) and math.isfinite(r[x]) and r[x]>0 for x in ('ttft_s','unstreamed_total_s')), 'invalid timing')
    _require(r['warmup_complete'] is True and r['all_gpu_processes_recorded'] is True, 'missing warmup or GPU occupancy')
    return True
