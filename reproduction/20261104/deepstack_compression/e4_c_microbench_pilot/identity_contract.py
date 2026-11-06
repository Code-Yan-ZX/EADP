"""Compare real diagnostic capture to the independently accepted E1 P3-D1."""
def validate(actual,ref,tensor_meta):
 p=actual['processor'];expected=ref['processor'];keys=('input_ids','attention_mask','pixel_values','image_grid_thw')
 if set(expected)!=set(keys) or any(tensor_meta(p[k])!=expected[k] for k in keys):raise ValueError('processor mismatch')
 visual=actual['visual']
 if len(visual)!=2 or len(visual[1])!=3 or tensor_meta(visual[0])!=ref['visual_sha256'] or [tensor_meta(x) for x in visual[1]]!=ref['deepstack_full_sha256']:raise ValueError('main or DS visual identity')
 ordered=actual['ordered'].detach().cpu().reshape(-1).tolist()
 if ordered!=ref['ordered_ids'] or sorted(ordered)!=ref['spatial_ids'] or len(ordered)!=256:raise ValueError('selector identity')
 pre=actual['prefill'];positions=pre['position_ids'].detach().cpu().tolist();delta=actual['rope_delta'].detach().cpu().tolist();mask=pre['visual_pos_masks'].detach().cpu().tolist();streams=pre['deepstack_visual_embeds']
 if not (tensor_meta(pre['inputs_embeds'])==ref['prefill_embed_sha256'] and tensor_meta(pre['attention_mask'])==ref['attention_mask_sha256'] and positions==ref['prefill_position_ids'] and delta==ref['rope_delta'] and mask==ref['injected_mask'] and len(streams)==3 and [tensor_meta(x) for x in streams]==ref['injected_deepstack']):raise ValueError('actual prefill or injection identity')
 return {'status':'passed','processor':True,'main_visual':True,'three_full_ds':True,'ordered_and_spatial_ids':True,'actual_prefill_embeds_mask_positions_delta':True,'three_actual_injected_ds':True}
