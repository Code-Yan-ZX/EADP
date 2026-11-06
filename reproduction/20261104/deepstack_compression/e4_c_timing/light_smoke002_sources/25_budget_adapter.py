"""Instrumented E4-A budget adapter; identity diagnostics, not timing."""
from off_adapter import E4OffAdapter


class BudgetAdapter(E4OffAdapter):
    def __init__(self, torch, wrapper, pruner_module, budget, ds_on):
        if budget not in (128, 256, 512, 1024) or type(ds_on) is not bool:
            raise ValueError('unsupported budget or DS mode')
        super().__init__(torch, wrapper, pruner_module, 'C-DROP')
        self.budget, self.ds_on = budget, ds_on

    def __enter__(self):
        super().__enter__()
        previous=self.w._processor_inputs
        def snapshot(*args,**kwargs):
            inputs=previous(*args,**kwargs)
            self.state['inputs']={k:(v.detach().clone() if hasattr(v,'detach') else v) for k,v in inputs.items()}
            return inputs
        self.w._processor_inputs=snapshot
        self.stack.callback(setattr,self.w,'_processor_inputs',previous)
        return self

    def _prefill(self, _module, args, kw):
        s, t, k = self.state, self.torch, self.budget
        s['llm_calls'] += 1
        if s['llm_calls'] != 1:
            return None
        if s['visual_calls'] != 1 or s['splits'] != [k]:
            raise ValueError('missing actual visual/pruner result')
        if k == 1024:
            if s['greedy_calls'] != 0:
                raise ValueError('keep-all must bypass greedy')
            spatial = t.arange(1024, device=s['visual'].device)
            s.update(ordered=spatial.clone(), quotas=None, selection_mode='keep_all')
        else:
            if s['greedy_calls'] != 1:
                raise ValueError('expected one actual author greedy call')
            ids = s['ordered'].detach().cpu().tolist()
            if len(ids) != k or len(set(ids)) != k or any(x < 0 or x >= 1024 for x in ids) or s['quotas'].tolist() != [k]:
                raise ValueError('invalid actual greedy selection')
            spatial = s['ordered'].sort().values
            s['selection_mode'] = 'author_greedy'
        if not t.equal(s['visual'].index_select(0, spatial), s['pruned']):
            raise ValueError('selected visual features mismatch')
        x, start = s['inputs'], s['start']
        full, delta = self.w.model.model.get_rope_index(x['input_ids'], x['image_grid_thw'], None, attention_mask=x['attention_mask'])
        retained = t.cat((t.arange(start, device=spatial.device), start + spatial,
                          t.arange(start + 1024, x['input_ids'].shape[1], device=spatial.device)))
        embeds = self.w.model.get_input_embeddings()(x['input_ids'])
        expected = t.cat((embeds[:, :start], s['pruned'].unsqueeze(0), embeds[:, start + 1024:]), dim=1)
        if not t.equal(expected, kw['inputs_embeds']) or not t.equal(x['attention_mask'].index_select(1, retained), kw['attention_mask']):
            raise ValueError('actual prefill features/mask mismatch')
        streams = s['deepstack']
        if len(streams) != 3 or any(v.shape != s['visual'].shape for v in streams):
            raise ValueError('expected three full aligned DS streams')
        selected = [v.index_select(0, spatial) for v in streams]
        mask = t.zeros_like(kw['attention_mask'], dtype=t.bool)
        mask[:, start:start+k] = True
        kw = dict(kw)
        kw['position_ids'] = full.index_select(-1, retained)
        kw['visual_pos_masks'] = mask if self.ds_on else None
        kw['deepstack_visual_embeds'] = selected if self.ds_on else None
        self.w.model.model.rope_deltas = delta + 1024-k
        s.update(spatial=spatial, retained=retained, full_pos=full, full_delta=delta,
                 selected_deepstack=selected, intended_ds_mask=mask if self.ds_on else None,
                 prefill_position_ids=kw['position_ids'], prefill_embeds=kw['inputs_embeds'],
                 prefill_mask=kw['attention_mask'], adjusted_delta=self.w.model.model.rope_deltas)
        return args, kw

    def _observe_prefill(self, _module, _args, kw):
        s, t = self.state, self.torch
        if s.get('actual_injection_verified'):
            return
        if not t.equal(kw['position_ids'], s['prefill_position_ids']):
            raise ValueError('actual position mismatch')
        mask, streams = kw.get('visual_pos_masks'), kw.get('deepstack_visual_embeds')
        if self.ds_on:
            if mask is None or not t.equal(mask, s['intended_ds_mask']) or not isinstance(streams, (tuple, list)) or len(streams) != 3:
                raise ValueError('actual DS injection mask/streams mismatch')
            if any(not t.equal(a, b) for a, b in zip(streams, s['selected_deepstack'])):
                raise ValueError('actual DS selected features mismatch')
        elif mask is not None or streams is not None:
            raise ValueError('unexpected DS injection')
        # The language model may mutate its input tensors in-place. Snapshot
        # the observed pre-hook values before entering its forward method.
        s['prefill_embeds']=kw['inputs_embeds'].detach().clone()
        s['prefill_mask']=kw['attention_mask'].detach().clone()
        s['prefill_position_ids']=kw['position_ids'].detach().clone()
        s['adjusted_delta']=self.w.model.model.rope_deltas.detach().clone()
        s['selected_deepstack']=[x.detach().clone() for x in s['selected_deepstack']]
        s['actual_ds_mask']=None if mask is None else mask.detach().clone()
        s['actual_deepstack']=None if streams is None else [x.detach().clone() for x in streams]
        s['actual_injection_verified'] = True

    def verify_calls(self):
        s = self.state
        if (s['visual_calls'] != 1 or s['main_merger_calls'] != 1 or not s['llm_calls']
                or s['greedy_calls'] != (0 if self.budget == 1024 else 1)
                or not s.get('actual_injection_verified')):
            raise ValueError('missing actual budget path')
        if any(n != 1 for n in s['block_calls']) or s['ds_calls'] != [1, 1, 1]:
            raise ValueError('unexpected visual block/DS calls')
        if sorted(s['ds_results']) != [(0, False), (1, False), (2, False)]:
            raise ValueError('DS mergers were not computed')
        return True
