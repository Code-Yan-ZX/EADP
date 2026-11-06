"""Temporary, single-image P3/DeepStack intervention for E0 diagnostics.

The stock Qwen model's state is restored after each invocation. A failure to
observe exact source features or the original selection aborts the run.
"""
class Adapt:
    def __init__(self, torch, wrapper, keep, ds, observer=None):
        if keep not in (256, 1024) or ds not in ('on', 'off', 'zero'):
            raise ValueError('unsupported intervention')
        self.torch, self.w, self.keep, self.ds, self.observer = torch, wrapper, keep, ds, observer
        self.state = {'llm_calls': 0}
        self.handles = []

    def __enter__(self):
        self.orig_processor = self.w._processor_inputs
        def processor(*a, **kw):
            x = self.orig_processor(*a, **kw)
            ids = x['input_ids']
            if ids.shape[0] != 1 or x.get('pixel_values_videos') is not None:
                raise ValueError('only one image and no video supported')
            pos = (ids[0] == self.w.model.config.image_token_id).nonzero(as_tuple=True)[0]
            if pos.numel() != 1024 or not self.torch.equal(pos, self.torch.arange(pos[0], pos[0]+1024, device=pos.device)):
                raise ValueError('expected exactly one contiguous 1024 token image')
            self.state.update(inputs=x, start=int(pos[0]))
            return x
        self.w._processor_inputs = processor
        self.handles.append(self.w.model.visual.register_forward_hook(self.visual))
        self.handles.append(self.w.pruner.register_forward_hook(self.pruned))
        self.handles.append(self.w.model.model.language_model.register_forward_pre_hook(self.prefill, with_kwargs=True))
        return self

    def visual(self, module, args, output):
        self.state['visual'], self.state['deepstack'] = output

    def pruned(self, module, args, output):
        self.state['pruned'], self.state['splits'] = output

    def prefill(self, module, args, kw):
        s, t = self.state, self.torch
        s['llm_calls'] += 1
        if s['llm_calls'] != 1:
            return
        x, start = s['inputs'], s['start']
        if self.keep == 1024:
            spatial = t.arange(1024, device=s['visual'].device)
            if s['splits'] != [1024] or not t.equal(s['visual'], s['pruned']):
                raise ValueError('keep-all pruner did not preserve features')
            ordered = spatial
        else:
            obs = self.observer
            if obs.errors or obs.calls != 1 or len(obs.results) != 1 or s['splits'] != [self.keep]:
                raise ValueError('missing or ambiguous EADP greedy observation')
            raw = obs.results[0]
            ordered = raw['ordered_ids'].reshape(-1)
            if raw['quotas'].tolist() != [self.keep] or ordered.numel() != self.keep:
                raise ValueError('unexpected greedy quota')
            ids = ordered.detach().cpu().tolist()
            if len(set(ids)) != self.keep or any(x < 0 or x >= 1024 for x in ids):
                raise ValueError('greedy IDs repeated or out of range')
            spatial = ordered.sort().values
            if not t.equal(s['visual'].index_select(0, spatial), s['pruned']):
                raise ValueError('greedy IDs do not reconstruct actual pruner output')
        full_pos, full_delta = self.w.model.model.get_rope_index(
            x['input_ids'], x['image_grid_thw'], None, attention_mask=x['attention_mask'])
        seq = x['input_ids'].shape[1]
        retained = t.cat((t.arange(start, device=spatial.device), start + spatial,
                          t.arange(start+1024, seq, device=spatial.device)))
        chosen_pos = full_pos.index_select(-1, retained)
        if kw['inputs_embeds'].shape[1] != retained.numel():
            raise ValueError('sequence length/selection mismatch')
        original = self.w.model.get_input_embeddings()(x['input_ids'])
        expected = t.cat((original[:, :start], s['visual'].index_select(0, spatial).unsqueeze(0), original[:, start+1024:]), dim=1)
        if not t.equal(expected, kw['inputs_embeds']):
            raise ValueError('compacted inputs do not equal actual selected features')
        expected_mask = x['attention_mask'].index_select(1, retained)
        if not t.equal(expected_mask, kw['attention_mask']):
            raise ValueError('attention mask mismatch')
        selected = [z.index_select(0, spatial) for z in s['deepstack']]
        if len(selected) != 3 or any(z.shape != s['pruned'].shape for z in selected):
            raise ValueError('three DeepStack streams expected')
        kw = dict(kw)
        kw['position_ids'] = chosen_pos
        self.w.model.model.rope_deltas = full_delta + (1024-self.keep)
        if self.ds != 'off':
            mask = t.zeros(kw['inputs_embeds'].shape[:2], dtype=t.bool, device=spatial.device)
            mask[:, start:start+self.keep] = True
            kw['visual_pos_masks'] = mask
            kw['deepstack_visual_embeds'] = selected if self.ds == 'on' else [t.zeros_like(z) for z in selected]
        s.update(ordered=ordered, spatial=spatial, retained=retained, position_ids=chosen_pos,
                 full_delta=full_delta, adjusted_delta=self.w.model.model.rope_deltas,
                 actual_prefill=kw, selected=selected)
        return args, kw

    def __exit__(self, typ, value, traceback):
        self.w._processor_inputs = self.orig_processor
        for h in self.handles:
            h.remove()
        return False
