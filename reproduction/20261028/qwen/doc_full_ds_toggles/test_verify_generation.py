"""Synthetic CPU tests of full-generation observation contracts; no model."""
import copy
import json
import unittest

from verify_generation import FIELDS, check_observation, check_prior_anchor, ids_sha, mask_sha


def meta(dtype, shape, byte='a'):
    return dict(dtype=dtype, shape=shape, sha256=byte * 64)


def fixture(arm):
    ids = [1] + [99] * 1024 + [2]
    mask = [x == 99 for x in ids]
    shape = [1, len(ids)]
    proc = dict(input_ids=dict(dtype='torch.int64', shape=shape, sha256=ids_sha(ids)),
        attention_mask=meta('torch.int64', shape), pixel_values=meta('torch.bfloat16', [4096, 1536]),
        image_grid_thw=meta('torch.int64', [1, 3]))
    item = dict(position=0, index='123', processor_expected=proc)
    prediction = dict(position=0, index='123', prediction='sample', native_failure=None)
    original = {key: None for key in FIELDS}
    original.update(inputs_embeds=meta('torch.bfloat16', [1, len(ids), 4096]),
        position_ids=meta('torch.int64', [3, 1, len(ids)]), attention_mask=proc['attention_mask'],
        cache_position=meta('torch.int64', [len(ids)]))
    mask_meta = dict(dtype='torch.bool', shape=shape, sha256=mask_sha(mask))
    visual = [meta('torch.bfloat16', [1024, 4096], str(i)) for i in range(3)]
    final = copy.deepcopy(original)
    if arm.startswith('plain'):
        original['visual_pos_masks'] = mask_meta
        for i in range(3): original['deepstack_' + str(i)] = visual[i]
    if arm in ('plain', 'eadp1024_ds_on'):
        final['visual_pos_masks'] = mask_meta
        for i in range(3): final['deepstack_' + str(i)] = copy.deepcopy(visual[i])
    scalars = dict(do_sample=False, max_new_tokens=2048, top_p=.001, top_k=1, temperature=.01, repetition_penalty=1.)
    input_keys = {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'} if arm.startswith('plain') else {'inputs_embeds', 'attention_mask'}
    obs = dict(position=0, index='123', arm=arm, native_failure=None, observer_errors=[],
        actual_processor=proc, input_ids=ids, image_mask=mask, image_mask_true_count=1024,
        image_mask_shape=shape, image_mask_sha256=mask_sha(mask), prefill_original=original,
        prefill_proposed=copy.deepcopy(final), prefill_final=final,
        vision_features=meta('torch.bfloat16', [1024, 4096]), visual_deepstack=visual,
        pruner_calls=0 if arm.startswith('plain') else 1,
        pruner_split_sizes=None if arm.startswith('plain') else [1024],
        pruner_input_features=None if arm.startswith('plain') else meta('torch.bfloat16', [1024, 4096]),
        pruner_output_features=None if arm.startswith('plain') else meta('torch.bfloat16', [1024, 4096]),
        intervention_attempts=int(arm in ('plain_ds_off','eadp1024_ds_on')),
        intervention_applied=arm in ('plain_ds_off','eadp1024_ds_on'), intervention_error=None,
        vision_calls=1, processor_calls=1, generate_calls=1, llm_calls=2, continuation_steps=1,
        continuation=[dict(position_ids=meta('torch.int64', [3,1,1]), cache_position=meta('torch.int64', [1]),
            rope_deltas=meta('torch.int64', [1,1]), visual_pos_masks_present=False, deepstack_visual_embeds_present=False)],
        generate_non_tensor_kwargs=scalars, generate_kwargs_keys=list(set(scalars) | input_keys),
        generate_tensor_kwargs=copy.deepcopy(proc) if arm.startswith('plain') else
            {'inputs_embeds':copy.deepcopy(original['inputs_embeds']), 'attention_mask':copy.deepcopy(proc['attention_mask'])},
        rope_deltas=meta('torch.int64', [1, 1]), output_suffix_ids=[3, 4],
        output_ids=ids + [3, 4] if arm.startswith('plain') else [3, 4], native_redecode='sample')
    return item, prediction, obs


class TestGenerationAudit(unittest.TestCase):
    def test_real_fixed16_archives_and_schema_without_model_or_GT(self):
        import common as c
        import numpy as np
        old_plan = c.read(c.OLD / 'run_plan.json')
        config = c.read(__import__('pathlib').Path(old_plan['recipe']['model_path']) / 'config.json')
        image_id, vocab = config['image_token_id'], config['text_config']['vocab_size']
        for index in (0, 7):
            for arm in ('plain', 'eadp1024', 'plain_ds_off', 'eadp1024_ds_on'):
                prior = c.read(c.PANEL / 'execution' / ('%02d_%s.json' % (index, arm)))
                a = prior['arrays']
                with np.load(prior['archive'], allow_pickle=False) as archive:
                    ids = archive['processor_input_ids'][0].tolist()
                    full = archive['output_ids'][0].tolist()
                    suffix = archive['output_suffix_ids'][0].tolist()
                    mask = archive['injection_image_mask'][0].tolist()
                proc = {k:a['processor_' + k] for k in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')}
                fields = lambda prefix: {k:a.get(prefix + k) for k in FIELDS}
                steps = []
                for j, presence in enumerate(prior['continuation_presence']):
                    steps.append(dict(position_ids=a.get('continuation_%04d_position_ids' % j),
                        cache_position=a.get('continuation_%04d_cache_position' % j),
                        rope_deltas=a.get('continuation_%04d_rope_deltas' % j),
                        visual_pos_masks_present=presence['visual_mask_present'],
                        deepstack_visual_embeds_present=presence['deepstack_present']))
                item = dict(index=prior['index'], position=prior['position'], processor_expected=proc)
                prediction = dict(index=prior['index'], position=prior['position'],
                                  prediction=prior['response'], native_failure=None)
                scalar = prior['generate_non_tensor_kwargs']
                tensor_keys = set(prior['generate_kwargs_keys']) - set(scalar)
                obs = dict(index=prior['index'], position=prior['position'], arm=arm, native_failure=None,
                    observer_errors=[], actual_processor=proc, input_ids=ids, output_ids=full,
                    output_suffix_ids=suffix, native_redecode=prior['response'], image_mask=mask,
                    image_mask_sha256=mask_sha(mask), image_mask_shape=[1, len(mask)], image_mask_true_count=sum(mask),
                    prefill_original=fields('prefill_original_'), prefill_proposed=fields('prefill_proposed_'),
                    prefill_final=fields('prefill_'), vision_features=a['visual_features'],
                    visual_deepstack=[a['visual_deepstack_' + str(i)] for i in range(3)],
                    # Prior factorial did not save pruner hook fields. These
                    # stand-ins exercise the new contract, not historical proof.
                    pruner_calls=0 if arm.startswith('plain') else 1,
                    pruner_split_sizes=None if arm.startswith('plain') else [1024],
                    pruner_input_features=None if arm.startswith('plain') else a['visual_features'],
                    pruner_output_features=None if arm.startswith('plain') else a['visual_features'],
                    rope_deltas=a['rope_deltas'], generate_kwargs_keys=prior['generate_kwargs_keys'],
                    generate_non_tensor_kwargs=scalar, generate_tensor_kwargs={k:a['generate_' + k] for k in tensor_keys},
                    intervention_attempts=prior['intervention_attempts'], intervention_applied=prior['intervention_applied'],
                    intervention_error=prior['intervention_error'], vision_calls=prior['vision_calls'],
                    processor_calls=prior['processor_calls'], generate_calls=prior['generate_calls'],
                    llm_calls=prior['llm_calls'], continuation_steps=prior['continuation_steps'], continuation=steps)
                check_observation(item, prediction, obs, arm, image_id, vocab)
                check_prior_anchor(c, index, arm, prediction, obs)

    def test_both_arms(self):
        for arm in ('plain', 'eadp1024', 'plain_ds_off', 'eadp1024_ds_on'):
            item, prediction, obs = fixture(arm)
            check_observation(item, prediction, obs, arm, 99, 100)

    def test_mask_and_ids_cannot_be_forged(self):
        item, prediction, obs = fixture('eadp1024_ds_on')
        obs['image_mask'][1] = False
        with self.assertRaises(AssertionError):
            check_observation(item, prediction, obs, 'eadp1024_ds_on', 99, 100)

    def test_plain_generation_prefix_and_continuation_are_checked(self):
        item, prediction, obs = fixture('plain_ds_off')
        obs['output_ids'][0] = 4
        with self.assertRaises(AssertionError):
            check_observation(item, prediction, obs, 'plain_ds_off', 99, 100)

    def test_continuation_requires_actual_metadata_and_no_deepstack(self):
        for field, value in [('position_ids', None), ('cache_position', None), ('rope_deltas', None),
                             ('visual_pos_masks_present', True), ('deepstack_visual_embeds_present', True)]:
            item, prediction, obs = fixture('plain_ds_off')
            obs['continuation'][0][field] = value
            with self.assertRaises(AssertionError):
                check_observation(item, prediction, obs, 'plain_ds_off', 99, 100)
        item, prediction, obs = fixture('plain_ds_off')
        obs['continuation'][0]['position_ids']['shape'] = [1, 1, 1]
        with self.assertRaises(AssertionError):
            check_observation(item, prediction, obs, 'plain_ds_off', 99, 100)
        # A different actual position hash is still an observation, not an
        # automatic failure; only fixed16 anchors require historical identity.
        item, prediction, obs = fixture('plain_ds_off')
        obs['continuation'][0]['position_ids']['sha256'] = 'f' * 64
        check_observation(item, prediction, obs, 'plain_ds_off', 99, 100)
        item, prediction, obs = fixture('plain_ds_off')
        obs['continuation'] = []
        with self.assertRaises(AssertionError):
            check_observation(item, prediction, obs, 'plain_ds_off', 99, 100)

    def test_position_change_and_wrong_intervention_rejected(self):
        item, prediction, obs = fixture('plain_ds_off')
        obs['prefill_final']['position_ids']['sha256'] = 'f' * 64
        obs['prefill_proposed']['position_ids']['sha256'] = 'f' * 64
        with self.assertRaises(AssertionError):
            check_observation(item, prediction, obs, 'plain_ds_off', 99, 100)
        item, prediction, obs = fixture('plain_ds_off')
        obs['intervention_applied'] = False
        with self.assertRaises(AssertionError):
            check_observation(item, prediction, obs, 'plain_ds_off', 99, 100)

    def test_injection_must_match_real_vision_metadata(self):
        item, prediction, obs = fixture('eadp1024_ds_on')
        obs['prefill_final']['deepstack_0']['sha256'] = 'f' * 64
        obs['prefill_proposed']['deepstack_0']['sha256'] = 'f' * 64
        with self.assertRaises(AssertionError):
            check_observation(item, prediction, obs, 'eadp1024_ds_on', 99, 100)

    def test_pruner_input_and_output_must_preserve_visual_features(self):
        item, prediction, obs = fixture('eadp1024_ds_on')
        obs['pruner_output_features']['sha256'] = 'f' * 64
        with self.assertRaises(AssertionError):
            check_observation(item, prediction, obs, 'eadp1024_ds_on', 99, 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
