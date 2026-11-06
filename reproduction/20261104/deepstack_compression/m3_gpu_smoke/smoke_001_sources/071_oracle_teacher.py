"""Model-agnostic single teacher pass; never loads a model or reference answer.

Caller owns verified inputs, frozen model, device, resource supervision and
target convention. Prompt positions and IDs must already match the supplied
visual stream (use compact_prompt before a pruned proposal); this module does
not infer or validate the original image-token segment. It alone is not an
approved real-model runner.
"""
import torch
from teacher_contract import teacher_inputs, mean_reference_nll

def _validate(model, prompt_ids, main_visual, ds_streams, target_ids, expected_visual_count):
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('teacher model must be eval and fully frozen')
    if (prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or target_ids.ndim != 2 or target_ids.shape[0] != 1
            or not 1 <= target_ids.shape[1] <= 128 or len(ds_streams) != 3
            or main_visual.ndim != 2 or main_visual.shape[0] != expected_visual_count):
        raise ValueError('invalid batch, target or visual count')
    if any(x.shape != main_visual.shape or x.dtype != main_visual.dtype or x.device != main_visual.device for x in ds_streams):
        raise ValueError('DeepStack feature geometry mismatch')

def _forward(model, prompt_ids, image_start, streams, prompt_positions, rope_delta, target_ids):
    embedding = model.get_input_embeddings()
    with torch.no_grad():
        prompt_embeds = embedding(prompt_ids).detach()
        answer_prefix = embedding(target_ids[:, :-1]).detach()
    embeds, positions, visual_mask = teacher_inputs(prompt_embeds, streams[0], image_start, answer_prefix,
                                                   prompt_positions, rope_delta)
    out = model.model.language_model(inputs_embeds=embeds, position_ids=positions,
                                     attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device),
                                     visual_pos_masks=visual_mask, deepstack_visual_embeds=streams[1:], use_cache=False)
    return mean_reference_nll(out.last_hidden_state, model.lm_head, prompt_ids.shape[1], target_ids)

def loss_and_gradients(model, prompt_ids, image_start, main_visual, ds_streams,
                       prompt_positions, rope_delta, target_ids, *, expected_visual_count=1024):
    _validate(model, prompt_ids, main_visual, ds_streams, target_ids, expected_visual_count)
    leaves = [x.detach().clone().requires_grad_(True) for x in (main_visual, *ds_streams)]
    with torch.enable_grad():
        loss = _forward(model, prompt_ids, image_start, leaves, prompt_positions, rope_delta, target_ids)
        grads = torch.autograd.grad(loss, leaves)
    if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in grads):
        raise ValueError('nonfinite teacher loss or feature gradient')
    return loss.detach(), [x.detach() for x in leaves], [g.detach() for g in grads]

def loss_only(model, prompt_ids, image_start, main_visual, ds_streams,
              prompt_positions, rope_delta, target_ids, *, expected_visual_count=256):
    """Score one proposal without a backward pass; never opens ground truth."""
    _validate(model, prompt_ids, main_visual, ds_streams, target_ids, expected_visual_count)
    with torch.no_grad():
        loss = _forward(model, prompt_ids, image_start, [main_visual, *ds_streams], prompt_positions, rope_delta, target_ids)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite teacher loss')
    return loss.detach()
