"""Pure teacher-forced sequence and loss helpers, without loading a model.

The real model, labels, exact tokenizer convention, sources and resource
policy must be separately frozen and validated before any oracle run.
"""
import torch
import torch.nn.functional as F

def compact_prompt(prompt_ids, prompt_positions, visual_start, full_visual_count, spatial_ids, rope_delta):
    """Gather P3 positions and text for one aligned visual-token subset.

    The caller must independently verify the original contiguous image-token
    segment and apply the same spatial IDs to every actual feature stream.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError('batch-one original prompt required')
    length = prompt_ids.shape[1]
    if (tuple(prompt_positions.shape) != (3, 1, length) or type(rope_delta) is not int
            or type(visual_start) is not int or type(full_visual_count) is not int
            or full_visual_count < 1 or visual_start < 0 or visual_start + full_visual_count > length
            or not spatial_ids or len(set(spatial_ids)) != len(spatial_ids)
            or any(type(i) is not int or i < 0 or i >= full_visual_count for i in spatial_ids)
            or list(spatial_ids) != sorted(spatial_ids)):
        raise ValueError('invalid original positions or spatial selection')
    kept = list(range(visual_start)) + [visual_start+i for i in spatial_ids] + list(range(visual_start+full_visual_count, length))
    index = torch.tensor(kept, device=prompt_ids.device, dtype=torch.long)
    return (prompt_ids.index_select(1, index), prompt_positions.index_select(-1, index.to(prompt_positions.device)),
            rope_delta + full_visual_count - len(spatial_ids), kept)

def teacher_inputs(prompt_embeds, visual, visual_start, answer_embeds, prompt_positions, rope_delta):
    """Construct no-cache teacher inputs and P3 positions for answer prefixes.

    answer_embeds contains the first n-1 reference-token embeddings. The last
    prompt hidden state predicts target token 0; subsequent prefix positions
    predict the following targets. Prompt is batch 1 without padding.
    """
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1 or visual.ndim != 2:
        raise ValueError('batch-one prompt and visual matrix required')
    length, hidden = prompt_embeds.shape[1:]
    count = visual.shape[0]
    if (visual.shape[1] != hidden or visual_start < 0 or visual_start + count > length
            or answer_embeds.ndim != 3 or answer_embeds.shape[0] != 1 or answer_embeds.shape[2] != hidden
            or tuple(prompt_positions.shape) != (3, 1, length) or type(rope_delta) is not int):
        raise ValueError('incompatible teacher geometry')
    x = torch.cat((prompt_embeds[:, :visual_start], visual.unsqueeze(0),
                   prompt_embeds[:, visual_start + count:], answer_embeds), dim=1)
    answer_positions = torch.arange(length + rope_delta, length + rope_delta + answer_embeds.shape[1],
                                    device=prompt_positions.device, dtype=prompt_positions.dtype)
    positions = torch.cat((prompt_positions, answer_positions.view(1, 1, -1).expand(3, 1, -1)), dim=-1)
    mask = torch.zeros((1, x.shape[1]), device=x.device, dtype=torch.bool)
    mask[:, visual_start:visual_start+count] = True
    return x, positions, mask

def mean_reference_nll(hidden, lm_head, prompt_length, target_ids):
    if (hidden.ndim != 3 or hidden.shape[0] != 1 or target_ids.ndim != 2 or target_ids.shape[0] != 1
            or not 1 <= target_ids.shape[1] <= 128 or prompt_length < 1
            or hidden.shape[1] != prompt_length + target_ids.shape[1] - 1):
        raise ValueError('invalid teacher target length')
    logits = lm_head(hidden[:, prompt_length-1:]).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction='mean')
