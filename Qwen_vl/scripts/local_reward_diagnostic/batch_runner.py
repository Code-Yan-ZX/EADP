"""Batch the frozen native decoder for equal-budget masks of one page.

Only execution batching changes. The frozen runner still prepares every row's
exact visual subset, original mRoPE positions and original text embeddings.
Finished rows feed EOS until the last row finishes, while their output and
decode-position histories stop at their own first EOS.
"""
from __future__ import annotations

import time

import torch


def _keep_array(value):
    return value.get("keep_idx") if isinstance(value, dict) else value


@torch.inference_mode()
def prefill_many(runner, page, listkeep):
    """Return logits [B,vocab], a batched native cache and per-row audits."""
    if not listkeep:
        raise ValueError("at least one mask is required")
    rows = [runner._compact(page, _keep_array(keep)) for keep in listkeep]
    lengths = {row[0].shape[1] for row in rows}
    budgets = {row[4]["n_vis"] for row in rows}
    starts = {row[4]["expected_decode_start"] for row in rows}
    if len(lengths) != 1 or len(budgets) != 1 or len(starts) != 1:
        raise ValueError("batching requires one page, one prompt length and the same K")
    embeddings = torch.cat([row[0] for row in rows], dim=0)
    positions = torch.cat([row[1] for row in rows], dim=1)
    visual_mask = torch.cat([row[2] for row in rows], dim=0)
    if any(len(row[3]) != 3 for row in rows):
        raise ValueError("three native DeepStack streams are required")
    ds = [torch.cat([row[3][stream] for row in rows], dim=0) for stream in range(3)]
    batch = len(rows)
    k = next(iter(budgets))
    if visual_mask.shape != embeddings.shape[:2] or positions.shape != (3, batch, embeddings.shape[1]):
        raise ValueError("batched mask/position shapes differ from native embeddings")
    if not bool((visual_mask.sum(-1) == k).all()) or any(value.shape[0] != batch * k for value in ds):
        raise ValueError("batched visual mask and flattened DeepStack ordering disagree")
    output = runner.text(
        inputs_embeds=embeddings, position_ids=positions,
        attention_mask=torch.ones(embeddings.shape[:2], dtype=torch.long, device=runner.device),
        visual_pos_masks=visual_mask, deepstack_visual_embeds=ds,
        use_cache=True, return_dict=True)
    logits = runner.model.lm_head(output.last_hidden_state[:, -1:])[:, 0]
    audits = [dict(row[4]) for row in rows]
    return logits, output.past_key_values, audits


@torch.inference_mode()
def generate_many(runner, page, listkeep, max_new_tokens=2048):
    """Return [(text, audit), ...] for same-page, same-K masks, in input order."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    started = time.monotonic()
    logits, cache, audits = prefill_many(runner, page, listkeep)
    batch = len(audits)
    prompt_length = audits[0]["prompt_tokens"]
    eos = runner.model.generation_config.eos_token_id
    eos = {int(eos)} if isinstance(eos, int) else set(eos or [])
    filler = min(eos) if eos else 0
    tokens = [[] for _ in audits]
    positions_history = [[] for _ in audits]
    finished = [False] * batch
    stop_reasons = ["max_new_tokens"] * batch
    for step in range(max_new_tokens):
        next_tokens = logits.argmax(-1).tolist()
        for row, token in enumerate(next_tokens):
            if finished[row]:
                continue
            tokens[row].append(int(token))
            if token in eos:
                finished[row] = True
                stop_reasons[row] = "eos"
        if all(finished) or step + 1 == max_new_tokens:
            break
        feed = [filler if finished[row] else token for row, token in enumerate(next_tokens)]
        decode_positions = [audit["expected_decode_start"] + step for audit in audits]
        for row, position in enumerate(decode_positions):
            if not finished[row]:
                positions_history[row].append(position)
        position_ids = torch.tensor(decode_positions, dtype=torch.long, device=runner.device)
        position_ids = position_ids.view(1, batch, 1).expand(3, -1, -1)
        output = runner.text(
            input_ids=torch.tensor(feed, dtype=torch.long, device=runner.device).view(batch, 1),
            position_ids=position_ids, past_key_values=cache,
            attention_mask=torch.ones((batch, prompt_length + step + 1), dtype=torch.long, device=runner.device),
            use_cache=True, return_dict=True)
        cache = output.past_key_values
        logits = runner.model.lm_head(output.last_hidden_state[:, -1:])[:, 0]
    elapsed = time.monotonic() - started
    results = []
    for row, audit in enumerate(audits):
        text = runner.processor.tokenizer.decode(tokens[row], skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        audit.update(token_ids=tokens[row], generated_tokens=len(tokens[row]),
                     truncated=stop_reasons[row] == "max_new_tokens", stop_reason=stop_reasons[row],
                     decode_positions=positions_history[row], execution_mode="same-page same-K batched native decoder",
                     batch_size=batch, batch_row=row, elapsed_seconds=elapsed,
                     elapsed_scope="shared batch wall time; do not sum per-row elapsed_seconds")
        results.append((text, audit))
    return results


def _first_difference(before, after):
    for index, (left, right) in enumerate(zip(before, after)):
        if left != right:
            return dict(index=index, serial=left, batch=right)
    if len(before) != len(after):
        index = min(len(before), len(after))
        return dict(index=index, serial=before[index] if index < len(before) else None,
                    batch=after[index] if index < len(after) else None)
    return None


@torch.inference_mode()
def batch_gate(runner, page, listkeep, native_tokens=8, prefix_tokens=16):
    """Outside-panel gate: stock full generation, then serial/batch mask prefixes.

    BF16 batching may change arithmetic. Report first-logit deltas and every
    changed generated token prefix instead of requiring bitwise logit equality.
    The caller must record/decide on any token mismatch before panel execution.
    """
    native_inputs = {key: value.to(runner.device) for key, value in page["native_inputs"].items()}
    stock = runner.model.generate(**native_inputs, do_sample=False, use_cache=True,
                                  max_new_tokens=native_tokens)
    stock_tokens = stock[0, page["ids"].shape[1]:].tolist()
    _, full_serial = runner.generate(page, max_new_tokens=native_tokens)
    full_difference = _first_difference(stock_tokens, full_serial["token_ids"])
    batched_logits, _, audits = prefill_many(runner, page, listkeep)
    prefill_records = []
    for row, keep in enumerate(listkeep):
        serial_logits, _, _ = runner.prefill(page, _keep_array(keep))
        difference = (serial_logits[0].float() - batched_logits[row].float()).abs()
        prefill_records.append(dict(row=row, max_logit_delta=float(difference.max()),
                                    logits_bitwise=torch.equal(serial_logits[0], batched_logits[row]),
                                    argmax_equal=int(serial_logits.argmax(-1)) == int(batched_logits[row].argmax(-1))))
    batch_outputs = generate_many(runner, page, listkeep, prefix_tokens)
    comparisons = []
    for row, (keep, (text, batch_info)) in enumerate(zip(listkeep, batch_outputs)):
        serial_text, serial_info = runner.generate(page, _keep_array(keep), prefix_tokens)
        difference = _first_difference(serial_info["token_ids"], batch_info["token_ids"])
        comparisons.append(dict(row=row, same_tokens=difference is None,
                                first_difference=difference, same_text=serial_text == text,
                                serial_tokens=serial_info["token_ids"], batch_tokens=batch_info["token_ids"],
                                same_decode_positions=serial_info["decode_positions"] == batch_info["decode_positions"],
                                serial_stop_reason=serial_info["stop_reason"], batch_stop_reason=batch_info["stop_reason"]))
    return dict(scope="outside-panel page; only batching differs; no reward inspected",
                native_generation_tokens=native_tokens, stock_full_tokens=stock_tokens,
                serial_full_tokens=full_serial["token_ids"], full_native_tokens_equal=full_difference is None,
                full_native_first_difference=full_difference, compressed_prefix_tokens=prefix_tokens,
                batch_size=len(listkeep), n_vis=audits[0]["n_vis"],
                ds_lengths_per_row=[audit["ds_lengths"] for audit in audits],
                prefills=prefill_records, prefix_comparisons=comparisons,
                all_prefixes_equal=all(item["same_tokens"] for item in comparisons),
                all_decode_positions_equal=all(item["same_decode_positions"] for item in comparisons),
                pass_all=(full_difference is None and all(item["same_tokens"] and item["same_decode_positions"]
                                                         for item in comparisons)),
                interpretation="logit bitwise equality is descriptive; changed BF16 argmax/prefixes are explicit and must not be hidden")
