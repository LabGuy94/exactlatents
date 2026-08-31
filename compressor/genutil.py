"""Batched greedy generation over ragged embedding contexts.

Batching several sequences per generation call amortizes Python-side decoding
overhead and keeps the accelerator occupied.

Sequences are left-padded with zero embeddings and a zero attention mask so
each context ends immediately before its first generated token. Transformers
derives position IDs from the attention mask, making padding invisible to the
model.
"""

import torch


@torch.no_grad()
def batched_generate(decoder, vec_list, max_new, batch_size=8, num_beams=1):
    """vec_list: list of (L_i, H) context-embedding tensors. max_new: int or
    per-item list. Returns a list of 1-D generated-token-id tensors in
    vec_list order (finished sequences carry trailing pad/eos ids — decode
    with skip_special_tokens=True). Batches are grouped by context length to
    bound padding waste."""
    if isinstance(max_new, int):
        max_new = [max_new] * len(vec_list)
    outs = [None] * len(vec_list)
    idx = sorted(range(len(vec_list)), key=lambda i: vec_list[i].shape[0])
    for s in range(0, len(idx), batch_size):
        grp = idx[s : s + batch_size]
        L = max(vec_list[i].shape[0] for i in grp)
        ref = vec_list[grp[0]]
        embeds = torch.zeros(len(grp), L, ref.shape[-1], dtype=ref.dtype, device=ref.device)
        mask = torch.zeros(len(grp), L, dtype=torch.long, device=ref.device)
        for b, i in enumerate(grp):
            v = vec_list[i]
            embeds[b, L - v.shape[0]:] = v
            mask[b, L - v.shape[0]:] = 1
        out = decoder.generate(
            inputs_embeds=embeds, attention_mask=mask,
            max_new_tokens=max(max_new[i] for i in grp),
            do_sample=False, num_beams=num_beams,
            temperature=None, top_p=None, top_k=None,
        )
        for b, i in enumerate(grp):
            outs[i] = out[b]
    return outs
