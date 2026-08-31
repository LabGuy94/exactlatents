"""Router stage 0 (oracle): verbatim anchor selection + anchored context building.

Anchors = the highest-rarity non-prose tokens of a function. They ride along
VERBATIM: each pooled group's vector is followed by its anchor tokens' raw
decoder embeddings, in sequence order. The decoder can then COPY hard tokens
(the copy ceiling proved copying is nearly free) instead of recalling them
from the pooled soup — targeting the commitment-failure class directly.

The training and evaluation paths share this implementation; the trainer's
batched path applies the same construction in vectorized form.
"""

import re

import torch

TRIPLE = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')
COMMENT = re.compile(r"#[^\n]*")


def prose_char_spans(code: str):
    return [(m.start(), m.end()) for rx in (TRIPLE, COMMENT) for m in rx.finditer(code)]


def anchor_positions(ids, offsets, code: str, rarity, frac: float):
    """Indices (sorted) of the top `frac` rarest non-prose tokens."""
    n = len(ids)
    k = int(n * frac)
    if k == 0:
        return []
    spans = prose_char_spans(code)
    scores = []
    for j, (tid, (a, b)) in enumerate(zip(ids, offsets)):
        if a == b:
            continue
        if any(a < e and b > s for s, e in spans):
            continue
        scores.append((float(rarity[tid]), j))
    scores.sort(reverse=True)
    return sorted(j for _, j in scores[:k])


def build_anchored_context(comp, tok, embed_table, code: str, device, frac: float, rarity):
    """[vec_g, anchors-of-group-g, vec_g+1, ...] -> (1, S, H) decoder embeds."""
    enc = tok(code, return_tensors="pt", add_special_tokens=False,
              return_offsets_mapping=True)
    ids = enc.input_ids.to(device)
    att = enc.attention_mask.to(device)
    vecs = comp(ids, att)                        # (1, G, H)
    if frac == 0:
        return vecs
    anchors = anchor_positions(ids[0].tolist(), enc.offset_mapping[0].tolist(),
                               code, rarity, frac)
    anchor_set = set(anchors)
    emb = embed_table(ids)                       # (1, T, H_dec)
    pf = comp.pooling_factor
    T = ids.shape[1]
    parts = []
    for g in range(vecs.shape[1]):
        parts.append(vecs[0, g : g + 1])
        for t in range(g * pf, min((g + 1) * pf, T)):
            if t in anchor_set:
                parts.append(emb[0, t : t + 1].to(vecs.dtype))
    return torch.cat(parts).unsqueeze(0)
