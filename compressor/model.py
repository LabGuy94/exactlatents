"""The compressor: encoder -> pooling -> projector.

Reads a chunk of code as text tokens, emits pooling_factor-times fewer
continuous vectors in the decoder's embedding space. The decoder is NOT
part of this module — it stays a frozen, untouched model that merely
receives these vectors via inputs_embeds.

Architecture follows the ARC-Encoder recipe (arXiv 2510.20535), our own
implementation on the HF stack:
  encoder   a small pretrained LM with its top layers dropped (top layers
            specialize toward next-token prediction; mid-layer states carry
            more general content)
  pooling   mask-aware mean over consecutive groups of `pooling_factor`
  projector 2-layer MLP into the decoder's hidden size
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_bidirectional_mask

MAX_PF = 8  # slot badges are allocated up to this pooling factor


class _Projector(nn.Sequential):
    """Sequential MLP that may also carry extra parameters/modules as
    attributes (never applied by this forward). Keeping them here means the
    optimizer, EMA, and every checkpoint path that handles `projector`
    automatically handles them too — no second state to forget.

    forward applies ONLY the numeric-keyed children (the MLP layers laid down
    by the constructor); attribute-assigned submodules like `latent_pooler`
    are registered children too, and plain Sequential would try to call them."""

    def forward(self, x):
        for name, m in self._modules.items():
            if name.isdigit():
                x = m(x)
        return x


class LatentPooler(nn.Module):
    """Latent-query allocation pooling. Each of the G = ceil(T/pf) output
    slots is a QUERY that cross-attends over ALL encoder token states and takes
    a learned weighted blend. Training decides the allocation instead of using
    the fixed ``slot g = mean(tokens[g*pf:(g+1)*pf])`` chopper. Uniform pooling
    gives roughly 55% of the budget to structure and prose, while
    identifiers/literals account for most of the information in fewer tokens.

    Initialization preserves phase-mean pooling exactly at step 0:
      * output = phase_mean + o_proj(attention), with o_proj zero-initialized
        (the LoRA B=0 trick) -> the attention branch contributes nothing until
        training grows it, so a warm-started projector/LoRA sees identical
        inputs at step 0;
      * queries come from each slot's own phase-mean -> locality by content;
      * an ALiBi-style distance penalty per head (learnable slope, geometric
        init) biases attention toward the slot's own region — steep heads stay
        local, shallow heads are near-global;
      * per-token importance logit = learned head (zero-init) + a fixed
        surprisal prior from the corpus rarity table (`rarity_tbl` buffer,
        data/rarity.pt) scaled by a learnable per-head gate — day one it
        already 'knows' colons are cheap and identifiers are expensive.

    fp32 throughout (it is trained, like the projector). ~17M params at D=2048.
    """

    def __init__(self, hidden: int, vocab_size: int, n_heads: int = 16):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads, self.d_head = n_heads, hidden // n_heads
        kw = {"dtype": torch.float32}
        self.q_norm = nn.LayerNorm(hidden, **kw)
        self.k_norm = nn.LayerNorm(hidden, **kw)
        self.q_proj = nn.Linear(hidden, hidden, bias=False, **kw)
        self.k_proj = nn.Linear(hidden, hidden, bias=False, **kw)
        self.v_proj = nn.Linear(hidden, hidden, bias=False, **kw)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, **kw)
        nn.init.zeros_(self.o_proj.weight)          # step-0 identity with phase-mean
        # locality: slope_h = exp(log_slope_h), penalty = -slope * |token - slot center|
        # geometric init a la ALiBi: 2^-0.5 (steep/local) ... 2^-8 (near-global)
        slopes = torch.tensor([2.0 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)])
        self.log_slope = nn.Parameter(slopes.log())
        # importance: learned content head (zero-init) + gated surprisal prior
        self.imp_head = nn.Linear(hidden, n_heads, bias=False, **kw)
        nn.init.zeros_(self.imp_head.weight)
        self.imp_gate = nn.Parameter(torch.full((n_heads,), 2.0))
        self.register_buffer("rarity_tbl", torch.zeros(vocab_size))  # filled by trainer / checkpoint

    def forward(self, h, pm, attention_mask, input_ids, pf):
        """h (B,T,D) raw encoder states; pm (B,G,D) phase-mean pooled;
        attention_mask (B,T); input_ids (B,T) — all T already pf-padded.
        Returns (B,G,D) = pm + learned allocation delta. fp32."""
        B, T, D = h.shape
        G = pm.shape[1]
        H, dh = self.n_heads, self.d_head

        q = self.q_proj(self.q_norm(pm)).view(B, G, H, dh).transpose(1, 2)   # (B,H,G,dh)
        k = self.k_proj(self.k_norm(h)).view(B, T, H, dh).transpose(1, 2)    # (B,H,T,dh)
        v = self.v_proj(h).view(B, T, H, dh).transpose(1, 2)                 # (B,H,T,dh)

        logits = q @ k.transpose(-1, -2) / (dh ** 0.5)                       # (B,H,G,T)
        centers = torch.arange(G, device=h.device, dtype=torch.float32) * pf + (pf - 1) / 2
        dist = (torch.arange(T, device=h.device, dtype=torch.float32)[None, :]
                - centers[:, None]).abs()                                    # (G,T)
        logits = logits - self.log_slope.exp().view(1, H, 1, 1) * dist       # locality
        imp = self.imp_head(h).permute(0, 2, 1)                              # (B,H,T)
        imp = imp + self.imp_gate.view(1, H, 1) * self.rarity_tbl[input_ids].unsqueeze(1)
        logits = logits + imp.unsqueeze(2)                                   # broadcast over G
        logits = logits.masked_fill(attention_mask.view(B, 1, 1, T) == 0, torch.finfo(logits.dtype).min)

        p = torch.softmax(logits, dim=-1)                                    # (B,H,G,T)
        if getattr(self, "keep_attn", False):
            # Expose the allocation distribution so the trainer can optionally
            # regularize per-span attention mass early in training.
            self.last_attn = p
        attn = p @ v                                                         # (B,H,G,dh)
        attn = attn.transpose(1, 2).reshape(B, G, D)
        return pm + self.o_proj(attn)


class Compressor(nn.Module):
    def __init__(
        self,
        encoder_name: str = "Qwen/Qwen3-1.7B",
        decoder_hidden: int = 2048,
        pooling_factor: int = 8,
        drop_top_layers: int = 2,
        freeze_encoder: bool = True,
        proj_width: int | None = None,
        proj_depth: int = 2,
        pooling: str = "mean",
        boundary: bool = False,
        bidirectional: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        hf_revision: str | None = None,
    ):
        super().__init__()
        self.pooling_factor = pooling_factor
        self.pooling = pooling
        self.boundary = boundary
        self.bidirectional = bidirectional
        # Pin the Hub revision when requested; None uses the default branch.
        _rev = {"revision": hf_revision} if hf_revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, **_rev)

        full = AutoModelForCausalLM.from_pretrained(encoder_name, dtype=dtype, **_rev)
        self.encoder = full.model  # transformer stack only, no LM head
        if drop_top_layers > 0:
            self.encoder.layers = nn.ModuleList(self.encoder.layers[:-drop_top_layers])
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        if bidirectional:
            # The encoder describes an existing function rather than generating
            # it, so a causal mask needlessly hides later identifiers. The mask
            # is built at runtime and replaced with create_bidirectional_mask.
            # Flipping is_causal is also required: SDPA's no-padding fast path
            # checks the module flag and would otherwise restore causality for
            # unpadded batches.
            lt = getattr(self.encoder.config, "layer_types", None)
            assert lt is None or all(t == "full_attention" for t in lt), \
                "bidirectional flip assumes full-attention layers only (no sliding)"
            for layer in self.encoder.layers:
                layer.self_attn.is_causal = False

        enc_hidden = self.encoder.config.hidden_size
        # fp32 on purpose: the projector is the only thing the optimizer touches,
        # and AdamW on bf16 master weights loses update precision. The frozen
        # giants stay bf16; we cast at the seam.
        w = proj_width or enc_hidden
        layers: list[nn.Module] = [nn.Linear(enc_hidden, w, dtype=torch.float32), nn.SiLU()]
        for _ in range(proj_depth - 2):
            layers += [nn.Linear(w, w, dtype=torch.float32), nn.SiLU()]
        layers += [nn.Linear(w, decoder_hidden, dtype=torch.float32)]
        self.projector = _Projector(*layers)
        if pooling == "attn":
            # slot badges: learned per-position tags added inside each group so
            # the pooled mixture keeps within-group ORDER (a mean is order-blind
            # — the observed `file_path_queue`->`file_queue_path` shuffles).
            self.projector.slot_pos = nn.Parameter(
                torch.randn(MAX_PF, enc_hidden, dtype=torch.float32) * 0.02)
            # importance probe: scores each token in the group; softmax weights
            # replace the flat average (rare fragments stop being diluted by
            # boilerplate neighbors).
            self.projector.pool_query = nn.Parameter(
                torch.randn(enc_hidden, dtype=torch.float32) * 0.02)
        if pooling == "latent":
            # Store the pooler on the projector so optimizer, EMA, and
            # checkpoint paths include it automatically. Loading a projector
            # without these parameters via strict=False preserves the
            # step-0-identity initialization.
            self.projector.latent_pooler = LatentPooler(enc_hidden, len(self.tokenizer))
        if boundary:
            # Learned <block>/</block> embeddings in decoder space mark where a
            # compressed block starts and ends regardless of context position.
            # Storing them on the projector makes optimizer, EMA, and checkpoint
            # handling automatic.
            self.projector.block_open = nn.Parameter(
                torch.randn(decoder_hidden, dtype=torch.float32) * 0.02)
            self.projector.block_close = nn.Parameter(
                torch.randn(decoder_hidden, dtype=torch.float32) * 0.02)
        self.out_dtype = dtype

    def boundary_pair(self):
        """(open, close) as (1, 1, H) tensors in out_dtype, or None."""
        if not self.boundary:
            return None
        return (self.projector.block_open.view(1, 1, -1).to(self.out_dtype),
                self.projector.block_close.view(1, 1, -1).to(self.out_dtype))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids -> (B, ceil(T/pf), decoder_hidden) vectors."""
        if self.bidirectional:
            # dict-wrapped mask bypasses HF's internal causal-mask construction
            # (a plain 2D ones-mask does NOT — the triangle gets added on top)
            embeds = self.encoder.embed_tokens(input_ids)
            bmask = create_bidirectional_mask(
                config=self.encoder.config, inputs_embeds=embeds,
                attention_mask=attention_mask)
            h = self.encoder(
                inputs_embeds=embeds, attention_mask={"full_attention": bmask},
                use_cache=False,
            ).last_hidden_state                               # (B, T, D)
        else:
            h = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state                               # (B, T, D)

        B, T, D = h.shape
        pf = self.pooling_factor
        pad = (-T) % pf
        if pad:
            h = nn.functional.pad(h, (0, 0, 0, pad))
            attention_mask = nn.functional.pad(attention_mask, (0, pad))
            input_ids = nn.functional.pad(input_ids, (0, pad))

        if self.pooling == "attn":
            hg = h.view(B, -1, pf, D).float()                      # (B, G, pf, D)
            hb = hg + self.projector.slot_pos[:pf]                 # badge each slot
            scores = hb @ self.projector.pool_query / (D ** 0.5)   # (B, G, pf)
            mask = attention_mask.view(B, -1, pf)
            scores = scores.masked_fill(mask == 0, -1e9)           # ignore padding
            wts = torch.softmax(scores, dim=-1).unsqueeze(-1)      # (B, G, pf, 1)
            pooled = (hb * wts).sum(dim=2)                         # (B, G, D)
            return self.projector(pooled).to(self.out_dtype)

        h_raw = h  # latent pooling attends over UNROTATED states
        if self.pooling in ("phase", "latent"):
            # RoPE-style within-group rotation BEFORE the mean: a mean is
            # order-blind (mean(A,B,C,D) == mean(D,C,B,A)); rotating state k by
            # angle k*theta_j per dim-pair stamps within-group ORDER onto the
            # average as phase. This parameter-free transform preserves
            # within-group order. Latent pooling keeps it as its residual base
            # for an identity initialization.
            k = (torch.arange(h.shape[1], device=h.device) % pf).float()   # (T+pad,)
            half = D // 2
            inv = 10000.0 ** (-torch.arange(half, device=h.device, dtype=torch.float32) / half)
            ang = k[:, None] * inv[None, :]                                # (T+pad, D/2)
            cos, sin = ang.cos().to(h.dtype), ang.sin().to(h.dtype)
            x, y = h[..., 0::2], h[..., 1::2]
            h = torch.stack([x * cos - y * sin, x * sin + y * cos], dim=-1).flatten(-2)

        # mean over each group of pf, counting only real (unmasked) tokens
        w = attention_mask.to(h.dtype).unsqueeze(-1)          # (B, T+pad, 1)
        summed = (h * w).view(B, -1, pf, D).sum(dim=2)        # (B, G, D)
        counts = w.view(B, -1, pf, 1).sum(dim=2).clamp(min=1) # (B, G, 1)
        pooled = summed / counts

        if self.pooling == "latent":
            pooled = self.projector.latent_pooler(
                h_raw.float(), pooled.float(), attention_mask, input_ids, pf)
            return self.projector(pooled).to(self.out_dtype)

        return self.projector(pooled.float()).to(self.out_dtype)

    def compress(self, text: str, device: str) -> torch.Tensor:
        """Convenience for single strings: text -> (1, G[+2], decoder_hidden).
        With boundary=True the block is wrapped in the learned <block>/</block>
        embeddings, so eval-time contexts match the training format."""
        enc = self.tokenizer(text, return_tensors="pt").to(device)
        vecs = self(enc.input_ids, enc.attention_mask)
        bp = self.boundary_pair()
        if bp is not None:
            vecs = torch.cat([bp[0], vecs, bp[1]], dim=1)
        return vecs
