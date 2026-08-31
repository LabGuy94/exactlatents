"""Load full-finetuning checkpoints into the compressor and decoder.

PyTorch checkpoints may be either a plain state dict for one component or a
consolidated mapping with ``enc``, ``dec``, ``projector``, and optional ``ema``
state dicts. Safetensors checkpoints use flat ``part.key`` names: the released
model file contains the ``enc``, ``dec``, and ``projector`` parts, while the
projector EMA file contains the ``ema`` part.

Loads are strict: a key mismatch indicates an incompatible architecture and
must fail rather than leave a module partially initialized. Loaded files are
cached because one model file commonly supplies several components.
"""

import os

import torch

# A consolidated trainer save always carries all three live weight parts.
_FULL_STATE_MARKER = {"enc", "dec", "projector"}
_SAFE_PARTS = _FULL_STATE_MARKER | {"ema"}
_SAFE_METADATA_CACHE: dict[str, dict[str, str]] = {}
_WEIGHTS_HINT = (
    "hf download labguy/exactlatents-qwen3-1.7b --local-dir weights/"
)

_CACHE: dict[str, object] = {}


def load_file(path):
    """Load a checkpoint once on CPU and cache its reconstructed object."""
    key = os.path.realpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[ft_load] checkpoint not found: {path}. Download released weights "
            f"with: {_WEIGHTS_HINT}"
        )
    if key not in _CACHE:
        if str(path).endswith(".safetensors"):
            from safetensors import safe_open
            from safetensors.torch import load_file as load_safetensors_file

            flat = load_safetensors_file(str(path), device="cpu")
            grouped = {}
            for flat_key, tensor in flat.items():
                part, separator, state_key = flat_key.partition(".")
                if not separator or part not in _SAFE_PARTS or not state_key:
                    raise ValueError(
                        f"[ft_load] {path}: invalid safetensors key {flat_key!r}; "
                        "expected 'enc.*', 'dec.*', 'projector.*', or 'ema.*'"
                    )
                grouped.setdefault(part, {})[state_key] = tensor
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                _SAFE_METADATA_CACHE[key] = dict(handle.metadata() or {})
            _CACHE[key] = grouped
        else:
            _CACHE[key] = torch.load(
                path, map_location="cpu", weights_only=False
            )
    return _CACHE[key]


def is_full_state(obj) -> bool:
    return isinstance(obj, dict) and _FULL_STATE_MARKER <= set(obj.keys())


def _check_sd(sd, part, path):
    assert isinstance(sd, dict) and sd, \
        f"[ft_load] {part} from {path}: empty/non-dict state — refusing to eval"
    bad = [k for k, v in sd.items() if not torch.is_tensor(v)]
    assert not bad, f"[ft_load] {part} from {path}: non-tensor entries {bad[:3]}"


def extract(path, part):
    """Return ``(state_dict, metadata)`` for one checkpoint component.

    ``part`` is one of ``enc``, ``dec``, ``projector``, or ``ema``. A
    consolidated PyTorch checkpoint and the released model safetensors file
    are indexed by component. Other PyTorch files remain plain state dicts.
    """
    key = os.path.realpath(path)
    obj = load_file(path)
    is_safe = str(path).endswith(".safetensors")
    if is_full_state(obj):
        if part == "ema" and obj.get("ema") is None:
            raise ValueError(
                f"[ft_load] {path}: consolidated state has ema=None; "
                "pass live projector weights explicitly instead"
            )
        sd = obj["ema"] if part == "ema" else obj[part]
        if is_safe:
            meta = dict(_SAFE_METADATA_CACHE[key])
            meta.update(
                path=str(path),
                shape="full_state",
                part=part,
                step=meta.get("step"),
            )
        else:
            meta = {
                "path": str(path),
                "shape": "full_state",
                "part": part,
                "step": obj.get("step"),
            }
    elif is_safe:
        if part not in obj:
            raise KeyError(f"[ft_load] {path}: safetensors file has no {part!r} part")
        sd = obj[part]
        meta = dict(_SAFE_METADATA_CACHE[key])
        meta.update(
            path=str(path), shape="plain", part=part, step=meta.get("step")
        )
    else:
        sd = obj
        meta = {
            "path": str(path),
            "shape": "plain",
            "part": part,
            "step": None,
        }
    _check_sd(sd, part, path)
    return sd, meta


def _load_into(module, sd, meta, label):
    n_par = sum(v.numel() for v in sd.values())
    src_dt = sorted({str(v.dtype).replace("torch.", "") for v in sd.values()})
    module.load_state_dict(sd)  # strict — wrong-architecture evals must die here
    step = f", step {meta['step']}" if meta["step"] is not None else ""
    print(f"[ft_load] {label} <- {meta['path']} ({meta['shape']}{step}, "
          f"{len(sd)} tensors, {n_par/1e9:.2f}B params, src {src_dt})")
    meta.update(n_tensors=len(sd), n_params=n_par, src_dtypes=src_dt)
    return meta


def load_encoder(comp, path):
    """Full-FT encoder weights into comp.encoder (the truncated trunk)."""
    return _load_into(comp.encoder, *extract(path, "enc"), label="encoder")


def load_decoder(model, path):
    """Full-FT decoder weights into an AutoModelForCausalLM."""
    return _load_into(model, *extract(path, "dec"), label="decoder")


def load_projector(projector, path, prefer="ema"):
    """Load projector weights, including any pooling or boundary parameters.

    For consolidated checkpoints, ``prefer`` selects EMA or live projector
    weights. A plain checkpoint already identifies the intended state by its
    path. The released EMA safetensors file contains one explicit ``ema`` part.
    """
    assert prefer in ("ema", "live"), prefer
    part = "ema" if prefer == "ema" else "projector"
    obj = load_file(path)
    if not is_full_state(obj):
        if (
            str(path).endswith(".safetensors")
            and isinstance(obj, dict)
            and "ema" in obj
            and "projector" not in obj
        ):
            part = "ema"
        else:
            part = "projector"
    return _load_into(
        projector, *extract(path, part), label=f"projector[{part}]"
    )
