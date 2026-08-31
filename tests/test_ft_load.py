"""Checkpoint-loader coverage for PyTorch and safetensors weights.

The tests exercise plain PyTorch exports, consolidated PyTorch checkpoints,
the released flat-key safetensors layout, and strict failure paths.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compressor import ft_load


def _mlp(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))


class _Comp(nn.Module):  # stand-in with a .encoder like Compressor
    def __init__(self, seed):
        super().__init__()
        self.encoder = _mlp(seed)


def _same(a, b):
    return all(torch.equal(x, y) for (_, x), (_, y) in
               zip(a.state_dict().items(), b.state_dict().items()))


def _state_same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(autouse=True)
def _fresh_cache():
    ft_load._CACHE.clear()
    ft_load._SAFE_METADATA_CACHE.clear()
    yield
    ft_load._CACHE.clear()
    ft_load._SAFE_METADATA_CACHE.clear()


def _full_state(tmp_path, enc, dec, proj, ema):
    p = tmp_path / "full_state_latest.pt"
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                "projector": proj.state_dict(),
                "ema": (ema.state_dict() if ema is not None else None),
                "opt": {"state": {}}, "sched": {}, "py_rng": None,
                "step": 10000, "args_hash": "x"}, p)
    return p


def _safe_state(tmp_path, enc, dec, proj, ema):
    metadata = {"format": "exactlatents.full_state.v1"}
    model_path = tmp_path / "model.safetensors"
    flat_model = {
        f"{part}.{key}": tensor.detach().clone()
        for part, module in (("enc", enc), ("dec", dec), ("projector", proj))
        for key, tensor in module.state_dict().items()
    }
    save_file(flat_model, model_path, metadata=metadata)

    ema_path = tmp_path / "projector_ema.safetensors"
    flat_ema = {
        f"ema.{key}": tensor.detach().clone()
        for key, tensor in ema.state_dict().items()
    }
    save_file(flat_ema, ema_path, metadata=metadata)
    return model_path, ema_path


def test_plain_export_roundtrip(tmp_path):
    src, dst = _Comp(0), _Comp(1)
    p = tmp_path / "encoder_final_bf16.pt"
    torch.save({k: v.to(torch.bfloat16) for k, v in src.encoder.state_dict().items()}, p)
    meta = ft_load.load_encoder(dst, p)
    assert meta["shape"] == "plain" and meta["step"] is None
    for v, w in zip(dst.encoder.state_dict().values(), src.encoder.state_dict().values()):
        assert torch.equal(v, w.to(torch.bfloat16).to(v.dtype))


def test_full_state_all_parts(tmp_path):
    enc, dec, proj, ema = _mlp(0), _mlp(1), _mlp(2), _mlp(3)
    p = _full_state(tmp_path, enc, dec, proj, ema)
    comp2, dec2, proj2 = _Comp(9), _mlp(8), _mlp(7)
    m1 = ft_load.load_encoder(comp2, p)
    m2 = ft_load.load_decoder(dec2, p)
    m3 = ft_load.load_projector(proj2, p, prefer="ema")
    assert (m1["shape"], m1["step"]) == ("full_state", 10000)
    assert _same(comp2.encoder, enc) and _same(dec2, dec)
    assert _same(proj2, ema) and not _same(proj2, proj)  # EMA, not live
    m4 = ft_load.load_projector(proj2, p, prefer="live")
    assert _same(proj2, proj) and m4["part"] == "projector"
    # one torch.load for all four extractions
    assert len(ft_load._CACHE) == 1


def test_safetensors_all_parts(tmp_path):
    enc, dec, proj, ema = _mlp(0), _mlp(1), _mlp(2), _mlp(3)
    model_path, ema_path = _safe_state(tmp_path, enc, dec, proj, ema)
    assert set(ft_load.load_file(model_path)) == {"enc", "dec", "projector"}
    assert set(ft_load.load_file(ema_path)) == {"ema"}

    for path, part, expected in (
        (model_path, "enc", enc.state_dict()),
        (model_path, "dec", dec.state_dict()),
        (model_path, "projector", proj.state_dict()),
        (ema_path, "ema", ema.state_dict()),
    ):
        state, meta = ft_load.extract(path, part)
        assert _state_same(state, expected)
        assert meta["format"] == "exactlatents.full_state.v1"

    comp2, dec2, proj2, ema2 = _Comp(9), _mlp(8), _mlp(7), _mlp(6)
    ft_load.load_encoder(comp2, model_path)
    ft_load.load_decoder(dec2, model_path)
    ft_load.load_projector(proj2, model_path, prefer="live")
    ft_load.load_projector(ema2, ema_path)
    assert _same(comp2.encoder, enc) and _same(dec2, dec)
    assert _same(proj2, proj) and _same(ema2, ema)
    assert len(ft_load._CACHE) == 2


def test_fp32_masters_downcast_into_bf16_module(tmp_path):
    src = _mlp(0)
    p = tmp_path / "full_state_latest.pt"
    torch.save({"enc": src.state_dict(), "dec": src.state_dict(),
                "projector": src.state_dict(), "ema": None, "step": 5}, p)
    dst = _Comp(1)
    dst.encoder.to(torch.bfloat16)
    ft_load.load_encoder(dst, p)
    for v, w in zip(dst.encoder.state_dict().values(), src.state_dict().values()):
        assert v.dtype == torch.bfloat16 and torch.equal(v, w.to(torch.bfloat16))


def test_ema_none_fails_loud(tmp_path):
    m = _mlp(0)
    p = tmp_path / "full_state_latest.pt"
    torch.save({"enc": m.state_dict(), "dec": m.state_dict(),
                "projector": m.state_dict(), "ema": None, "step": 5}, p)
    with pytest.raises(ValueError, match="ema=None"):
        ft_load.load_projector(_mlp(1), p, prefer="ema")


def test_empty_state_fails(tmp_path):
    p = tmp_path / "empty.pt"
    torch.save({}, p)
    with pytest.raises(AssertionError, match="empty"):
        ft_load.load_decoder(_mlp(0), p)


def test_strict_key_mismatch_fails(tmp_path):
    p = tmp_path / "decoder_final_bf16.pt"
    torch.save(_mlp(0).state_dict(), p)
    wrong = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(RuntimeError):
        ft_load.load_decoder(wrong, p)


def test_missing_weights_error_has_download_hint(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match=r"hf download labguy/exactlatents-qwen3-1\.7b --local-dir weights/",
    ):
        ft_load.load_file(tmp_path / "missing.safetensors")
