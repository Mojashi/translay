"""Engine-level speedups for PFN's PLaMo 3 MLX port.

The shipped modeling_plamo_mlx.py favours bit-compatibility with the HF reference over speed:
hand-written RMSNorm and RoPE (a dozen small kernels each, 144 norms + 48 ropes per token) and
FP16 quantisation scales that force every BF16 linear through an FP32 promotion. At decode time a
2B model is launch-bound, so those kernels, not the weights, dominate.

`accelerate(model)` swaps those pieces for fused MLX kernels in place. It changes rounding, so
every caller must measure translation agreement against the reference path, not assume it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.tokenizer_utils import TokenizerWrapper


def _fast_rms_norm(self, hidden_states: mx.array) -> mx.array:
    return mx.fast.rms_norm(hidden_states, self.scale, self.variance_epsilon)


def _fast_rope_init(rope) -> None:
    # mx.fast.rope takes wavelengths (base ** (2i/d)), the reciprocal of HF's inv_freq.
    rope._wavelengths = (1.0 / rope._inv_freq).astype(mx.float32)


def _fast_rope(self, x: mx.array, offset=0) -> mx.array:
    if self._concentration != 1.0:
        x = x * self._concentration
    return mx.fast.rope(
        x, x.shape[-1], traditional=False, base=None, scale=1.0, offset=offset, freqs=self._wavelengths
    )


def _fast_mlp(self, x: mx.array) -> mx.array:
    gate, value = mx.split(self.gate_up_proj(x), 2, axis=-1)
    return self.down_proj(nn.silu(gate) * value)


def _cast_quant_params(model: nn.Module, dtype) -> int:
    """Store quantisation scales/biases in the activation dtype so matmuls stay in BF16."""
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            for name in ("scales", "biases"):
                arr = getattr(m, name, None)
                if arr is not None and arr.dtype != dtype:
                    setattr(m, name, arr.astype(dtype))
                    n += 1
    return n


def accelerate(model: Any, *, norms=True, rope=True, mlp=True, quant_dtype=True) -> dict:
    # The modeling file is loaded as an anonymous module, so reach its classes through instances.
    layers: list[Any] = list(model.layers)
    layer0 = layers[0]
    rms_cls, rope_cls, mlp_cls = type(layer0.pre_mixer_norm), type(layer0.mixer.rope), type(layer0.mlp)
    applied = {}
    if norms:
        rms_cls.__call__ = _fast_rms_norm
        applied["norms"] = True
    if rope:
        for layer in layers:
            _fast_rope_init(layer.mixer.rope)
        rope_cls.__call__ = _fast_rope
        applied["rope"] = True
    if mlp:
        mlp_cls.__call__ = _fast_mlp
        applied["mlp"] = True
    if quant_dtype:
        applied["quant_params_cast"] = _cast_quant_params(model, mx.bfloat16)
    mx.eval(model.parameters())
    return applied


@dataclass
class Segment:
    """One independent sequence inside a packed forward: its new tokens, its own KV caches, and
    from which of its positions predictions are wanted."""
    tokens: list[int]
    caches: list[Any]
    logits_from: int


def packed_forward(model: Any, segments: list[Segment]) -> list[list[int]]:
    """Run several independent sequences (e.g. one per target language) through ONE forward pass.

    Everything position-wise — embeddings, norms, QKV/O projections, MLP, lm_head — runs once on
    the concatenation of all segments, so the weights are read from memory once instead of once
    per language (decode is bandwidth-bound). Only attention is split per segment, each against its
    own cache and RoPE offset. No padding, and each segment's cache stays separately rewindable.

    Returns the greedy prediction for every position >= logits_from of each segment.
    """
    inner = model.model
    decoder = inner.layers
    lengths = [len(s.tokens) for s in segments]
    bounds, start = [], 0
    for n in lengths:
        bounds.append((start, start + n))
        start += n

    h = inner.embed_tokens(mx.array([t for s in segments for t in s.tokens])[None])
    if inner.config.scale_embedding:
        h = h * inner.config.hidden_size ** 0.5

    # Masks depend on each cache's fill level *before* this step, as in Plamo3Decoder.
    masks = []
    for s, (a, b) in zip(segments, bounds):
        seg_h = h[:, a:b]
        full = create_attention_mask(seg_h, s.caches[decoder.full_idx])
        sliding = (create_attention_mask(seg_h, s.caches[decoder.swa_idx], window_size=decoder.window_size)
                   if decoder.swa_idx is not None else None)
        masks.append((full, sliding))

    for li, layer in enumerate(decoder.layers):
        attn = layer.mixer
        x = layer.pre_mixer_norm(h)
        qkv = attn.qkv_proj(x).astype(x.dtype)
        outs = []
        for si, (a, b) in enumerate(bounds):
            n = b - a
            q, k, v = mx.split(qkv[:, a:b], [attn.q_proj_dim, attn.q_proj_dim + attn.k_proj_dim], axis=-1)
            q = attn.q_norm(q.reshape(1, n, attn.q_num_heads, attn.qk_dim).transpose(0, 2, 1, 3))
            k = attn.k_norm(k.reshape(1, n, attn.k_num_heads, attn.qk_dim).transpose(0, 2, 1, 3))
            v = v.reshape(1, n, attn.v_num_heads, attn.v_dim).transpose(0, 2, 1, 3)
            cache = segments[si].caches[li]
            q = attn.rope(q, offset=cache.offset)
            k = attn.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
            mask = masks[si][0] if layer.full_attn else masks[si][1]
            o = scaled_dot_product_attention(q, k, v, cache=cache, scale=attn.scale, mask=mask)
            outs.append(o.transpose(0, 2, 1, 3).reshape(1, n, attn.q_num_heads * attn.v_dim))
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
        h = h + layer.post_mixer_norm(attn.o_proj(o).astype(x.dtype))
        h = h + layer.post_mlp_norm(layer.mlp(layer.pre_mlp_norm(h)))
    h = inner.norm(h)

    wanted = [h[:, a + s.logits_from:b] for s, (a, b) in zip(segments, bounds)]
    picked = wanted[0] if len(wanted) == 1 else mx.concatenate(wanted, axis=1)
    preds = np.array(mx.argmax(model.lm_head(picked), axis=-1)[0]).tolist()
    result, i = [], 0
    for s, n in zip(segments, lengths):
        k = n - s.logits_from
        result.append([int(p) for p in preds[i:i + k]])
        i += k
    return result


def load_plamo(path: str, *, fast: bool = True) -> tuple[Any, TokenizerWrapper]:
    """Load a PLaMo 3 Translate MLX checkpoint, optionally with the fused-kernel path."""
    loaded = load(path, trust_remote_code=True, tokenizer_config={"trust_remote_code": True})
    model, tokenizer = loaded[0], loaded[1]
    if fast:
        accelerate(model)
    return model, tokenizer
