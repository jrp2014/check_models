import mlx.core as mx
import mlx.nn as nn
from ..base import LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from ..cache import KVCache as KVCache, RotatingKVCache as RotatingKVCache
from .config import TextConfig as TextConfig
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass
from typing import Any

class RMSNorm(nn.Module):
    weight: Incomplete
    eps: Incomplete
    def __init__(self, dims: int, eps: float = 1e-05) -> None: ...
    def __call__(self, x): ...

class Attention(nn.Module):
    n_heads: Incomplete
    n_kv_heads: Incomplete
    repeats: Incomplete
    head_dim: Incomplete
    layer_idx: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    is_sliding: Incomplete
    rope: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None) -> mx.array: ...

class MLP(nn.Module):
    gate_proj: Incomplete
    down_proj: Incomplete
    up_proj: Incomplete
    def __init__(self, dim, hidden_dim) -> None: ...
    def __call__(self, x) -> mx.array: ...

def clip_residual(x, y=None): ...

class TransformerBlock(nn.Module):
    num_attention_heads: Incomplete
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    pre_feedforward_layernorm: Incomplete
    post_feedforward_layernorm: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None) -> mx.array: ...

class Gemma3Model(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    num_hidden_layers: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    norm: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array, inputs_embeds: mx.array = None, mask: mx.array = None, cache=None): ...

class LanguageModel(nn.Module):
    config: Incomplete
    model_type: Incomplete
    model: Incomplete
    lm_head: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array, inputs_embeds: mx.array | None = None, mask: mx.array | None = None, cache=None): ...
    def sanitize(self, weights): ...
    @property
    def layers(self): ...
    @property
    def head_dim(self): ...
    @property
    def n_kv_heads(self): ...
    def make_cache(self): ...
