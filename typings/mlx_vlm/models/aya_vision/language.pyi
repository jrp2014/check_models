import mlx.core as mx
import mlx.nn as nn
from ..base import LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from ..cache import KVCache as KVCache, RotatingKVCache as RotatingKVCache
from .config import TextConfig as TextConfig
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass

class Attention(nn.Module):
    config: Incomplete
    layer_idx: Incomplete
    n_heads: Incomplete
    n_kv_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    rope: Incomplete
    use_sliding_window: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: tuple[mx.array, mx.array] | None = None) -> mx.array: ...

class MLP(nn.Module):
    gate_proj: Incomplete
    up_proj: Incomplete
    down_proj: Incomplete
    def __init__(self, dim, hidden_dim) -> None: ...
    def __call__(self, x): ...

class TransformerBlock(nn.Module):
    hidden_size: Incomplete
    n_heads: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    config: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: tuple[mx.array, mx.array] | None = None) -> mx.array: ...

class CohereModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    num_hidden_layers: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    norm: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array, inputs_embeds: mx.array = None, mask: mx.array = None, cache=None): ...

class LanguageModel(nn.Module):
    model_type: Incomplete
    model: Incomplete
    config: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array, inputs_embeds: mx.array = None, mask: mx.array = None, cache=None): ...
    def make_cache(self): ...
    @property
    def layers(self): ...
    @property
    def head_dim(self): ...
    @property
    def n_kv_heads(self): ...
