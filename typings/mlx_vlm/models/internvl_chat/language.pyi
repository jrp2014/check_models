import mlx.core as mx
import mlx.nn as nn
from ..base import LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from ..cache import KVCache as KVCache
from .config import TextConfig as TextConfig
from _typeshed import Incomplete

class Attention(nn.Module):
    n_heads: Incomplete
    n_kv_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    def __init__(self, args: TextConfig) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array: ...

class MLP(nn.Module):
    gate_proj: Incomplete
    down_proj: Incomplete
    up_proj: Incomplete
    def __init__(self, dim, hidden_dim) -> None: ...
    def __call__(self, x) -> mx.array: ...

class Qwen2VLDecoderLayer(nn.Module):
    num_attention_heads: Incomplete
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    args: Incomplete
    def __init__(self, args: TextConfig) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array: ...

class Qwen2Model(nn.Module):
    args: Incomplete
    vocab_size: Incomplete
    num_hidden_layers: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    norm: Incomplete
    def __init__(self, args: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array, inputs_embeds: mx.array | None = None, mask: mx.array | None = None, cache=None): ...

class LanguageModel(nn.Module):
    args: Incomplete
    model_type: Incomplete
    model: Incomplete
    lm_head: Incomplete
    def __init__(self, args: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array, inputs_embeds: mx.array | None = None, mask: mx.array | None = None, cache=None): ...
    @property
    def layers(self): ...
    @property
    def head_dim(self): ...
    @property
    def n_kv_heads(self): ...
