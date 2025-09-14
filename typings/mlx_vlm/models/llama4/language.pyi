import mlx.core as mx
import mlx.nn as nn
from ..base import LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from ..cache import ChunkedKVCache as ChunkedKVCache, KVCache as KVCache
from .config import TextConfig as TextConfig
from _typeshed import Incomplete
from typing import Any

class Attention(nn.Module):
    n_heads: Incomplete
    n_kv_heads: Incomplete
    use_rope: Incomplete
    attn_temperature_tuning: Incomplete
    floor_scale: Incomplete
    attn_scale: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    use_qk_norm: Incomplete
    rope: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None) -> mx.array: ...

class MLP(nn.Module):
    gate_proj: Incomplete
    down_proj: Incomplete
    up_proj: Incomplete
    def __init__(self, config: TextConfig, intermediate_size: int = None) -> None: ...
    def __call__(self, x) -> mx.array: ...

class MoE(nn.Module):
    top_k: Incomplete
    num_experts: Incomplete
    experts: Incomplete
    router: Incomplete
    shared_expert: Incomplete
    def __init__(self, config) -> None: ...
    def __call__(self, x) -> mx.array: ...

class TransformerBlock(nn.Module):
    num_attention_heads: Incomplete
    hidden_size: Incomplete
    self_attn: Incomplete
    use_chunked_attention: Incomplete
    is_moe_layer: Incomplete
    feed_forward: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    config: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None) -> mx.array: ...

class LlamaModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    num_hidden_layers: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    norm: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def create_chunked_attention_mask(self, seq_len: int, attention_chunk_size: int, start: int = 0, offset: int = 0) -> mx.array: ...
    def __call__(self, input_ids: mx.array = None, input_embeds: mx.array = None, mask: mx.array = None, cache=None): ...

class LanguageModel(nn.Module):
    config: Incomplete
    model_type: Incomplete
    model: Incomplete
    lm_head: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, input_ids: mx.array = None, input_embeds: mx.array = None, mask: mx.array = None, cache=None): ...
    def sanitize(self, weights): ...
    @property
    def layers(self): ...
    @property
    def n_kv_heads(self): ...
    @property
    def head_dim(self): ...
    def make_cache(self): ...
