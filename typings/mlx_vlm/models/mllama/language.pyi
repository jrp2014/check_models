import mlx.core as mx
import mlx.nn as nn
from ..base import BaseModelConfig as BaseModelConfig, LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from ..cache import KVCache as KVCache
from .config import TextConfig as TextConfig
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass, field as field

class MllamaTextCrossAttention(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    num_key_value_heads: Incomplete
    num_key_value_groups: Incomplete
    layer_idx: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int | None = None) -> None: ...
    def __call__(self, hidden_states: mx.array, cross_attention_states: mx.array | None = None, attention_mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array: ...

class MllamaTextSelfAttention(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    num_key_value_heads: Incomplete
    num_key_value_groups: Incomplete
    scale: Incomplete
    layer_idx: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    rope: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array: ...

class MllamaTextMLP(nn.Module):
    gate_proj: Incomplete
    up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x): ...

class MllamaSelfAttentionDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, hidden_states: mx.array, mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array: ...

class MllamaCrossAttentionDecoderLayer(nn.Module):
    hidden_size: Incomplete
    cross_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    cross_attn_attn_gate: Incomplete
    cross_attn_mlp_gate: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, hidden_states: mx.array, cross_attention_states: mx.array, attention_mask: mx.array | None = None, full_text_row_masked_out_mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array: ...

class MllamaTextModel(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    hidden_size: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    norm: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, input_ids: mx.array | None = None, mask: mx.array | None = None, position_ids: mx.array | None = None, cross_attention_states: mx.array | None = None, cross_attention_mask: mx.array | None = None, full_text_row_masked_out_mask: mx.array | None = None, inputs_embeds: mx.array | None = None, cache: KVCache | None = None) -> mx.array: ...

class LanguageModel(nn.Module):
    config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, input_ids: mx.array | None = None, mask: mx.array | None = None, cross_attention_states: mx.array | None = None, cross_attention_mask: mx.array | None = None, full_text_row_masked_out_mask: mx.array | None = None, inputs_embeds: mx.array | None = None, cache: KVCache | None = None) -> tuple[mx.array, mx.array | None]: ...
    @staticmethod
    def sanitize(weights): ...
    @property
    def layers(self): ...
    @property
    def head_dim(self): ...
    @property
    def n_kv_heads(self): ...
