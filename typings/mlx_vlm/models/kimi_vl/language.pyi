import mlx.core as mx
import mlx.nn as nn
from ..base import LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from ..cache import KVCache as KVCache
from .config import TextConfig as TextConfig
from _typeshed import Incomplete
from typing import Any

def yarn_find_correction_dim(num_rotations, dim, base: int = 10000, max_position_embeddings: int = 2048): ...
def yarn_find_correction_range(low_rot, high_rot, dim, base: int = 10000, max_position_embeddings: int = 2048): ...
def yarn_get_mscale(scale: int = 1, mscale: int = 1): ...
def yarn_linear_ramp_mask(min_val, max_val, dim): ...

class DeepseekV3YarnRotaryEmbedding(nn.Module):
    mscale: Incomplete
    def __init__(self, dim, max_position_embeddings: int = 2048, base: int = 10000, scaling_factor: float = 1.0, original_max_position_embeddings: int = 4096, beta_fast: int = 32, beta_slow: int = 1, mscale: int = 1, mscale_all_dim: int = 0) -> None: ...
    def __call__(self, x, offset: int = 0): ...

def clipped_silu(x, gate): ...

class DeepseekV3Attention(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    num_heads: Incomplete
    max_position_embeddings: Incomplete
    rope_theta: Incomplete
    q_lora_rank: Incomplete
    qk_rope_head_dim: Incomplete
    kv_lora_rank: Incomplete
    v_head_dim: Incomplete
    qk_nope_head_dim: Incomplete
    q_head_dim: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    q_a_proj: Incomplete
    q_a_layernorm: Incomplete
    q_b_proj: Incomplete
    kv_a_proj_with_mqa: Incomplete
    kv_a_layernorm: Incomplete
    kv_b_proj: Incomplete
    o_proj: Incomplete
    rope: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None) -> mx.array: ...

class DeepseekV3MLP(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    gate_proj: Incomplete
    up_proj: Incomplete
    down_proj: Incomplete
    def __init__(self, config: TextConfig, hidden_size: int = None, intermediate_size: int = None) -> None: ...
    def __call__(self, x): ...

@mx.compile
def group_expert_select(gates, e_score_correction_bias, top_k, n_group, topk_group, routed_scaling_factor, norm_topk_prob): ...

class MoEGate(nn.Module):
    config: Incomplete
    top_k: Incomplete
    norm_topk_prob: Incomplete
    n_routed_experts: Incomplete
    routed_scaling_factor: Incomplete
    n_group: Incomplete
    topk_group: Incomplete
    weight: Incomplete
    e_score_correction_bias: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x): ...

class DeepseekV3MoE(nn.Module):
    config: Incomplete
    num_experts_per_tok: Incomplete
    switch_mlp: Incomplete
    gate: Incomplete
    shared_experts: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x): ...

class DeepseekV3DecoderLayer(nn.Module):
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None) -> mx.array: ...

class DeepseekV3Model(nn.Module):
    vocab_size: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    start_idx: int
    end_idx: Incomplete
    num_layers: Incomplete
    norm: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x: mx.array, inputs_embeds: mx.array | None = None, cache: Any | None = None, mask: mx.array | None = None) -> mx.array: ...

class LanguageModel(nn.Module):
    config: Incomplete
    model_type: Incomplete
    model: Incomplete
    lm_head: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array, inputs_embeds: mx.array | None = None, cache: Any | None = None, mask: mx.array | None = None): ...
    def sanitize(self, weights): ...
    def embed_tokens(self, x): ...
    @property
    def layers(self): ...
    @property
    def n_kv_heads(self): ...
    def cast_predicate(self): ...
