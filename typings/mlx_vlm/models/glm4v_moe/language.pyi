import mlx.core as mx
import mlx.nn as nn
from ..base import LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from .config import ModelConfig as ModelConfig, TextConfig as TextConfig
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass
from functools import partial as partial
from typing import Any

class GLM4VRotaryEmbedding(nn.Module):
    rope_type: Incomplete
    max_seq_len_cached: Incomplete
    original_max_seq_len: Incomplete
    config: Incomplete
    rope_init_fn: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x, position_ids): ...

def rotate_half_llm(x): ...
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section): ...

class Glm4vMoeAttention(nn.Module):
    n_heads: Incomplete
    n_kv_heads: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    rope_scaling: Incomplete
    def __init__(self, args: TextConfig) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None, position_embeddings: mx.array | None = None) -> mx.array: ...

class Glm4vMoeMLP(nn.Module):
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

class MoE(nn.Module):
    config: Incomplete
    num_experts_per_tok: Incomplete
    switch_mlp: Incomplete
    gate: Incomplete
    shared_experts: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x): ...

class Glm4vMoeDecoderLayer(nn.Module):
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None, position_embeddings: mx.array | None = None) -> mx.array: ...

class GLM4VModel(nn.Module):
    vocab_size: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    start_idx: int
    end_idx: Incomplete
    num_layers: Incomplete
    norm: Incomplete
    rotary_emb: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array, inputs_embeds: mx.array | None = None, cache: Any | None = None, mask: mx.array | None = None, position_ids: mx.array | None = None) -> mx.array: ...

class LanguageModel(nn.Module):
    args: Incomplete
    config: Incomplete
    model_type: Incomplete
    model: Incomplete
    lm_head: Incomplete
    rope_deltas: Incomplete
    def __init__(self, args: TextConfig, config: ModelConfig = None) -> None: ...
    def get_rope_index(self, input_ids: mx.array, image_grid_thw: mx.array | None = None, video_grid_thw: mx.array | None = None, attention_mask: mx.array | None = None): ...
    def __call__(self, inputs: mx.array, inputs_embeds: mx.array | None = None, mask: mx.array | None = None, cache=None, **kwargs): ...
    def sanitize(self, weights): ...
    @property
    def layers(self): ...
    @property
    def cast_predicate(self): ...
    @property
    def n_kv_heads(self): ...
