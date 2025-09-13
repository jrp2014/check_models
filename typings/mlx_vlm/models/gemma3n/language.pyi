import mlx.core as mx
import mlx.nn as nn
from ..base import LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from ..cache import KVCache as KVCache, RotatingKVCache as RotatingKVCache
from .config import TextConfig as TextConfig
from _typeshed import Incomplete
from typing import Any

class Gemma3nRMSNorm(nn.Module):
    eps: Incomplete
    scale_shift: Incomplete
    with_scale: Incomplete
    weight: Incomplete
    def __init__(self, dim: int, eps: float = 1e-06, scale_shift: float = 0.0, with_scale: bool = True) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class RMSNoScale(nn.Module):
    eps: Incomplete
    def __init__(self, eps: float = 1e-05) -> None: ...
    def __call__(self, x): ...

class Gemma3nLaurelBlock(nn.Module):
    config: Incomplete
    linear_left: Incomplete
    linear_right: Incomplete
    post_laurel_norm: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Gemma3nAttention(nn.Module):
    is_sliding: Incomplete
    n_heads: Incomplete
    n_kv_heads: Incomplete
    repeats: Incomplete
    head_dim: Incomplete
    layer_idx: Incomplete
    scale: float
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    v_norm: Incomplete
    is_kv_shared_layer: Incomplete
    rope: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int, is_kv_shared_layer: bool) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None) -> mx.array: ...

def gelu_topk(inputs, std_multiplier): ...

class MLP(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    gate_proj: Incomplete
    up_proj: Incomplete
    down_proj: Incomplete
    activation_sparsity: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int = 0) -> None: ...
    def __call__(self, x: mx.array): ...

class Gemma3nAltUp(nn.Module):
    config: Incomplete
    correct_output_scale: Incomplete
    correction_coefs: Incomplete
    prediction_coefs: Incomplete
    modality_router: Incomplete
    router_norm: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def compute_router_modalities(self, x: mx.array) -> mx.array: ...
    def predict(self, x: mx.array) -> mx.array: ...
    def correct(self, predictions: mx.array, activated: mx.array): ...

class Gemma3nDecoderLayer(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    layer_idx: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    pre_feedforward_layernorm: Incomplete
    post_feedforward_layernorm: Incomplete
    is_sliding: Incomplete
    sliding_window: Incomplete
    hidden_size_per_layer_input: Incomplete
    altup: Incomplete
    laurel: Incomplete
    per_layer_input_gate: Incomplete
    per_layer_projection: Incomplete
    post_per_layer_input_norm: Incomplete
    def __init__(self, config: TextConfig, layer_idx: int, is_kv_shared_layer: bool) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: Any | None = None, per_layer_input: mx.array | None = None): ...

class Gemma3Model(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    hidden_size_per_layer_input: Incomplete
    vocab_size: Incomplete
    vocab_size_per_layer_input: Incomplete
    num_hidden_layers: Incomplete
    first_kv_shared_layer_idx: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    embed_tokens_per_layer: Incomplete
    per_layer_model_projection: Incomplete
    per_layer_projection_norm: Incomplete
    altup_projections: Incomplete
    altup_unembed_projections: Incomplete
    norm: Incomplete
    first_sliding_idx: Incomplete
    first_full_idx: Incomplete
    layer_idx_to_cache_idx: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array = None, inputs_embeds: mx.array = None, mask: mx.array = None, cache=None, **kwargs): ...
    def get_per_layer_inputs(self, input_ids: mx.array) -> mx.array: ...
    def project_per_layer_inputs(self, inputs_embeds: mx.array, per_layer_inputs: mx.array) -> mx.array: ...

def logit_softcap(softcap, x): ...

class LanguageModel(nn.Module):
    config: Incomplete
    model_type: Incomplete
    model: Incomplete
    final_logit_softcapping: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array = None, inputs_embeds: mx.array | None = None, mask: mx.array | None = None, cache=None, **kwargs): ...
    def sanitize(self, weights): ...
    @property
    def layers(self): ...
    @property
    def head_dim(self): ...
    @property
    def n_kv_heads(self): ...
    def make_cache(self): ...
