import mlx.core as mx
import mlx.nn as nn
from ..base import BaseModelConfig as BaseModelConfig
from .config import VisionConfig as VisionConfig
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass, field as field

def check_array_shape(arr): ...

class MllamaVisionAttention(nn.Module):
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, hidden_state: mx.array, attention_mask: mx.array | None = None) -> mx.array: ...

class MllamaVisionMLP(nn.Module):
    fc1: Incomplete
    fc2: Incomplete
    gelu: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, hidden_states: mx.array) -> mx.array: ...

class MllamaVisionEncoderLayer(nn.Module):
    hidden_size: Incomplete
    num_attention_heads: Incomplete
    is_gated: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    gate_attn: Incomplete
    gate_ffn: Incomplete
    def __init__(self, config: VisionConfig, is_gated: bool = False) -> None: ...
    def __call__(self, hidden_state: mx.array, attention_mask: mx.array | None = None) -> mx.array: ...

class MllamaVisionEncoder(nn.Module):
    layers: Incomplete
    def __init__(self, config: VisionConfig, num_layers: int = 32, is_gated: bool = False) -> None: ...
    def __call__(self, hidden_states: mx.array, attention_mask: mx.array | None = None) -> tuple[mx.array, list[mx.array]]: ...

class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    max_num_tiles: Incomplete
    hidden_size: Incomplete
    max_aspect_ratio_id: Incomplete
    is_gated: Incomplete
    embedding: Incomplete
    gate: Incomplete
    def __init__(self, config: VisionConfig, is_gated: bool = True) -> None: ...
    def __call__(self, hidden_state: mx.array, aspect_ratio_ids: mx.array) -> mx.array: ...

class MllamaPrecomputedPositionEmbedding(nn.Module):
    max_num_tiles: Incomplete
    max_aspect_ratio_id: Incomplete
    num_patches: Incomplete
    hidden_size: Incomplete
    scale: Incomplete
    gate: Incomplete
    embedding: Incomplete
    tile_embedding: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, hidden_state: mx.array, aspect_ratio_ids: mx.array) -> mx.array: ...

class VisionModel(nn.Module):
    image_size: Incomplete
    patch_size: Incomplete
    max_num_tiles: Incomplete
    hidden_size: Incomplete
    num_channels: Incomplete
    intermediate_layers_indices: Incomplete
    num_patches: Incomplete
    scale: Incomplete
    patch_embedding: Incomplete
    class_embedding: Incomplete
    gated_positional_embedding: Incomplete
    pre_tile_positional_embedding: Incomplete
    post_tile_positional_embedding: Incomplete
    layernorm_pre: Incomplete
    layernorm_post: Incomplete
    transformer: Incomplete
    global_transformer: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, pixel_values: mx.array, aspect_ratio_ids: mx.array, aspect_ratio_mask: mx.array) -> mx.array: ...
    @staticmethod
    def sanitize(weights): ...
