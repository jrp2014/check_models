import mlx.core as mx
import mlx.nn as nn
from ..kernels import grid_sample as grid_sample
from .config import VisionConfig as VisionConfig
from _typeshed import Incomplete

def check_array_shape(arr): ...
def rotate_half(x): ...
def apply_rotary_pos_emb_vision(tensor, freqs) -> mx.array: ...

class Glm4vMoeVisionRotaryEmbedding(nn.Module):
    dim: Incomplete
    theta: Incomplete
    def __init__(self, dim: int, theta: float = 10000.0) -> None: ...
    def __call__(self, seqlen: int) -> mx.array: ...

class Glm4vVisionEmbeddings(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    num_patches: Incomplete
    num_positions: Incomplete
    position_embedding: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, embeddings, lengths, image_shapes, h_coords, w_coords): ...

class Glm4vMoeVisionPatchEmbed(nn.Module):
    config: Incomplete
    patch_size: Incomplete
    temporal_patch_size: Incomplete
    in_channels: Incomplete
    embed_dim: Incomplete
    proj: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, hidden_states: mx.array) -> mx.array: ...

class Glm4vMoeVisionPatchMerger(nn.Module):
    proj: Incomplete
    post_projection_norm: Incomplete
    gate_proj: Incomplete
    up_proj: Incomplete
    down_proj: Incomplete
    def __init__(self, dim: int, context_dim: int, bias: bool = False) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Glm4vMoeVisionAttention(nn.Module):
    num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    qkv: Incomplete
    proj: Incomplete
    def __init__(self, dim: int, num_heads: int = 16) -> None: ...
    def __call__(self, x: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array = None) -> mx.array: ...

class Glm4vMoeVisionMLP(nn.Module):
    gate_proj: Incomplete
    up_proj: Incomplete
    down_proj: Incomplete
    def __init__(self, dim, hidden_dim) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Glm4vMoeVisionBlock(nn.Module):
    norm1: Incomplete
    norm2: Incomplete
    attn: Incomplete
    mlp: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> mx.array: ...

class VisionModel(nn.Module):
    config: Incomplete
    model_type: Incomplete
    spatial_merge_size: Incomplete
    embeddings: Incomplete
    patch_embed: Incomplete
    window_size: Incomplete
    patch_size: Incomplete
    spatial_merge_unit: Incomplete
    rotary_pos_emb: Incomplete
    blocks: Incomplete
    merger: Incomplete
    post_conv_layernorm: Incomplete
    downsample: Incomplete
    post_layernorm: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def rot_pos_emb(self, grid_thw): ...
    def __call__(self, hidden_states: mx.array, grid_thw: mx.array, output_hidden_states: bool | None = None) -> mx.array: ...
    def sanitize(self, weights): ...
