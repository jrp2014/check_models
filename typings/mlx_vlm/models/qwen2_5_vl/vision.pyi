import mlx.core as mx
import mlx.nn as nn
from .config import VisionConfig as VisionConfig
from _typeshed import Incomplete

def check_array_shape(arr): ...
def rotate_half(x): ...
def apply_rotary_pos_emb_vision(tensor, freqs) -> mx.array: ...

class VisionRotaryEmbedding(nn.Module):
    dim: Incomplete
    theta: Incomplete
    def __init__(self, dim: int, theta: float = 10000.0) -> None: ...
    def __call__(self, seqlen: int) -> mx.array: ...

class PatchEmbed(nn.Module):
    patch_size: Incomplete
    temporal_patch_size: Incomplete
    in_channels: Incomplete
    hidden_size: Incomplete
    proj: Incomplete
    def __init__(self, patch_size: int = 14, temporal_patch_size: int = 2, in_channels: int = 3, hidden_size: int = 1152) -> None: ...
    def __call__(self, hidden_states: mx.array) -> mx.array: ...

class PatchMerger(nn.Module):
    hidden_size: Incomplete
    ln_q: Incomplete
    mlp: Incomplete
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Attention(nn.Module):
    num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    qkv: Incomplete
    proj: Incomplete
    def __init__(self, dim: int, num_heads: int = 16) -> None: ...
    def __call__(self, x: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array = None) -> mx.array: ...

class MLP(nn.Module):
    gate_proj: Incomplete
    up_proj: Incomplete
    down_proj: Incomplete
    def __init__(self, dim, hidden_dim) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Qwen2VLVisionBlock(nn.Module):
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
    patch_embed: Incomplete
    window_size: Incomplete
    patch_size: Incomplete
    spatial_merge_unit: Incomplete
    fullatt_block_indexes: Incomplete
    rotary_pos_emb: Incomplete
    blocks: Incomplete
    merger: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def rot_pos_emb(self, grid_thw): ...
    def get_window_index(self, grid_thw): ...
    def __call__(self, hidden_states: mx.array, grid_thw: mx.array, output_hidden_states: bool | None = None) -> mx.array: ...
    def sanitize(self, weights): ...
