import mlx.core as mx
import mlx.nn as nn
from ..kernels import bicubic_interpolate as bicubic_interpolate
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

class Learnable2DInterpPosEmb(nn.Module):
    height: Incomplete
    width: Incomplete
    interpolation_mode: Incomplete
    weight: Incomplete
    def __init__(self, height: int, width: int, dim: int, interpolation_mode: str = 'bicubic') -> None: ...
    def __call__(self, x: mx.array, grid_hws: mx.array) -> mx.array: ...

class PatchEmbed(nn.Module):
    patch_size: Incomplete
    num_channels: Incomplete
    embed_dim: Incomplete
    init_pos_emb_height: Incomplete
    proj: Incomplete
    pos_emb: Incomplete
    def __init__(self, patch_size: int = 14, num_channels: int = 3, embed_dim: int = 1152, init_pos_emb_height: int = 64) -> None: ...
    def __call__(self, hidden_states: mx.array, grid_thw: mx.array) -> mx.array: ...

def view_as_complex(x): ...
def view_as_real(x): ...
def apply_rope(q: mx.array, k: mx.array, freqs_cis: mx.array) -> tuple[mx.array, mx.array]: ...

class Attention(nn.Module):
    num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    wqkv: Incomplete
    wo: Incomplete
    def __init__(self, dim: int, num_heads: int = 16) -> None: ...
    def __call__(self, x: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array = None) -> mx.array: ...

class MLP(nn.Module):
    activation_fn: Incomplete
    fc0: Incomplete
    fc1: Incomplete
    def __init__(self, dim, hidden_dim) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Qwen2VLVisionBlock(nn.Module):
    norm0: Incomplete
    norm1: Incomplete
    attn: Incomplete
    mlp: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, hidden_states, cu_seqlens, rotary_pos_emb) -> mx.array: ...

class Rope2DPosEmb(nn.Module):
    dim: Incomplete
    max_height: Incomplete
    max_width: Incomplete
    theta_base: Incomplete
    def __init__(self, dim: int, max_height: int, max_width: int, theta_base: int = 10000) -> None: ...
    def extra_repr(self): ...
    def get_freqs_cis(self, grid_hws: mx.array) -> mx.array: ...

def patch_merger(x: mx.array, grid_hws: mx.array, merge_kernel_size: list[int, int] = (2, 2)) -> list[mx.array]: ...

class VisionModel(nn.Module):
    config: Incomplete
    model_type: Incomplete
    spatial_merge_size: Incomplete
    merge_kernel_size: Incomplete
    patch_embed: Incomplete
    rotary_pos_emb: Incomplete
    blocks: Incomplete
    final_layernorm: Incomplete
    rope_pos_emb: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, hidden_states: mx.array, grid_thw: mx.array, output_hidden_states: bool | None = None) -> mx.array: ...
    def sanitize(self, weights): ...
