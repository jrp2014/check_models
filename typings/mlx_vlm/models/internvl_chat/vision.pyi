import mlx.core as mx
import mlx.nn as nn
from ..base import interpolate as interpolate
from .config import VisionConfig as VisionConfig
from _typeshed import Incomplete

def check_array_shape(arr): ...

class Attention(nn.Module):
    dims: Incomplete
    num_heads: Incomplete
    scale: Incomplete
    qkv_bias: Incomplete
    qkv: Incomplete
    proj: Incomplete
    qk_normalization: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x, mask=None): ...

class MLP(nn.Module):
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class EncoderLayer(nn.Module):
    embed_dim: Incomplete
    intermediate_size: Incomplete
    norm_type: Incomplete
    attn: Incomplete
    mlp: Incomplete
    norm1: Incomplete
    norm2: Incomplete
    ls1: Incomplete
    ls2: Incomplete
    drop_path1: Incomplete
    drop_path2: Incomplete
    def __init__(self, config: VisionConfig, drop_path_rate: float = 0.0) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array: ...

class Encoder(nn.Module):
    layers: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array, output_hidden_states: bool | None = None, mask: mx.array | None = None) -> mx.array: ...

class VisionEmbeddings(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    class_embedding: Incomplete
    patch_embedding: Incomplete
    num_patches: Incomplete
    num_positions: Incomplete
    position_embedding: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class VisionModel(nn.Module):
    model_type: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array, output_hidden_states: bool | None = None) -> mx.array: ...
    def sanitize(self, weights): ...
