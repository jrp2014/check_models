import mlx.core as mx
import mlx.nn as nn
from ..kernels import bicubic_interpolate as bicubic_interpolate
from .config import VisionConfig as VisionConfig
from _typeshed import Incomplete

class Attention(nn.Module):
    num_heads: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    out_proj: Incomplete
    def __init__(self, dims: int, num_heads: int, query_input_dims: int | None = None, key_input_dims: int | None = None, value_input_dims: int | None = None, value_dims: int | None = None, value_output_dims: int | None = None, bias: bool = True) -> None: ...
    def __call__(self, x, mask=None): ...

class MLP(nn.Module):
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class EncoderLayer(nn.Module):
    embed_dim: Incomplete
    self_attn: Incomplete
    layer_norm1: Incomplete
    mlp: Incomplete
    layer_norm2: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
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
    patch_embedding: Incomplete
    num_patches: Incomplete
    position_embedding_size: Incomplete
    position_embedding: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    @staticmethod
    def resize_positional_embeddings(positional_embeddings: mx.array, spatial_shapes: mx.array, max_length: int) -> mx.array: ...
    def __call__(self, pixel_values: mx.array, spatial_shapes: mx.array = None) -> mx.array: ...

class VisionModel(nn.Module):
    model_type: Incomplete
    embeddings: Incomplete
    encoder: Incomplete
    post_layernorm: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array, output_hidden_states: bool | None = None, spatial_shapes: mx.array | None = None) -> mx.array: ...
    def sanitize(self, weights): ...
