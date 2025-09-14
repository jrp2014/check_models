import mlx.core as mx
import mlx.nn as nn
from .config import VisionConfig as VisionConfig
from _typeshed import Incomplete

def check_array_shape(arr): ...
def position_ids_in_meshgrid(patch_embeds_list, max_width): ...
def generate_block_attention_mask(patch_embeds_list, tensor): ...
def rotate_half(x): ...
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1): ...

class Attention(nn.Module):
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    q_proj: Incomplete
    o_proj: Incomplete
    def __init__(self, dims: int, num_heads: int, query_input_dims: int | None = None, key_input_dims: int | None = None, value_input_dims: int | None = None, value_dims: int | None = None, value_output_dims: int | None = None, bias: bool = False) -> None: ...
    def __call__(self, queries, keys, values, position_embeddings, mask=None): ...

class MLP(nn.Module):
    gate_proj: Incomplete
    down_proj: Incomplete
    up_proj: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x) -> mx.array: ...

class EncoderLayer(nn.Module):
    embed_dim: Incomplete
    attention: Incomplete
    attention_norm: Incomplete
    feed_forward: Incomplete
    ffn_norm: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array, position_embeddings: mx.array, mask: mx.array | None = None) -> mx.array: ...

class Encoder(nn.Module):
    layers: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...

class PixtralRotaryEmbedding:
    dim: Incomplete
    base: Incomplete
    inv_freq: Incomplete
    def __init__(self, config) -> None: ...
    def __call__(self, x, position_ids): ...

class PixtralVisionModel(nn.Module):
    config: Incomplete
    patch_conv: Incomplete
    ln_pre: Incomplete
    transformer: Incomplete
    patch_positional_embedding: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: list[mx.array], output_hidden_states: bool | None = None) -> mx.array: ...

class VisionModel(nn.Module):
    model_type: Incomplete
    vision_model: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: list[mx.array], output_hidden_states: bool | None = None) -> mx.array: ...
    def sanitize(self, weights): ...
