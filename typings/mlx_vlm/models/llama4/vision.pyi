import mlx.core as mx
import mlx.nn as nn
from ..base import pixel_shuffle as pixel_shuffle
from .config import VisionConfig as VisionConfig
from _typeshed import Incomplete

def check_array_shape(arr): ...

class Llama4MultiModalProjector(nn.Module):
    linear_1: Incomplete
    def __init__(self, config) -> None: ...
    def __call__(self, image_features): ...

class Llama4VisionPixelShuffleMLP(nn.Module):
    pixel_shuffle_ratio: Incomplete
    inner_dim: Incomplete
    output_dim: Incomplete
    mlp: Incomplete
    def __init__(self, config) -> None: ...
    def __call__(self, encoded_patches: mx.array) -> mx.array: ...

def reshape_for_broadcast(freqs_ci: mx.array, query: mx.array): ...
def view_as_complex(x): ...
def view_as_real(x): ...
def vision_apply_rotary_emb(query: mx.array, key: mx.array, freqs_ci: mx.array) -> tuple[mx.array, mx.array]: ...

class Llama4VisionAttention(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    num_key_value_groups: int
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, hidden_states: mx.array, freqs_ci: mx.array, mask: mx.array | None = None, cache: mx.array | None = None): ...

class Llama4VisionMLP(nn.Module):
    config: Incomplete
    activation_fn: Incomplete
    is_projector: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(self, config, bias: bool = True, is_projector: bool = False) -> None: ...
    def __call__(self, hidden_states: mx.array) -> mx.array: ...

class Llama4VisionEncoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, hidden_state: mx.array, freqs_ci: mx.array, mask: mx.array | None = None): ...

class Llama4VisionEncoder(nn.Module):
    config: Incomplete
    layers: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, hidden_states: mx.array, freqs_ci: mx.array, mask: mx.array | None = None): ...

class Llama4UnfoldConvolution(nn.Module):
    kernel_size: Incomplete
    stride: Incomplete
    linear: Incomplete
    def __init__(self, config) -> None: ...
    def unfold(self, input_tensor): ...
    def __call__(self, hidden_states: mx.array) -> mx.array: ...

class Llama4VisionRotaryEmbedding:
    freqs_ci: Incomplete
    def __init__(self, config) -> None: ...
    def __call__(self, hidden_states): ...

class VisionModel(nn.Module):
    image_size: Incomplete
    patch_size: Incomplete
    hidden_size: Incomplete
    num_channels: Incomplete
    model_type: Incomplete
    num_patches: Incomplete
    scale: Incomplete
    class_embedding: Incomplete
    positional_embedding_vlm: Incomplete
    patch_embedding: Incomplete
    rotary_embedding: Incomplete
    layernorm_pre: Incomplete
    layernorm_post: Incomplete
    model: Incomplete
    vision_adapter: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def get_input_embeddings(self): ...
    def __call__(self, pixel_values: mx.array, output_attentions: bool | None = None, output_hidden_states: bool | None = None, capture_activations: bool | None = True): ...
    def sanitize(self, weights): ...
