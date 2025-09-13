import mlx.core as mx
import mlx.nn as nn
from ..base import check_array_shape as check_array_shape
from .config import AudioConfig as AudioConfig, ModelConfig as ModelConfig
from .language import Gemma3nRMSNorm as Gemma3nRMSNorm
from _typeshed import Incomplete

def convert_torch_to_mlx_pad_width(padding, input_shape): ...

class Gemma3nAudioRelativePositionEmbedding(nn.Module):
    config: Incomplete
    num_heads: Incomplete
    channels: Incomplete
    head_dim: Incomplete
    max_backward: Incomplete
    max_forward: Incomplete
    pos_proj: Incomplete
    def __init__(self, config: AudioConfig, *args, **kwargs) -> None: ...
    def __call__(self, queries: mx.array, keys: mx.array) -> mx.array: ...

class Gemma3nAudioAttention(nn.Module):
    config: Incomplete
    num_heads: Incomplete
    hidden_size: Incomplete
    head_dim: Incomplete
    chunk_size: Incomplete
    max_future_horizon: Incomplete
    max_past_horizon: Incomplete
    attention_invalid_logits_value: Incomplete
    attention_logits_soft_cap: Incomplete
    context_size: Incomplete
    relative_position_embedding: Incomplete
    per_dim_scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    def __init__(self, config: AudioConfig, *args, **kwargs) -> None: ...
    def unfold_mlx(self, x, dimension, size, step): ...
    def __call__(self, x: mx.array, mask: mx.array) -> mx.array: ...

class Gemma3nCumulativeGroupNorm(nn.Module):
    num_channels: Incomplete
    feature_dims: Incomplete
    eps: Incomplete
    use_scale: Incomplete
    use_bias: Incomplete
    weight: Incomplete
    bias: Incomplete
    reduction_axes: Incomplete
    def __init__(self, num_channels: int, feature_dims: tuple[int], eps: float = 0.001, use_scale: bool = True, use_bias: bool = False) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array: ...

class Gemma3nAudioSSCPConvBlock(nn.Module):
    config: Incomplete
    manual_padding: Incomplete
    conv: Incomplete
    norm: Incomplete
    def __init__(self, idx: int, input_freq_dim: int, config: AudioConfig, manual_padding: tuple[int, int, int, int] = (0, 0, 0, 0), *args, **kwargs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Gemma3nAudioSubSampleConvProjection(nn.Module):
    config: Incomplete
    conv_0: Incomplete
    conv_1: Incomplete
    input_proj_in_features: Incomplete
    input_proj_linear: Incomplete
    def __init__(self, config: AudioConfig, *args, **kwargs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Gemma3nAudioConformerAttention(nn.Module):
    config: Incomplete
    post_in_shape: Incomplete
    post_in_features: Incomplete
    pre_attn_norm: Incomplete
    attn: Incomplete
    post: Incomplete
    post_norm: Incomplete
    def __init__(self, config: AudioConfig, *args, **kwargs) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array) -> mx.array: ...

class Gemma3nAudioConformerFeedForward(nn.Module):
    config: Incomplete
    pre_layer_norm: Incomplete
    ffw_layer_1: Incomplete
    ffw_layer_2: Incomplete
    post_layer_norm: Incomplete
    def __init__(self, config: AudioConfig, *args, **kwargs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Gemma3nAudioConformerLightConv1d(nn.Module):
    config: Incomplete
    pre_layer_norm: Incomplete
    linear_start: Incomplete
    depthwise_conv1d: Incomplete
    conv_norm: Incomplete
    linear_end: Incomplete
    causal_padding: Incomplete
    def __init__(self, config: AudioConfig, *args, **kwargs) -> None: ...
    def __call__(self, audio_encodings: mx.array) -> mx.array: ...

class Gemma3nAudioConformerBlock(nn.Module):
    config: Incomplete
    ffw_layer_start: Incomplete
    attention: Incomplete
    lconv1d: Incomplete
    ffw_layer_end: Incomplete
    norm: Incomplete
    def __init__(self, config: AudioConfig, *args, **kwargs) -> None: ...
    def __call__(self, audio_encodings: mx.array, audio_mel_mask: mx.array) -> mx.array: ...

class AudioModel(nn.Module):
    config: Incomplete
    subsample_conv_projection: Incomplete
    conformer: Incomplete
    def __init__(self, config: AudioConfig, *args, **kwargs) -> None: ...
    def __call__(self, audio_mel: mx.array, audio_mel_mask: mx.array) -> tuple[mx.array, mx.array]: ...
    def sanitize(self, weights): ...
