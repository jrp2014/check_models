import mlx.core as mx
import mlx.nn as nn
from ..kernels import bicubic_interpolate as bicubic_interpolate, nearest_interpolate as nearest_interpolate
from _typeshed import Incomplete
from mlx_vlm.models.gemma3n.config import EdgeResidualConfig as EdgeResidualConfig, MultiQueryAttentionBlockConfig as MultiQueryAttentionBlockConfig, UniversalInvertedResidualConfig as UniversalInvertedResidualConfig, VisionConfig as VisionConfig

class MobileNetV5MultiScaleFusionAdapter(nn.Module):
    in_channels: Incomplete
    out_channels: Incomplete
    output_resolution: Incomplete
    expansion_ratio: Incomplete
    interpolation_mode: Incomplete
    use_layer_scale: Incomplete
    layer_scale_init_value: Incomplete
    noskip: Incomplete
    ffn: Incomplete
    norm: Incomplete
    def __init__(self, in_chs: list[int], out_chs: int, output_resolution: int, expansion_ratio: float = 2.0, interpolation_mode: str = 'nearest', use_layer_scale: bool = False, layer_scale_init_value: float = 1e-05, noskip: bool = True) -> None: ...
    def __call__(self, inputs: list[mx.array]) -> mx.array: ...

class LayerScale2d(nn.Module):
    inplace: Incomplete
    gamma: Incomplete
    def __init__(self, dim: int, init_values: float = 1e-05, inplace: bool = False) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

def rms_norm2d(x: mx.array, normalized_shape: list[int], weight: mx.array | None = None, eps: float = 1e-05): ...

class RMSNormAct2d(nn.RMSNorm):
    normalized_shape: Incomplete
    drop: Incomplete
    act: Incomplete
    def __init__(self, num_channels, eps: float = 1e-06, apply_act: bool = True) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class UniversalInvertedResidual(nn.Module):
    has_skip: Incomplete
    dw_start: Incomplete
    pw_exp: Incomplete
    dw_mid: Incomplete
    pw_proj: Incomplete
    layer_scale: Incomplete
    def __init__(self, in_chs: int, out_chs: int, dw_kernel_size_start: int = 0, dw_kernel_size_mid: int = 3, dw_kernel_size_end: int = 0, stride: int = 1, dilation: int = 1, group_size: int = 1, pad_type: str = '', noskip: bool = False, exp_ratio: float = 1.0, norm_layer=..., conv_kwargs: dict | None = None, drop_path_rate: float = 0.0, layer_scale_init_value: float | None = 1e-05) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class ConvNormAct(nn.Module):
    out_chs: Incomplete
    conv: Incomplete
    bn: Incomplete
    def __init__(self, conv_cls, in_chs: int, out_chs: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False, apply_act: bool = True, eps: float = 1e-06) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

def pad_same(x, kernel_size: list[int], stride: list[int], dilation: list[int] = (1, 1), value: float = 0): ...
def get_padding_value(padding, kernel_size, **kwargs) -> tuple[tuple, bool]: ...
def get_same_padding(input_size: int, kernel_size: int, stride: int, dilation: int = 1) -> int: ...
def get_padding(kernel_size, stride: int = 1, dilation: int = 1, **_): ...
def is_static_pad(kernel_size, stride: int = 1, dilation: int = 1, **_): ...

class Conv2dSame(nn.Conv2d):
    kernel_size: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class EdgeResidual(nn.Module):
    has_skip: Incomplete
    conv_exp: Incomplete
    bn1: Incomplete
    conv_pwl: Incomplete
    bn2: Incomplete
    def __init__(self, in_chs: int, out_chs: int, exp_kernel_size: int = 3, stride: int = 1, dilation: int = 1, group_size: int = 0, pad_type: str = '', force_in_chs: int = 0, noskip: bool = False, expand_ratio: float = 1.0, pw_kernel_size: int = 1, norm_layer=...) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class MobileAttention(nn.Module):
    has_skip: Incomplete
    query_strides: Incomplete
    kv_stride: Incomplete
    has_query_stride: Incomplete
    norm: Incomplete
    attn: Incomplete
    layer_scale: Incomplete
    drop_path: Incomplete
    def __init__(self, in_chs: int, out_chs: int, stride: int = 1, dw_kernel_size: int = 3, dilation: int = 1, group_size: int = 1, pad_type: str = '', num_heads: int = 8, key_dim: int = 64, value_dim: int = 64, use_multi_query: bool = True, query_strides: tuple[int, int] = (1, 1), kv_stride: int = 1, cpe_dw_kernel_size: int = 3, noskip: bool = False, act_layer=..., aa_layer=None, drop_path_rate: float = 0.0, attn_drop: float = 0.0, proj_drop: float = 0.0, layer_scale_init_value: float | None = 1e-05, use_bias: bool = False) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

def create_conv2d(in_channels, out_channels, kernel_size, stride: int = 1, dilation: int = 1, depthwise: bool = False, bias: bool = False, **kwargs): ...
def to_2tuple(x): ...

class NamedSequential(nn.Module):
    def __init__(self) -> None: ...
    def add_module(self, name, module) -> None: ...
    def __call__(self, x): ...

class MultiQueryAttention2d(nn.Module):
    num_heads: Incomplete
    query_strides: Incomplete
    kv_stride: Incomplete
    fused_attn: bool
    key_dim: Incomplete
    value_dim: Incomplete
    scale: Incomplete
    query: Incomplete
    key: Incomplete
    value: Incomplete
    attn_drop: Incomplete
    output: Incomplete
    proj_drop: Incomplete
    def __init__(self, dim: int, dim_out: int | None = None, num_heads: int = 8, key_dim: int = 64, value_dim: int = 64, query_strides: tuple[int, int] = (1, 1), kv_stride: int = 1, dilation: int = 1, padding: str = '', dw_kernel_size: int = 3, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None: ...
    def __call__(self, x: mx.array, attn_mask: mx.array | None = None) -> mx.array: ...

def num_groups(group_size: int | None, channels: int) -> int: ...
def make_divisible(v, divisor: int = 8, min_value=None, round_limit: float = 0.9): ...
def gemma3n_mobilenet_def(): ...

class VisionTower(nn.Module):
    conv_stem: Incomplete
    num_features: Incomplete
    msfa_indices: Incomplete
    msfa_output_resolution: Incomplete
    msfa: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def build(self): ...
    def __call__(self, x: mx.array, output_hidden_states: bool | None = None) -> mx.array: ...

class VisionModel(nn.Module):
    model_type: Incomplete
    timm_model: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array, output_hidden_states: bool | None = None) -> mx.array: ...
    def sanitize(self, weights): ...
