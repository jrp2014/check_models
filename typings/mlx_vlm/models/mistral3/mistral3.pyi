import mlx.core as mx
import mlx.nn as nn
from ..base import BaseModelConfig as BaseModelConfig
from ..pixtral import LanguageModel as LanguageModel, Model as PixtralModel, VisionModel as VisionModel
from .config import ModelConfig as ModelConfig
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass
from pathlib import Path as Path

def unfold(input: mx.array, kernel_size: int | tuple[int, int] | list[int], dilation: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 0, stride: int | tuple[int, int] | list[int] = 1) -> mx.array: ...

class Mistral3PatchMerger(nn.Module):
    config: Incomplete
    spatial_merge_size: Incomplete
    patch_size: Incomplete
    merging_layer: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, image_features: mx.array, image_sizes: mx.array) -> mx.array: ...

class Mistral3MultiModalProjector(nn.Module):
    norm: Incomplete
    patch_merger: Incomplete
    linear_1: Incomplete
    gelu: Incomplete
    linear_2: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x: mx.array, image_sizes: mx.array) -> mx.array: ...

class Model(PixtralModel):
    config: Incomplete
    multi_modal_projector: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None, **kwargs): ...
    @property
    def layers(self): ...
