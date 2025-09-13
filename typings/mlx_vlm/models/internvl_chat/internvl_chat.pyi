import mlx.core as mx
import mlx.nn as nn
from ..base import BaseModelConfig as BaseModelConfig, pixel_shuffle as pixel_shuffle
from .config import ModelConfig as ModelConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path

class Model(nn.Module):
    config: Incomplete
    vision_model: Incomplete
    language_model: Incomplete
    downsample_ratio: Incomplete
    mlp1: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None): ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array, cache=None, **kwargs): ...
