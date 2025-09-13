import mlx.core as mx
import mlx.nn as nn
from .config import ModelConfig as ModelConfig, TextConfig as TextConfig, VisionConfig as VisionConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path
from transformers import AutoConfig as AutoConfig

class KimiVLMultiModalProjector(nn.Module):
    hidden_size: Incomplete
    pre_norm: Incomplete
    linear_1: Incomplete
    act: Incomplete
    linear_2: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, image_features: list[mx.array]) -> mx.array: ...

class Model(nn.Module):
    model_type: Incomplete
    config: Incomplete
    vision_tower: Incomplete
    language_model: Incomplete
    multi_modal_projector: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None, grid_thw: mx.array | None = None): ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, cache=None, **kwargs): ...
    def sanitize(self, weights): ...
