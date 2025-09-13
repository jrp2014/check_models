import mlx.core as mx
import mlx.nn as nn
from ..base import BaseImageProcessor as BaseImageProcessor
from .config import ModelConfig as ModelConfig, VisionConfig as VisionConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path
from transformers import AutoConfig as AutoConfig

class ImageProcessor(BaseImageProcessor):
    def preprocess(self, images): ...

class LlavaMultiModalProjector(nn.Module):
    linear_1: Incomplete
    gelu: Incomplete
    linear_2: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class SigLipVisionTower(nn.Module):
    vision_tower: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array, output_hidden_states: bool | None = None) -> mx.array: ...

class Model(nn.Module):
    model_type: Incomplete
    config: Incomplete
    vision_tower: Incomplete
    language_model: Incomplete
    mm_projector: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None): ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array | None = None, cache: tuple[mx.array, mx.array] | None = None, **kwargs): ...
    def sanitize(self, weights): ...
