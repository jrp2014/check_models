import mlx.core as mx
import mlx.nn as nn
from .config import ModelConfig as ModelConfig, TextConfig as TextConfig, VisionConfig as VisionConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path

class Model(nn.Module):
    config: Incomplete
    vision_tower: Incomplete
    language_model: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None, grid_thw: mx.array | None = None): ...
    @staticmethod
    def merge_input_ids_with_image_features(image_token_id, video_token_id, image_features, inputs_embeds, input_ids): ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array | None = None, mask: mx.array | None = None, cache=None, **kwargs): ...
    def sanitize(self, weights): ...
