import mlx.core as mx
import mlx.nn as nn
from .config import ModelConfig as ModelConfig
from .language import LanguageModel as LanguageModel, RMSNorm as RMSNorm
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path

class Gemma3MultiModalProjector(nn.Module):
    mm_input_projection_weight: Incomplete
    mm_soft_emb_norm: Incomplete
    patches_per_image: Incomplete
    tokens_per_side: Incomplete
    kernel_size: Incomplete
    avg_pool: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

def masked_scatter(final_embedding: mx.array, image_mask_expanded: mx.array, scaled_image_features: mx.array): ...

class Model(nn.Module):
    model_type: Incomplete
    config: Incomplete
    vision_tower: Incomplete
    language_model: Incomplete
    multi_modal_projector: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None, mask: mx.array | None = None): ...
    @staticmethod
    def prepare_inputs_for_multimodal(hidden_size, pad_token_id, image_token_index, image_features, inputs_embeds, input_ids, attention_mask): ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array | None = None, cache: mx.array | None = None, **kwargs): ...
