import mlx.core as mx
import mlx.nn as nn
from .audio import AudioModel as AudioModel
from .config import ModelConfig as ModelConfig, TextConfig as TextConfig
from .language import Gemma3nRMSNorm as Gemma3nRMSNorm, LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete

def masked_scatter(input_tensor, mask, source): ...

class Gemma3nMultimodalEmbedder(nn.Module):
    multimodal_hidden_size: Incomplete
    eps: Incomplete
    vocab_offset: Incomplete
    vocab_size: Incomplete
    text_hidden_size: Incomplete
    embedding: Incomplete
    hard_embedding_norm: Incomplete
    soft_embedding_norm: Incomplete
    embedding_projection: Incomplete
    embedding_post_projection_norm: Incomplete
    def __init__(self, multimodal_config: ModelConfig, text_config: TextConfig) -> None: ...
    def __call__(self, input_ids: mx.array = None, inputs_embeds: mx.array = None) -> mx.array: ...

class Model(nn.Module):
    model_type: Incomplete
    config: Incomplete
    language_model: Incomplete
    vocab_size: Incomplete
    vocab_size_per_layer_input: Incomplete
    vision_tower: Incomplete
    embed_vision: Incomplete
    audio_tower: Incomplete
    embed_audio: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None, input_features: mx.array | None = None, input_features_mask: mx.array | None = None, **kwargs): ...
    def get_audio_features(self, input_features, input_features_mask): ...
    @staticmethod
    def get_image_features(pixel_values, vision_tower, config, embed_vision): ...
    def construct_special_modality_mask(self, input_ids, inputs_embeds, token_id, modality: str = 'image'): ...
    @staticmethod
    def merge_multimodal_and_text(inputs_embeds, features, special_modality_mask, modality: str = 'image'): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array | None = None, cache: mx.array | None = None, **kwargs): ...
    def sanitize(self, weights): ...
    @property
    def layers(self): ...
