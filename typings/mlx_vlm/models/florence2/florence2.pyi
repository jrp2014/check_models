import mlx.core as mx
import mlx.nn as nn
from .config import ModelConfig as ModelConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path

def shift_tokens_right(input_ids: mx.array, pad_token_id: int, decoder_start_token_id: int) -> mx.array: ...

class LearnedPositionEmbedding2D(nn.Module):
    row_embeddings: Incomplete
    column_embeddings: Incomplete
    def __init__(self, embedding_dim: int = 256, num_pos: int = 50) -> None: ...
    def __call__(self, x): ...

class PositionalEmbeddingCosine1D(nn.Module):
    embed_dim: Incomplete
    max_seq_len: Incomplete
    pos_idx_to_embed: Incomplete
    def __init__(self, embed_dim: int = 512, max_seq_len: int = 1024) -> None: ...
    def __call__(self, seq_embeds: mx.array) -> mx.array: ...

class Model(nn.Module):
    config: Incomplete
    vision_tower: Incomplete
    language_model: Incomplete
    image_projection: Incomplete
    image_proj_norm: Incomplete
    image_pos_embed: Incomplete
    visual_temporal_embed: Incomplete
    image_feature_source: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    @property
    def layers(self): ...
    def __call__(self, input_ids=None, pixel_values=None, cache=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None, **kwargs): ...
    @staticmethod
    def sanitize(weights): ...
