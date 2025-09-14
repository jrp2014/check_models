import mlx.nn as nn
from ..base import LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from ..cache import SimpleKVCache as SimpleKVCache
from .config import TextConfig as TextConfig
from _typeshed import Incomplete

class Florence2Attention(nn.Module):
    embed_dim: Incomplete
    num_heads: Incomplete
    is_decoder: Incomplete
    is_causal: Incomplete
    head_dim: Incomplete
    scaling: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    q_proj: Incomplete
    out_proj: Incomplete
    def __init__(self, config: TextConfig, is_decoder: bool = False, is_causal: bool = False) -> None: ...
    def __call__(self, hidden_states, key_value_states=None, cache: SimpleKVCache | None = None, attention_mask=None, layer_head_mask=None): ...

class Florence2EncoderLayer(nn.Module):
    embed_dim: Incomplete
    self_attn: Incomplete
    self_attn_layer_norm: Incomplete
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    final_layer_norm: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, hidden_states, attention_mask=None): ...

class Florence2DecoderLayer(nn.Module):
    embed_dim: Incomplete
    self_attn: Incomplete
    dropout: Incomplete
    activation_fn: Incomplete
    activation_dropout: Incomplete
    self_attn_layer_norm: Incomplete
    encoder_attn: Incomplete
    encoder_attn_layer_norm: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    final_layer_norm: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, hidden_states, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None, cache: tuple[SimpleKVCache, SimpleKVCache] | None = None): ...

class Florence2Encoder(nn.Module):
    config: Incomplete
    dropout: Incomplete
    layerdrop: Incomplete
    embed_scale: Incomplete
    offset: int
    embed_positions: Incomplete
    layers: Incomplete
    layernorm_embedding: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, input_ids=None, inputs_embeds=None, attention_mask=None): ...

class Florence2Decoder(nn.Module):
    config: Incomplete
    dropout: Incomplete
    layerdrop: Incomplete
    padding_idx: Incomplete
    max_target_positions: Incomplete
    embed_scale: Incomplete
    offset: int
    embed_positions: Incomplete
    layers: Incomplete
    layernorm_embedding: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, head_mask=None, cross_attn_head_mask=None, inputs_embeds=None, cache=None): ...

class Florence2LanguageModel(nn.Module):
    config: Incomplete
    shared: Incomplete
    encoder: Incomplete
    decoder: Incomplete
    embed_scale: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, input_ids=None, inputs_embeds=None, decoder_input_ids=None, decoder_inputs_embeds=None, attention_mask=None, decoder_attention_mask=None, encoder_outputs=None, cache=None): ...

class LanguageModel(nn.Module):
    config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, input_ids=None, inputs_embeds=None, decoder_input_ids=None, decoder_inputs_embeds=None, attention_mask=None, decoder_attention_mask=None, encoder_outputs=None, cache=None): ...
    @property
    def layers(self): ...
    @property
    def head_dim(self): ...
    @property
    def n_kv_heads(self): ...
    def make_cache(self): ...
