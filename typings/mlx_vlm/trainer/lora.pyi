import mlx.nn as nn
from _typeshed import Incomplete

class LoRaLayer(nn.Module):
    original_layer: Incomplete
    dropout: Incomplete
    A: Incomplete
    B: Incomplete
    alpha: Incomplete
    def __init__(self, linear: nn.Linear | nn.QuantizedLinear, rank: int, alpha: float = 0.1, dropout: float = 0.0) -> None: ...
    def __call__(self, x): ...

def replace_lora_with_linear(model) -> None: ...
