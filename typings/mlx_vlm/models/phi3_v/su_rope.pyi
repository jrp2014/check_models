from _typeshed import Incomplete

class Phi3SuScaledRotaryEmbedding:
    inv_freq_short: Incomplete
    inv_freq_long: Incomplete
    original_max_position_embeddings: Incomplete
    scaling_factor: Incomplete
    def __init__(self, dims: int, traditional: bool = False, base: float = 10000.0, scale: float = 1.0, max_position_embeddings: int = 131072, original_max_position_embeddings: int = 4096, short_factor: list[float] | float = 1.0, long_factor: list[float] | float = 1.0) -> None: ...
    def __call__(self, x, offset: int = 0): ...
