from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx import core as mx
from mlx import nn

def get_prompt(model_type: str, processor: Any, conversation: dict[str, Any]) -> str:
    ...

class Dataset:
    def __init__(
        self,
        hf_dataset: Any,
        config: dict[str, Any],
        processor: Any,
        image_processor: Any = ...,
        take: int = ...,
        split: str = ...,
        image_resize_shape: tuple[int, int] = ...,
    ) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        ...
    


def grad_checkpoint(layer: nn.Module) -> None:
    ...

@dataclass
class TrainingArgs:
    batch_size: int = ...
    iters: int = ...
    val_batches: int = ...
    steps_per_report: int = ...
    steps_per_eval: int = ...
    steps_per_save: int = ...
    max_seq_length: int = ...
    adapter_file: str = ...
    grad_checkpoint: bool = ...


def default_loss(
    model: nn.Module,
    inputs: dict[str, Any],
    targets: Any,
    lengths: list[int],
) -> tuple[float, Any]:
    ...

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: mx.optimizers.Optimizer,
        train_on_completions: bool = ...,
        assistant_id: str | None = ...,
        clip_gradients: float | None = ...,
    ) -> None:
        ...

    def loss_fn(self, model: nn.Module, batch: dict[str, Any]) -> tuple[float, Any]:
        ...

    def train_step(self, batch: dict[str, Any]) -> float:
        ...

    @mx.compile
    def train_epoch(self, dataloader: Any) -> float:
        ...
    


def save_adapter(model: nn.Module, adapter_file: str | Path) -> None:
    ...

