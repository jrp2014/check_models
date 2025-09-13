import mlx.core as mx
import mlx.nn as nn
from _typeshed import Incomplete
from dataclasses import dataclass, field
from pathlib import Path

def get_prompt(model_type, processor, conversation): ...

class Dataset:
    dataset: Incomplete
    processor: Incomplete
    config: Incomplete
    image_processor: Incomplete
    image_resize_shape: Incomplete
    def __init__(self, hf_dataset, config, processor, image_processor=None, take=None, split=None, image_resize_shape=None) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx): ...

def grad_checkpoint(layer): ...

@dataclass
class TrainingArgs:
    batch_size: int = field(default=4, metadata={'help': 'Minibatch size.'})
    iters: int = field(default=100, metadata={'help': 'Iterations to train for.'})
    val_batches: int = field(default=25, metadata={'help': 'Number of validation batches, -1 uses the entire validation set.'})
    steps_per_report: int = field(default=10, metadata={'help': 'Number of training steps between loss reporting.'})
    steps_per_eval: int = field(default=200, metadata={'help': 'Number of training steps between validations.'})
    steps_per_save: int = field(default=100, metadata={'help': 'Save the model every number steps'})
    max_seq_length: int = field(default=2048, metadata={'help': 'Maximum sequence length.'})
    adapter_file: str = field(default='adapters.safetensors', metadata={'help': 'Save/load path for the trained adapter weights.'})
    grad_checkpoint: bool = field(default=False, metadata={'help': 'Use gradient checkpointing to reduce memory use.'})

def default_loss(model, inputs, targets, lengths): ...

class Trainer:
    model: Incomplete
    optimizer: Incomplete
    train_on_completions: Incomplete
    assistant_id: Incomplete
    clip_gradients: Incomplete
    def __init__(self, model, optimizer, train_on_completions: bool = False, assistant_id: int = 77091, clip_gradients=None) -> None: ...
    def loss_fn(self, model, batch): ...
    def train_step(self, batch): ...
    @mx.compile
    def train_epoch(self, dataloader): ...

def save_adapter(model: nn.Module, adapter_file: str | Path): ...
