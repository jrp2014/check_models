from _typeshed import Incomplete
from string import Template as Template
from tokenizers import Encoding as Encoding, Tokenizer as Tokenizer
from typing import Any, Callable, NamedTuple

dirname: Incomplete
css_filename: Incomplete
css: Incomplete

class Annotation:
    start: int
    end: int
    label: int
    def __init__(self, start: int, end: int, label: str) -> None: ...
AnnotationList = list[Annotation]
PartialIntList = list[int | None]

class CharStateKey(NamedTuple):
    token_ix: int | None
    anno_ix: int | None

class CharState:
    char_ix: int | None
    anno_ix: int | None
    tokens: list[int]
    def __init__(self, char_ix) -> None: ...
    @property
    def token_ix(self): ...
    @property
    def is_multitoken(self): ...
    def partition_key(self) -> CharStateKey: ...

class Aligned: ...

class EncodingVisualizer:
    unk_token_regex: Incomplete
    tokenizer: Incomplete
    default_to_notebook: Incomplete
    annotation_coverter: Incomplete
    def __init__(self, tokenizer: Tokenizer, default_to_notebook: bool = True, annotation_converter: Callable[[Any], Annotation] | None = None) -> None: ...
    def __call__(self, text: str, annotations: AnnotationList = [], default_to_notebook: bool | None = None) -> str | None: ...
    @staticmethod
    def calculate_label_colors(annotations: AnnotationList) -> dict[str, str]: ...
    @staticmethod
    def consecutive_chars_to_html(consecutive_chars_list: list[CharState], text: str, encoding: Encoding): ...

def HTMLBody(children: list[str], css_styles=...) -> str: ...
