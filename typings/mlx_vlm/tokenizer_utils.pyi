from typing import Any, Protocol

class TokenizerProtocol(Protocol):
    def decode(
        self,
        token_ids: list[int],
        **kwargs: dict[str, Any],
    ) -> str: ...

REPLACEMENT_CHAR: str = ...

class StreamingDetokenizer:
    __slots__ = ["_segment", "text", "tokens"]

    def reset(self) -> None: ...

    def add_token(
        self,
        token: int,
        skip_special_token_ids: list[int] = ...,
    ) -> None: ...

    def finalize(self) -> None: ...

    @property
    def last_segment(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def tokens(self) -> list[int]: ...

class NaiveStreamingDetokenizer(StreamingDetokenizer):
    def __init__(self, tokenizer: TokenizerProtocol) -> None: ...

    def reset(self) -> None: ...

    def add_token(
        self,
        token: int,
        skip_special_token_ids: list[int] = ...,
    ) -> None: ...

    def finalize(self) -> None: ...

class SPMStreamingDetokenizer(StreamingDetokenizer):
    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        trim_space: bool = ...,
    ) -> None: ...

    def reset(self) -> None: ...

    def add_token(
        self,
        token: int,
        skip_special_token_ids: list[int] = ...,
    ) -> None: ...

    def finalize(self) -> None: ...

class BPEStreamingDetokenizer(StreamingDetokenizer):
    _byte_decoder: dict[int, bytes]

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        trim_space: bool = ...,
    ) -> None: ...

    def reset(self) -> None: ...

    def add_token(
        self,
        token: int,
        skip_special_token_ids: list[int] = ...,
    ) -> None: ...

    def finalize(self) -> None: ...

    @classmethod
    def make_byte_decoder(cls) -> dict[int, bytes]: ...

class TokenizerWrapper:
    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        detokenizer_class: type[StreamingDetokenizer] = ...,
    ) -> None: ...

    def __getattr__(self, attr: str) -> TokenizerProtocol: ...

def load_tokenizer(
    model_path: str,
    return_tokenizer: bool = ...,
    tokenizer_config_extra: dict[str, Any] = ...,
) -> TokenizerWrapper | StreamingDetokenizer: ...

