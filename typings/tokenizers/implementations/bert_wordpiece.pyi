from .base_tokenizer import BaseTokenizer as BaseTokenizer
from tokenizers import AddedToken as AddedToken, Tokenizer as Tokenizer, decoders as decoders, trainers as trainers
from tokenizers.models import WordPiece as WordPiece
from tokenizers.normalizers import BertNormalizer as BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer as BertPreTokenizer
from tokenizers.processors import BertProcessing as BertProcessing
from typing import Iterator

class BertWordPieceTokenizer(BaseTokenizer):
    def __init__(self, vocab: str | dict[str, int] | None = None, unk_token: str | AddedToken = '[UNK]', sep_token: str | AddedToken = '[SEP]', cls_token: str | AddedToken = '[CLS]', pad_token: str | AddedToken = '[PAD]', mask_token: str | AddedToken = '[MASK]', clean_text: bool = True, handle_chinese_chars: bool = True, strip_accents: bool | None = None, lowercase: bool = True, wordpieces_prefix: str = '##') -> None: ...
    @staticmethod
    def from_file(vocab: str, **kwargs): ...
    def train(self, files: str | list[str], vocab_size: int = 30000, min_frequency: int = 2, limit_alphabet: int = 1000, initial_alphabet: list[str] = [], special_tokens: list[str | AddedToken] = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'], show_progress: bool = True, wordpieces_prefix: str = '##'): ...
    def train_from_iterator(self, iterator: Iterator[str] | Iterator[Iterator[str]], vocab_size: int = 30000, min_frequency: int = 2, limit_alphabet: int = 1000, initial_alphabet: list[str] = [], special_tokens: list[str | AddedToken] = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'], show_progress: bool = True, wordpieces_prefix: str = '##', length: int | None = None): ...
