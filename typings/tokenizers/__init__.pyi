from .implementations import BertWordPieceTokenizer as BertWordPieceTokenizer, ByteLevelBPETokenizer as ByteLevelBPETokenizer, CharBPETokenizer as CharBPETokenizer, SentencePieceBPETokenizer as SentencePieceBPETokenizer, SentencePieceUnigramTokenizer as SentencePieceUnigramTokenizer
from .tokenizers import AddedToken as AddedToken, Encoding as Encoding, NormalizedString as NormalizedString, PreTokenizedString as PreTokenizedString, Regex as Regex, Token as Token, Tokenizer as Tokenizer, __version__ as __version__, decoders as decoders, models as models, normalizers as normalizers, pre_tokenizers as pre_tokenizers, processors as processors, trainers as trainers
from enum import Enum

Offsets = tuple[int, int]
TextInputSequence = str
PreTokenizedInputSequence = list[str] | tuple[str]
TextEncodeInput = TextInputSequence | tuple[TextInputSequence, TextInputSequence] | list[TextInputSequence]
PreTokenizedEncodeInput = PreTokenizedInputSequence | tuple[PreTokenizedInputSequence, PreTokenizedInputSequence] | list[PreTokenizedInputSequence]
InputSequence = TextInputSequence | PreTokenizedInputSequence
EncodeInput = TextEncodeInput | PreTokenizedEncodeInput

class OffsetReferential(Enum):
    ORIGINAL = 'original'
    NORMALIZED = 'normalized'

class OffsetType(Enum):
    BYTE = 'byte'
    CHAR = 'char'

class SplitDelimiterBehavior(Enum):
    REMOVED = 'removed'
    ISOLATED = 'isolated'
    MERGED_WITH_PREVIOUS = 'merged_with_previous'
    MERGED_WITH_NEXT = 'merged_with_next'
    CONTIGUOUS = 'contiguous'
