from .generate import DEFAULT_MAX_TOKENS as DEFAULT_MAX_TOKENS, DEFAULT_MODEL_PATH as DEFAULT_MODEL_PATH, DEFAULT_PROMPT as DEFAULT_PROMPT, DEFAULT_SEED as DEFAULT_SEED, DEFAULT_TEMPERATURE as DEFAULT_TEMPERATURE, DEFAULT_TOP_P as DEFAULT_TOP_P, generate as generate, stream_generate as stream_generate
from .prompt_utils import apply_chat_template as apply_chat_template
from .utils import load as load
from .version import __version__ as __version__
from _typeshed import Incomplete
from fastapi import Request as Request
from pydantic import BaseModel
from typing import Any, Literal
from typing_extensions import Required, TypeAlias, TypedDict

app: Incomplete
MAX_IMAGES: int
model_cache: Incomplete

def load_model_resources(model_path: str, adapter_path: str | None): ...
def get_cached_model(model_path: str, adapter_path: str | None = None): ...
def unload_model_sync(): ...

class ResponseInputTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[Literal['input_text']]

class ResponseInputImageParam(TypedDict, total=False):
    detail: Literal['high', 'low', 'auto']
    type: Required[Literal['input_image']]
    image_url: Required[str]
    file_id: str | None
ResponseInputContentParam: TypeAlias = ResponseInputTextParam | ResponseInputImageParam
ResponseInputMessageContentListParam: TypeAlias = list[ResponseInputContentParam]

class ResponseOutputText(TypedDict, total=False):
    text: Required[str]
    type: Required[Literal['output_text']]
ResponseOutputMessageContentList: TypeAlias = list[ResponseOutputText]

class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system', 'developer']
    content: str | ResponseInputMessageContentListParam | ResponseOutputMessageContentList

class OpenAIRequest(BaseModel):
    input: str | list[ChatMessage]
    model: str
    max_output_tokens: int
    temperature: float
    top_p: float
    stream: bool

class OpenAIUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int

class OpenAIErrorObject(BaseModel):
    code: str | None
    message: str | None
    param: str | None
    type: str | None

class OpenAIResponse(BaseModel):
    id: str
    object: Literal['response']
    created_at: int
    status: Literal['completed', 'failed', 'in_progress', 'incomplete']
    error: OpenAIErrorObject | None
    instructions: str | None
    max_output_tokens: int | None
    model: str
    output: list[ChatMessage | Any]
    output_text: str | None
    temperature: float | None
    top_p: float | None
    truncation: Literal['auto', 'disabled'] | str
    usage: OpenAIUsage
    user: str | None

class BaseStreamEvent(BaseModel):
    type: str

class ContentPartOutputText(BaseModel):
    type: Literal['output_text']
    text: str
    annotations: list[str]

class MessageItem(BaseModel):
    id: str
    type: Literal['message']
    status: Literal['in_progress', 'completed']
    role: str
    content: list[ContentPartOutputText]

class ResponseCreatedEvent(BaseStreamEvent):
    type: Literal['response.created']
    response: OpenAIResponse

class ResponseInProgressEvent(BaseStreamEvent):
    type: Literal['response.in_progress']
    response: OpenAIResponse

class ResponseOutputItemAddedEvent(BaseStreamEvent):
    type: Literal['response.output_item.added']
    output_index: int
    item: MessageItem

class ResponseContentPartAddedEvent(BaseStreamEvent):
    type: Literal['response.content_part.added']
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText

class ResponseOutputTextDeltaEvent(BaseStreamEvent):
    type: Literal['response.output_text.delta']
    item_id: str
    output_index: int
    content_index: int
    delta: str

class ResponseOutputTextDoneEvent(BaseStreamEvent):
    type: Literal['response.output_text.done']
    item_id: str
    output_index: int
    content_index: int
    text: str

class ResponseContentPartDoneEvent(BaseStreamEvent):
    type: Literal['response.content_part.done']
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText

class ResponseOutputItemDoneEvent(BaseStreamEvent):
    type: Literal['response.output_item.done']
    output_index: int
    item: MessageItem

class ResponseCompletedEvent(BaseStreamEvent):
    type: Literal['response.completed']
    response: OpenAIResponse
StreamEvent = ResponseCreatedEvent | ResponseInProgressEvent | ResponseOutputItemAddedEvent | ResponseContentPartAddedEvent | ResponseOutputTextDeltaEvent | ResponseOutputTextDoneEvent | ResponseContentPartDoneEvent | ResponseOutputItemDoneEvent | ResponseCompletedEvent

async def openai_endpoint(request: Request): ...

class VLMRequest(BaseModel):
    model: str
    adapter_path: str | None
    image: list[str]
    audio: list[str]
    prompt: str
    system: str | None
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    resize_shape: tuple[int, int] | None

class GenerationRequest(VLMRequest):
    stream: bool

class ChatRequest(GenerationRequest):
    prompt: list[ChatMessage]

class UsageStats(OpenAIUsage):
    prompt_tps: float
    generation_tps: float
    peak_memory: float

class GenerationResponse(BaseModel):
    text: str
    model: str
    usage: UsageStats | None

class StreamChunk(BaseModel):
    chunk: str
    model: str
    usage: UsageStats | None

async def health_check(): ...
async def unload_model_endpoint(): ...
def main() -> None: ...
