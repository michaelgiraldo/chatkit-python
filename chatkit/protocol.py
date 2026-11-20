"""Parse ChatKit requests and encode responses at a transport boundary."""

from pydantic import BaseModel, TypeAdapter, ValidationError

from .errors import InvalidRequestError
from .types import ChatKitReq, ThreadStreamEvent

_request_adapter = TypeAdapter[ChatKitReq](ChatKitReq)


def parse_request(body: str | bytes | bytearray) -> ChatKitReq:
    """Decode JSON into a validated ChatKit request."""
    try:
        return _request_adapter.validate_json(body)
    except ValidationError as error:
        raise InvalidRequestError(error) from error


def encode_message(message: BaseModel) -> bytes:
    """Encode a response model in ChatKit's JSON wire format."""
    return message.model_dump_json(
        by_alias=True,
        exclude_none=True,
        context={"exclude_metadata": True},
    ).encode("utf-8")


def encode_sse(event: ThreadStreamEvent) -> bytes:
    """Encode one ChatKit event as a Server-Sent Event frame."""
    return b"data: " + encode_message(event) + b"\n\n"
