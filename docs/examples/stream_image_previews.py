from collections.abc import AsyncGenerator
from contextlib import aclosing

from agents import RunResultStreaming

from chatkit.agents import AgentContext, ResponseStreamConverter, stream_agent_response
from chatkit.types import ThreadStreamEvent


async def stream_image_previews[TContext](
    agent_context: AgentContext[TContext],
    result: RunResultStreaming,
    converter: ResponseStreamConverter,
) -> AsyncGenerator[ThreadStreamEvent, None]:
    async with aclosing(
        stream_agent_response(agent_context, result, converter=converter)
    ) as events:
        async for event in events:
            yield event
