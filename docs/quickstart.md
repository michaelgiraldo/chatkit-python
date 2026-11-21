# Quick start

Run the self-hosted ChatKit demo from this checkout. It uses sample replies and
widgets and runs without an API key.

## Run your ChatKit server

Install `uv`, then run this command from the root of the checkout.

```sh
make demo
```

Open [the demo on port 8000](http://127.0.0.1:8000/). The page loads the
official ChatKit browser client from its CDN and sends requests to
`/chatkit` on the same local server. To use a different port, run
`make demo DEMO_PORT=8001` and open that port in the browser.

You can also start the demo with Python 3.14 and pip.

```sh
python -m pip install '.[fastapi]'
python -m uvicorn docs.examples.quickstart:app --host 127.0.0.1 --port 8000
```

## Try the browser flow

1. Send `hello`. The reply is **Hello, world!**
2. Send `widget`, then click **Save demo state**. The card shows **Widget state saved.**
3. Send `slow`. Wait for the first text to appear, then click **Stop response**
   while the reply is still streaming.
4. Reload the page and reopen the thread from **Conversation history**. The
   saved widget and the partial reply are still there. Send another message to
   verify that the thread accepts a follow-up.

## How the demo works

The [runnable backend](examples/quickstart.py) receives every ChatKit
request at `/chatkit`. Its `ChatKitServer.respond(turn)` yields events for
a `Turn` with the thread, trigger, and request context. The endpoint calls
`chatkit.fastapi.process_request(server, request, context={})`. The adapter
closes streams on disconnect and returns JSON errors for invalid requests,
missing resources, and conflicting thread state.
[FastAPI adapter reference](api/chatkit/fastapi.md)

## Store chat data

The backend uses an [in-memory ThreadStore](examples/in_memory_thread_store.py)
for threads and their items. Data lasts for the current process. For production,
use a durable store that scopes each operation to the authenticated request
context. Configure [attachment storage](guides/accept-rich-user-input.md#configure-an-upload-strategy)
when your app supports files.

## Verify your backend

With the demo server running, send a test message directly to the backend.

```sh
curl --no-buffer --fail-with-body http://127.0.0.1:8000/chatkit \
  --header 'Content-Type: application/json' \
  --data '{
    "type": "threads.create",
    "params": {
      "input": {
        "content": [{"type": "input_text", "text": "Hi there!"}],
        "attachments": [],
        "inference_options": {}
      }
    }
  }'
```

Look for a `thread.item.done` event with `item.type` set to `assistant_message`
and the text `Hello, world!`. The hardcoded responder runs without an API key.

## Generate model responses

Install the Agents extra alongside FastAPI to use the model integration.

```sh
python -m pip install '.[fastapi,agents]'
```

For a text-only model-backed demo, replace the example's `respond` method
with the implementation below. Set `OPENAI_API_KEY` in your environment and
choose a model your project can use.

```python
from collections.abc import AsyncIterator
from contextlib import aclosing

from agents import Agent, Runner

from chatkit.agent_input import simple_to_agent_input
from chatkit.agents import AgentContext, stream_agent_response
from chatkit.events import ThreadStreamEvent
from chatkit.server import ChatKitServer
from chatkit.turns import Turn

assistant = Agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    model="gpt-6-luna",
)


class MyChatKitServer(ChatKitServer[dict[str, object]]):
    async def respond(
        self, turn: Turn[dict[str, object]]
    ) -> AsyncIterator[ThreadStreamEvent]:
        # Read the newest page and pass it to the model in chronological order.
        items_page = await self.store.load_thread_items(
            turn.thread.id,
            after=None,
            limit=20,
            order="desc",
            context=turn.context,
        )
        input_items = await simple_to_agent_input(list(reversed(items_page.data)))

        agent_context = AgentContext(
            thread=turn.thread,
            store=self.store,
            request_context=turn.context,
            resume_from=None,
        )
        result = Runner.run_streamed(assistant, input_items, context=agent_context)
        async with aclosing(stream_agent_response(agent_context, result)) as events:
            async for event in events:
                yield event
```

The `simple_to_agent_input` helper converts the stored items into model input.
The stream helper converts agent events into ChatKit events and closes the run
when the client stops the response. See [Respond to a user message](guides/respond-to-user-message.md)
for persistent model state and tool continuations.
