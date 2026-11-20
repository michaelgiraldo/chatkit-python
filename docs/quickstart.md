# Quick start

Run the self-hosted Python backend from this checkout. It streams a fixed
reply and lets you verify ChatKit before adding a model or database.

## Render chat UI

!!! note "Quick start uses React"
    This section shows the React integration using `@openai/chatkit-react`.  
    If you’re not using React, you can render ChatKit directly with vanilla JavaScript using `@openai/chatkit`. See the [ChatKit.js quick start](https://openai.github.io/chatkit-js/quickstart/) for details.

Install the React bindings used in this example. This version includes
`@openai/chatkit` 1.9.0.

```sh
npm install @openai/chatkit-react@1.6.1
```

In your index.html, load ChatKit.js:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <script src="https://cdn.platform.openai.com/deployments/chatkit/chatkit.js"></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
```

Wire up a minimal React app. Point `api.url` at your ChatKit server endpoint and pass the domain key you configured there.

```tsx
import {ChatKit, useChatKit} from "@openai/chatkit-react";

export function App() {
  const chatkit = useChatKit({
    api: {
      url: "http://localhost:8000/chatkit",
      domainKey: "local-dev", // domain keys are optional in dev
    },
  });

  return <ChatKit control={chatkit.control} />;
}
```

Start the example server below to send messages from the UI.

## Run your ChatKit server

This fork requires Python 3.14 or newer. From the root of this fork's checkout,
install the package and FastAPI for the example server. Then expose a single
`/chatkit` endpoint that forwards requests to a `ChatKitServer` instance.

```sh
python -m pip install '.[fastapi]'
```

The [runnable backend](examples/quickstart.py) receives every ChatKit
request at `/chatkit`. Its `ChatKitServer.respond(turn)` yields events for
a `Turn` with the thread, trigger, and request context. The endpoint calls
`chatkit.fastapi.process_request(server, request, context={})`. The adapter
closes streams on disconnect and returns JSON errors for invalid requests.
[FastAPI adapter reference](api/chatkit/fastapi.md)

## Store chat data

The backend uses an [in-memory ThreadStore](examples/in_memory_thread_store.py)
for threads and their items. Data lasts for the current process. For production,
use a durable store that scopes each operation to the authenticated request
context. Configure [attachment storage](guides/accept-rich-user-input.md#configure-an-upload-strategy)
when your app supports files.

## Verify your backend

Start the example from the root of the checkout.

```sh
python -m uvicorn docs.examples.quickstart:app --port 8000
```

In another terminal, send a test message directly to the backend.

```sh
curl --no-buffer --fail-with-body http://localhost:8000/chatkit \
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

Replace the fixed reply in the example with an Agents SDK call. Set
`OPENAI_API_KEY` in your environment before running.

Use ChatKit's Agents SDK helpers to simplify request conversion and streaming. The `simple_to_agent_input` helper translates ChatKit thread items to agent input items, and `stream_agent_response` turns the streamed run into ChatKit events:


```python
from collections.abc import AsyncIterator
from contextlib import aclosing

from agents import Agent, Runner
from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.turns import Turn
from chatkit.types import ThreadStreamEvent

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

        # Stream the run through ChatKit events
        agent_context = AgentContext(
            thread=turn.thread, store=self.store, request_context=turn.context
        )
        result = Runner.run_streamed(assistant, input_items, context=agent_context)
        async with aclosing(stream_agent_response(agent_context, result)) as events:
            async for event in events:
                yield event
```
