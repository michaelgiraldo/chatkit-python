# Quick start

To get a basic ChatKit app running—a React chat UI talking to a Python server—clone and run the starter app:


```sh
git clone https://github.com/openai/openai-chatkit-starter-app.git
cd openai-chatkit-starter-app/chatkit
npm install
npm run dev
```

The sections below explain the core components and steps behind the starter app.

## Render chat UI

!!! note "Quick start uses React"
    This section shows the React integration using `@openai/chatkit-react`.  
    If you’re not using React, you can render ChatKit directly with vanilla JavaScript using `@openai/chatkit`. See the [ChatKit.js quick start](https://openai.github.io/chatkit-js/quickstart/) for details.

Install the React bindings:

```sh
npm install @openai/chatkit-react
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

The chat UI will render, but sending messages will fail until you start the server below and provide a store for threads and messages.

## Run your ChatKit server

This fork requires Python 3.14 or newer. From the root of this fork's checkout,
install the package and FastAPI for the example server. Then expose a single
`/chatkit` endpoint that forwards requests to a `ChatKitServer` instance.

```sh
python -m pip install . fastapi
```

Save [chatkit_response.py](examples/chatkit_response.py) next to `main.py`. The
helper closes the ChatKit stream when the HTTP response ends, so ChatKit can
save a partial reply if the client disconnects.

Create `main.py` with a minimal server that replies “Hello, world!” You can
add a model call in [Respond to a user message](guides/respond-to-user-message.md).

```python
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import Response

from chatkit.server import ChatKitServer, StreamingResult
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
)
from chatkit_response import ChatKitStreamingResponse

app = FastAPI()


class MyChatKitServer(ChatKitServer[dict]):
    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: dict,
    ) -> AsyncIterator[ThreadStreamEvent]:
        # Streams a fixed "Hello, world!" assistant message
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                thread_id=thread.id,
                id=self.store.generate_item_id("message", thread, context),
                created_at=datetime.now(),
                content=[AssistantMessageContent(text="Hello, world!")],
            ),
        )


# Place the MyChatKitStore definition from the next section above this line.
server = MyChatKitServer(store=MyChatKitStore())


@app.post("/chatkit")
async def chatkit(request: Request):
    result = await server.process(await request.body(), context={})
    if isinstance(result, StreamingResult):
        return ChatKitStreamingResponse(result)
    return Response(content=result.json, media_type="application/json")
```

All ChatKit requests go to this single endpoint. Set `api.url` on the React side to match (`/chatkit` here), and `ChatKitServer` routes each request internally.


## Store chat data

ChatKit servers require a store to load and save threads, messages, and other items.

For this quickstart, use a small in-memory store so conversations persist while the process is running, without introducing a database. This keeps the example minimal while still matching real ChatKit behavior.


```python
from collections import defaultdict
from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata


class MyChatKitStore(Store[dict]):
    def __init__(self):
        self.threads: dict[str, ThreadMetadata] = {}
        self.items: dict[str, list[ThreadItem]] = defaultdict(list)

    async def load_thread(self, thread_id: str, context: dict) -> ThreadMetadata:
        if thread_id not in self.threads:
            raise NotFoundError(f"Thread {thread_id} not found")
        return self.threads[thread_id]

    async def save_thread(self, thread: ThreadMetadata, context: dict) -> None:
        self.threads[thread.id] = thread

    async def load_threads(
        self, limit: int, after: str | None, order: str, context: dict
    ) -> Page[ThreadMetadata]:
        threads = list(self.threads.values())
        return self._paginate(
            threads,
            after,
            limit,
            order,
            sort_key=lambda t: t.created_at,
            cursor_key=lambda t: t.id,
        )

    async def load_thread_items(
        self, thread_id: str, after: str | None, limit: int, order: str, context: dict
    ) -> Page[ThreadItem]:
        items = self.items.get(thread_id, [])
        return self._paginate(
            items,
            after,
            limit,
            order,
            sort_key=lambda i: i.created_at,
            cursor_key=lambda i: i.id,
        )

    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: dict
    ) -> None:
        self.items[thread_id].append(item)

    async def save_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
        items = self.items[thread_id]
        for idx, existing in enumerate(items):
            if existing.id == item.id:
                items[idx] = item
                return
        items.append(item)

    async def load_item(
        self, thread_id: str, item_id: str, context: dict
    ) -> ThreadItem:
        for item in self.items.get(thread_id, []):
            if item.id == item_id:
                return item
        raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")

    async def delete_thread(self, thread_id: str, context: dict) -> None:
        self.threads.pop(thread_id, None)
        self.items.pop(thread_id, None)

    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: dict
    ) -> None:
        self.items[thread_id] = [
            item for item in self.items.get(thread_id, []) if item.id != item_id
        ]

    def _paginate(
        self,
        rows: list,
        after: str | None,
        limit: int,
        order: str,
        sort_key,
        cursor_key,
    ):
        sorted_rows = sorted(rows, key=sort_key, reverse=order == "desc")
        start = 0
        if after:
            for idx, row in enumerate(sorted_rows):
                if cursor_key(row) == after:
                    start = idx + 1
                    break
        data = sorted_rows[start : start + limit]
        has_more = start + limit < len(sorted_rows)
        next_after = cursor_key(data[-1]) if has_more and data else None
        return Page(data=data, has_more=has_more, after=next_after)

    # Attachments are intentionally not implemented for the quickstart

    async def save_attachment(self, attachment: Attachment, context: dict) -> None:
        raise NotImplementedError()

    async def load_attachment(self, attachment_id: str, context: dict) -> Attachment:
        raise NotImplementedError()

    async def delete_attachment(self, attachment_id: str, context: dict) -> None:
        raise NotImplementedError()
```

This store implements only the methods required for basic chat while the server is running; persistence across restarts and attachments are intentionally omitted.

For production, replace this with a database-backed store (for example, Postgres or MySQL) so threads and items persist across restarts.

## Verify your backend

With the server and store in `main.py`, start the example server from that directory.

```sh
python -m uvicorn main:app --port 8000
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

Replace the hardcoded "Hello, World!" reply from [Run your ChatKit server](#run-your-chatkit-server) with an Agents SDK call to generate real responses. Set `OPENAI_API_KEY` in your environment before running.

Use ChatKit's Agents SDK helpers to simplify request conversion and streaming. The `simple_to_agent_input` helper translates ChatKit thread items to agent input items, and `stream_agent_response` turns the streamed run into ChatKit events:


```python
from contextlib import aclosing

from agents import Agent, Runner
from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response

assistant = Agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4.1-mini",
)


class MyChatKitServer(ChatKitServer[dict]):
    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: dict,
    ) -> AsyncIterator[ThreadStreamEvent]:
        # Convert recent thread items (which includes the user message) to model input
        items_page = await self.store.load_thread_items(
            thread.id,
            after=None,
            limit=20,
            order="asc",
            context=context,
        )
        input_items = await simple_to_agent_input(items_page.data)

        # Stream the run through ChatKit events
        agent_context = AgentContext(
            thread=thread, store=self.store, request_context=context
        )
        result = Runner.run_streamed(assistant, input_items, context=agent_context)
        async with aclosing(stream_agent_response(agent_context, result)) as events:
            async for event in events:
                yield event
```
