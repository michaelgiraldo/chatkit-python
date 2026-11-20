# Let users pick tools and models

This guide shows how to expose a tool menu and model picker in the composer UI, read the user’s choices as inference options on the server, and fork your inference pipeline based on those choices.

At a high level:

- `composer.tools` controls which tools appear in the composer tool menu (the plus button).
- `composer.models` controls which models appear in the model picker in the composer below the text input.
- The selected tool and model arrive on the message in a `UserMessageTurn` or `RetryTurn`.

## Configure tools in the composer

Configure the tools that should appear in the composer tool menu when you initialize ChatKit on the client:

```ts
const chatkit = useChatKit({
  // ...
  composer: {
    tools: [
      {
        id: "summarize",
        icon: "book-open",
        label: "Summarize",
        placeholderOverride: "Summarize the current page or document.",
      },
      {
        id: "search_tickets",
        icon: "search",
        label: "Search tickets",
        shortLabel: "Search",
        placeholderOverride: "Search support tickets for similar issues.",
      },
    ],
  },
});
```

Each entry defines a user-facing label/shortLabel and a stable `id` you’ll use on the server to decide how to handle the turn.

## Configure the model picker in the composer

Expose a small set of model choices so users can trade off speed vs quality.

```ts
const chatkit = useChatKit({
  // ...
  composer: {
    models: [
      {
        id: "gpt-6-luna",
        label: "GPT-6 Luna",
        default: true,
      },
      {
        id: "gpt-6-sol",
        label: "GPT-6 Sol",
      },
    ],
  },
});
```

The selected model id flows through to your server so you can route requests to the right underlying model or configuration.

## Read tool and model choices on the server

On the server, `UserMessageItem.inference_options` carries the tool choice and model id for that turn.

```python
from collections.abc import AsyncIterator

from chatkit.errors import CustomStreamError
from chatkit.events import ThreadStreamEvent
from chatkit.models import InferenceOptions
from chatkit.server import ChatKitServer
from chatkit.turns import RetryTurn, Turn, UserMessageTurn


class MyChatKitServer(ChatKitServer[RequestContext]):
    async def respond(
        self, turn: Turn[RequestContext]
    ) -> AsyncIterator[ThreadStreamEvent]:
        options = InferenceOptions()
        if isinstance(turn.input, (UserMessageTurn, RetryTurn)):
            options = turn.input.message.inference_options

        model = options.model or "gpt-6-luna"
        if model not in {"gpt-6-luna", "gpt-6-sol"}:
            raise CustomStreamError("Select an available model.")

        tool_choice = options.tool_choice.id if options.tool_choice else None
        if tool_choice not in {None, "summarize", "search_tickets"}:
            raise CustomStreamError("Select an available tool.")

        async for event in self.stream_inference(turn, model, tool_choice):
            yield event
```

Implement `stream_inference` using your app's inference pipeline and yield
ChatKit events from it. The `InferenceOptions` object can leave `model` or
`tool_choice` unset. Apply defaults and validate any client-supplied choice
against the models and tools the authenticated user can use.

## Fork your inference pipeline based on user choices

Use the tool and model choices to branch into different agents, prompts, or tools.

### Route to different tools or agents

For example, use the composer’s tool id to decide which agent (or tool set) to run:

```python
if tool_choice == "summarize":
    agent = summarization_agent
elif tool_choice == "search_tickets":
    agent = ticket_search_agent
else:
    agent = default_agent

result = Runner.run_streamed(agent, input_items, context=agent_context)
```

You control which tools each agent exposes; the composer’s tool menu just lets the user express intent up front instead of relying purely on model heuristics.

### Choose models per turn

Use the selected model id to pick an underlying model or configuration when you call the OpenAI Responses API (or another provider):

```python
response = await client.responses.create(
    model=model,
    input=...,
    # other options
)
```

You can also use the model choice as a coarse “mode” flag—for example, always enabling safer or more verbose prompting on certain models.

### Combine tools and models

Nothing stops you from combining both choices. A common pattern is:

- Use the composer tool menu to decide **what kind of work** to do (summarization, search, drafting, etc.).
- Use the model picker to decide **how heavy** the model pass should be (fast vs quality, cheap vs expensive).

This keeps the chat UI simple while still giving advanced users control over how their requests are handled end to end.


