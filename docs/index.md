# Turnspan for ChatKit

--8<-- "README.md:start-here"

## Documentation

- [Try the demo and connect a model](quickstart.md)
- [Core concepts](concepts/threads.md)
- [Respond to a user message](guides/respond-to-user-message.md)
- [Prepare your app for production](guides/prepare-your-app-for-production.md)
- [API reference](api/chatkit/index.md)
- [ChatKit JS docs](https://openai.github.io/chatkit-js/)
- [Changelog](release.md)
- [Upstream status](upstream-status.md)

## Project scope

Turnspan connects the ChatKit browser component to an application-owned
Python backend. It handles ChatKit requests, thread and widget events,
stream cleanup, and the lifecycle of browser tools. It defines the storage
contract for those operations, and each application supplies its store.
The optional Agents SDK adapter translates model streams into ChatKit
events and carries native model context across browser tool callbacks.

Each application defines its responses, authentication, models, and
provider configuration. Prompts, lead routing, other business rules, and
app-specific tests belong with the application.

When proposing a change for Turnspan, ask whether a second application
would need the same ChatKit server behavior. A browser tool result reaching
the right pending call is a shared SDK behavior. What a business does with
that result is an application decision.
