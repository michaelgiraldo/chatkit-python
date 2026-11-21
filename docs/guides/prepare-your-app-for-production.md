# Prepare your app for production

This guide covers the operational work you should do before rolling out a ChatKit‑powered experience in production:

- Set up **localization** so prompts, system messages, and tool output match the user’s locale.
- Configure **monitoring and logging** so you can debug issues and correlate ChatKit traffic with your backend traces.
- Review **security and authentication** for your ChatKit endpoint.
- Register and use **domain keys** to lock ChatKit down to your approved hostnames.

Use it as a checklist alongside your own product’s launch process.

## Localize prompts, UI copy, and tool output

By the time you go live, you should have a clear story for which locales you support and how locale flows from the client into your backend and model prompts.

ChatKit always picks a **single active locale** and:

- Uses the **browser locale by default**.
- Lets you **override the locale on the client** (for example, from your own locale picker) by passing the `locale` option when you initialize ChatKit.

Resolve the locale for your backend from the user's preferences or the
request's `Accept-Language` header. Validate it against the locales your app
supports and apply a fallback.

At a minimum:

- **Decide which locales you support** (for example `["en", "fr", "de"]`) and what the default/fallback is.
- **Localize tool output and error messages** so the assistant’s replies feel consistent with the rest of your product.

For example, you might include locale in your per‑request context:

```python
from dataclasses import dataclass


@dataclass
class RequestContext:
    user_id: str
    locale: str
```

Then, when you build prompts or tool output, read `context.locale` and render language‑appropriate text using your localization system. For example, with `gettext`:

```python
from pathlib import Path
import gettext

from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.events import ProgressUpdateEvent


LOCALE_DIR = Path(__file__).parent / "locales"
_translations: dict[str, gettext.NullTranslations] = {}


def get_translations(locale: str) -> gettext.NullTranslations:
    """Return a gettext translation object for the given locale."""
    if locale not in _translations:
        _translations[locale] = gettext.translation(
            "messages",  # your .po/.mo domain
            localedir=LOCALE_DIR,
            languages=[locale],
            fallback=True,
        )
    return _translations[locale]


@function_tool()
async def load_document(
    ctx: RunContextWrapper[AgentContext],
    document_id: str,
):
    locale = ctx.context.request_context.locale
    _ = get_translations(locale).gettext
    await ctx.context.stream(
        ProgressUpdateEvent(icon="document", text=_("Loading document…"))
    )
    doc = await get_document_by_id(document_id)
    if not doc:
        raise ValueError(_("We couldn’t find that document."))
    return doc
```

When you call the model (for example via the OpenAI Responses API), include the user’s locale either directly in the prompt or as part of a system message so the model responds in the right language.

## Monitor logs and errors

You should be able to answer questions like “what went wrong for this user at this time?” and “are ChatKit requests healthy right now?” before you roll out broadly.

### Capture client logs

On the **client side**, subscribe to ChatKit’s log and error events and forward them into your own telemetry system, tagged with:

- User identifier (or stable anonymous id).
- Session or request id.
- The current thread id.

In React, use the `onLog` and `onError` callbacks for
[ChatKit.js events](https://openai.github.io/chatkit-js/api/openai/chatkit/type-aliases/chatkitevents/).

```tsx
import { ChatKit, useChatKit } from "@openai/chatkit-react";

export function SupportChat({
  backendUrl,
  domainKey,
  userId,
}: {
  backendUrl: string;
  domainKey: string;
  userId: string;
}) {
  const { control } = useChatKit({
    api: { url: backendUrl, domainKey },
    onLog: ({ name, data }) =>
      sendToTelemetry({
        name,
        // Avoid forwarding raw message text or tool arguments directly.
        data: scrubSensitiveFields(data),
        userId,
      }),
    onError: ({ error }) =>
      sendToTelemetry({
        name: "chatkit.error",
        error: scrubSensitiveFields(error),
        userId,
      }),
  });

  return <ChatKit control={control} className="h-[600px]" />;
}
```

With the web component, listen for `chatkit.log` and `chatkit.error` events:

```ts
import type { OpenAIChatKit } from "@openai/chatkit";

const chatkit = document.getElementById("my-chat") as OpenAIChatKit;

chatkit.addEventListener("chatkit.log", ({ detail }) => {
  sendToTelemetry({
    name: detail.name,
    data: scrubSensitiveFields(detail.data),
    userId,
  });
});

chatkit.addEventListener("chatkit.error", (event) => {
  sendToTelemetry({
    name: "chatkit.error",
    error: scrubSensitiveFields(event.detail.error),
    userId,
  });
});
```

These events can include **PII and message contents**, so avoid blanket-forwarding entire payloads; instead, extract and forward only the fields you need (for example, error codes, item ids, thread ids, and high‑level event names) and/or scrub sensitive fields before sending them to your logging provider.

### Monitor your ChatKit endpoint

On the **backend**, you should still capture basic logs around your ChatKit endpoint so you can correlate client telemetry with server behavior:

- Incoming HTTP request (path, method, user id, thread id).
- Calls to `ChatKitServer.process` and your `ThreadStore` implementation.
- Outbound calls to OpenAI or other model providers.
- Any errors raised from tools or your own business logic.


## Security and authentication

Production deployments should treat your ChatKit endpoint as a privileged backend:

- **Authenticate every request** to your `/chatkit` endpoint (for example, with your existing session cookies, bearer tokens, or signed JWTs).
- **Authorize access to threads and attachments** based on your own user and tenant model.
- **Protect secrets** such as OpenAI API keys and internal service credentials in environment variables or a secret manager—never in source control.
- **Validate inputs** before calling tools or downstream systems.

You should also be explicit about how you handle **prompt injection**:

- Treat all user text, attachments, and tool output as untrusted input.
- Avoid building any `role="system"` model inputs from values that might come from the user (including fields like subject lines, titles, or descriptions).
- Keep system messages static or derived only from trusted configuration so users cannot silently change your instructions to the model.

### Authenticate your ChatKit endpoint

The Python SDK expects your own app to handle authentication; `ChatKitServer` works with whatever `RequestContext` you choose. A common pattern is to:

1. Authenticate the incoming HTTP request using your web framework (session middleware, OAuth bearer tokens, etc.).
2. Build a `RequestContext` that includes the authenticated user id, org/tenant, and any roles or scopes.
3. Pass that context into `process_request`.

This example assumes FastAPI authentication middleware that supplies
`request.user` with the application's user identity.

```python
from fastapi import Depends, FastAPI, HTTPException, Request, Response

from chatkit.fastapi import process_request


def get_current_user(request: Request) -> str:
    if not request.user.is_authenticated:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return request.user.identity


app = FastAPI()
store = MyThreadStore(...)
server = MyChatKitServer(store)


@app.post("/chatkit")
async def chatkit_endpoint(
    request: Request,
    user_id: str = Depends(get_current_user),
) -> Response:
    context = RequestContext(user_id=user_id, locale=resolve_locale(request))
    return await process_request(server, request, context)
```

Implement `resolve_locale` using the preferences and fallback described above.
Enforce per-user or per-tenant access in your thread store, attachment stores,
and tools. Check the authenticated request context before returning or changing data.

### Check thread history with a signed visitor session

After deploying, list threads through the same HTTPS endpoint the browser uses.
A health check or successful sign-in can pass while the app's chat server or
store still fails to initialize. Turnspan handles `threads.list` by calling
`store.load_threads` with the authenticated request context. This reads history
without creating a thread or invoking the responder.

Use your app's normal session flow to obtain a short-lived credential for a
fresh test visitor whose history is empty. Session issuance and the credential
format belong to your app. For a bearer-token deployment, supply the session
token in `CHATKIT_SESSION_TOKEN` and the deployed ChatKit URL in `CHATKIT_URL`.
For cookie-based authentication, send the visitor's session cookie instead.

```bash
set -o pipefail
curl --silent --show-error --fail --max-time 15 \
  --header 'Content-Type: application/json' --header @- \
  --data '{"type":"threads.list","params":{}}' "${CHATKIT_URL:?}" <<EOF | \
  jq --exit-status '.data == [] and .has_more == false and .after == null'
Authorization: Bearer ${CHATKIT_SESSION_TOKEN:?}
EOF
```

Expect HTTP 200 and an empty JSON page, typically
`{"data":[],"has_more":false}`. The command prints only whether the page matches.
If the route denies the request, check the session and the URL. If it returns
HTTP 500, check the app's server construction, store configuration and database
read permissions. Repeat without the credential and confirm the protected route
rejects the request according to your app's authentication policy.

### Handle PII and data retention

Because ChatKit threads and items can contain user text, attachments, and tool output:

- **Decide what you persist** and for how long. Apply retention policies to threads, items, and attachments.
- **Avoid storing unnecessary PII** in thread metadata or tool return values.
- **Encrypt data at rest** using your database’s built‑in features or application‑level encryption where needed.

## Domain keys

For a self-hosted backend, ChatKit uses a [domain key](https://openai.github.io/chatkit-js/api/openai/chatkit/type-aliases/customapiconfig/#domainkey) to verify your registered frontend domain. Your backend still authenticates each request.

1. Open the [domain allowlist](https://platform.openai.com/settings/organization/security/domain-allowlist) for your OpenAI organization.
2. Register each hostname that will host your ChatKit UI (for example, `app.example.com`, `support.example.com`).
3. Copy the generated **domain key** for each entry.

Your client configuration should include that `domainKey` alongside the URL to your ChatKit Python backend.

```ts
const options = {
  api: {
    url: "https://your-domain.com/api/chatkit",
    // Copy this value from the domain allowlist entry.
    domainKey: "your-domain-key",
  },
};
```

When you go live, make sure all of your production hostnames are registered in the domain allowlist.
