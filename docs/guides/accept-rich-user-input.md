# Accept rich user input

This guide explains how a ChatKit server accepts user input beyond plain text, such as attachments, structured follow-up answers, and @-mentions, and makes it available to your inference pipeline.

At a high level:

- Attachments let users upload files that your model can read.
- Structured input lets the assistant ask for specific follow-up answers.
- @-mentions let users tag entities so the model does not have to guess from free text.

## Attachments: let users upload files

Let users attach files/images by turning on client support, choosing an upload strategy, wiring the upload endpoints, and converting attachments to model inputs.

### Enable attachments in the client

Turn on attachments in the composer and configure client-side limits:

```ts
const chatkit = useChatKit({
  // ...
  composer: {
    attachments: {
      enabled: true,
      // configure accepted MIME types, count, and size limits here
    },
  },
});
```

Under the hood this maps to `ChatKitOptions.composer.attachments`; see the [`composer.attachments` docs](https://openai.github.io/chatkit-js/api/openai/chatkit/type-aliases/composeroption/#attachments) for all available options.

### Configure an upload strategy

Set [`ChatKitOptions.api.uploadStrategy`](https://openai.github.io/chatkit-js/api/openai/chatkit/type-aliases/fileuploadstrategy/) to:

- **Direct** uploads send the bytes to your backend, which saves the attachment record and returns it.
- **Two-phase** uploads first register an attachment with ChatKit. The returned record contains an `upload_descriptor` that tells the client where to upload the bytes.

For either strategy, pass your file gateway as `attachment_store` and
your metadata repository as `attachment_metadata_store` when constructing the
server. `AttachmentStore` prepares uploads and deletes files.
`AttachmentMetadataStore` saves, loads, and deletes attachment records.

```python
server = MyChatKitServer(
    store=thread_store,
    attachment_store=file_gateway,
    attachment_metadata_store=attachment_metadata_store,
)
```

### Enforce attachment access control

Use the authenticated request context to authorize reads and writes in both
stores and in any upload or preview endpoints. Only return IDs, bytes, or
signed URLs to authorized callers.

### Direct upload

Add the upload endpoint referenced in `uploadStrategy`. It must:

- accept `multipart/form-data` with a `file` field,
- store the bytes wherever you like,
- create `Attachment` metadata, persist it via `AttachmentMetadataStore.save_attachment`, and
- return the `Attachment` JSON.

Implement `AttachmentStore.delete_attachment` to delete the stored bytes.
ChatKit then removes the corresponding record from the metadata store.

Example client configuration:

```js
{
  type: "direct",
  uploadUrl: "/files",
}
```

Example FastAPI direct upload endpoint:

```python
from fastapi import HTTPException, Request, Response
from starlette.datastructures import UploadFile


@app.post("/files")
async def upload_file(request: Request) -> Response:
    context = await authenticate(request)
    async with request.form() as form:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(status_code=400, detail="Expected a file upload.")
        attachment = await upload_to_blob_store(file, context)
    await attachment_metadata_store.save_attachment(attachment, context)

    return Response(
        content=attachment.model_dump_json(context={"exclude_metadata": True}),
        media_type="application/json",
    )
```

Install `python-multipart` in your app to parse this form. Apply your
server-side size and MIME type limits in `upload_to_blob_store`.

### Two-phase upload

Implement `AttachmentStore.create_attachment` to return an attachment with an
upload descriptor. ChatKit saves the returned record in the metadata store.
The client then uploads the bytes using the descriptor.

Client configuration:

```js
{
  type: "two_phase",
}
```

Example two-phase store issuing a multipart upload URL:

```python
from pydantic import AnyUrl

from chatkit.store import AttachmentStore
from chatkit.models import Attachment, AttachmentUploadDescriptor, FileAttachment
from chatkit.requests import AttachmentCreateParams


class BlobAttachmentStore(AttachmentStore[RequestContext]):
    async def create_attachment(
        self, input: AttachmentCreateParams, context: RequestContext
    ) -> Attachment:
        att_id = self.generate_attachment_id(input.mime_type, context)
        upload_url = await issue_multipart_upload_url(att_id, input.mime_type, context)
        return FileAttachment(
            id=att_id,
            mime_type=input.mime_type,
            name=input.name,
            upload_descriptor=AttachmentUploadDescriptor(
                url=AnyUrl(upload_url),
                method="POST",
            ),
        )

    async def delete_attachment(
        self, attachment_id: str, context: RequestContext
    ) -> None:
        await delete_blob(attachment_id, context)
```

### Convert attachments to model input

The message in a `UserMessageTurn` or `RetryTurn` exposes attachments at
`turn.input.message.attachments`. Subclass `ThreadItemConverter` and implement
`attachment_to_message_content` to convert attachments from the thread history
into model input before calling `Runner.run_streamed`.

Example using a blob fetch helper:

```python
import base64

from openai.types.responses import ResponseInputFileParam, ResponseInputImageParam

from chatkit.agent_input import ThreadItemConverter
from chatkit.models import Attachment, ImageAttachment


async def read_bytes(
    attachment_id: str, context: RequestContext
) -> bytes: ...  # Read from your blob store using the authenticated request context.


def as_data_url(mime: str, content: bytes) -> str:
    return "data:" + mime + ";base64," + base64.b64encode(content).decode("utf-8")


class MyConverter(ThreadItemConverter):
    def __init__(self, context: RequestContext):
        self.context = context

    async def attachment_to_message_content(self, attachment: Attachment):
        content = await read_bytes(attachment.id, self.context)
        if isinstance(attachment, ImageAttachment):
            return ResponseInputImageParam(
                type="input_image",
                detail="auto",
                image_url=as_data_url(attachment.mime_type, content),
            )
        if attachment.mime_type == "application/pdf":
            return ResponseInputFileParam(
                type="input_file",
                file_data=as_data_url(attachment.mime_type, content),
                filename=attachment.name or "unknown",
            )
        raise ValueError(f"Unsupported attachment type: {attachment.mime_type}")
```

### Show image attachment previews in thread

Set `ImageAttachment.preview_url` to allow the client to render thumbnails.

- If your preview URLs are **permanent/public**, set `preview_url` once when creating the attachment and persist it.
- For **expiring URLs**, generate a fresh `preview_url` when returning metadata in `ThreadStore.load_thread_items` and `AttachmentMetadataStore.load_attachment`. Use a short-lived signed URL or a redirect to one. The final URL must serve image bytes with the CORS headers your client requires.

## Dictation: speech-to-text input

Enable dictation so users can record audio and have it transcribed into text before sending.

### Enable dictation in the client

Turn on dictation in the composer:

```ts
const chatkit = useChatKit({
  // ...
  composer: {
    dictation: {
      enabled: true,
    },
  },
});
```

This maps to `ChatKitOptions.composer.dictation`.

### Implement `ChatKitServer.transcribe`

When dictation is enabled, the client records audio and sends it to your backend for transcription. Implement `ChatKitServer.transcribe` to accept audio input and return a transcription result.

The client sends one of:

- `"audio/webm;codecs=opus"` (preferred for Chrome/Firefox/Safari 18.4+)
- `"audio/mp4"` (fallback for older Safari/iOS)
- `"audio/ogg;codecs=opus"` (alternative for some environments)

The raw value is available as `audio_input.mime_type`. Use `audio_input.media_type` when you only need the base media type (`"audio/webm"`, `"audio/ogg"`, or `"audio/mp4"`).

Example transcription method using the OpenAI Audio API:

```python
from fastapi import HTTPException
from openai import AsyncOpenAI

from chatkit.models import AudioInput, TranscriptionResult

client = AsyncOpenAI()


async def transcribe(
    self, audio_input: AudioInput, context: RequestContext
) -> TranscriptionResult:
    ext = {
        "audio/webm": "webm",
        "audio/mp4": "m4a",
        "audio/ogg": "ogg",
    }.get(audio_input.media_type)
    if not ext:
        raise HTTPException(status_code=400, detail="Unexpected audio format")

    transcription = await client.audio.transcriptions.create(
        model="gpt-transcribe",
        file=(f"audio.{ext}", audio_input.data, audio_input.media_type),
    )
    return TranscriptionResult(text=transcription.text)
```

Return a `TranscriptionResult` that includes the final `text` that should appear in the composer.

## Structured input: ask for specific answers

Structured input lets your assistant ask the user for small, typed follow-up answers as part of the conversation. The prompt takes over the composer with focused controls, plus a skip option. Use structured input when the next step depends on specific choices or short fields, and free-text back-and-forth would be slower or easier to misread.

The server streams a [`StructuredInputItem`](../api/chatkit/models.md#chatkit.models.StructuredInputItem) during `respond`. ChatKit renders the questions in the composer area instead of the normal free-text input, then records the result on the same thread item when the user answers or skips.

### Stream a structured input item

Yield a `StructuredInputItem` from `respond` when you need the user to answer before continuing.

```python
from datetime import datetime

from chatkit.events import ThreadItemDoneEvent
from chatkit.models import (
    StructuredInputFreeform,
    StructuredInputItem,
    StructuredInputMultipleChoice,
    StructuredInputMultipleChoiceOption,
)


yield ThreadItemDoneEvent(
    item=StructuredInputItem(
        id=self.store.generate_item_id("message", thread, context),
        thread_id=thread.id,
        created_at=datetime.now(),
        inputs=[
            StructuredInputMultipleChoice(
                id="priority",
                question="What priority should I use?",
                options=[
                    StructuredInputMultipleChoiceOption(value="Low"),
                    StructuredInputMultipleChoiceOption(value="Medium"),
                    StructuredInputMultipleChoiceOption(value="High"),
                ],
            ),
            StructuredInputFreeform(
                id="notes",
                question="Any extra context?",
                description="Optional details to include.",
            ),
        ],
    )
)
return
```

Use [`StructuredInputMultipleChoice`](../api/chatkit/models.md#chatkit.models.StructuredInputMultipleChoice) for choice prompts and [`StructuredInputFreeform`](../api/chatkit/models.md#chatkit.models.StructuredInputFreeform) for short text answers. Set `multiple=True` on multiple-choice input when the user may submit more than one value.

After the user submits or skips, ChatKit records the result on the structured input item.

### Convert structured input submissions into model input

The default [`ThreadItemConverter`](../api/chatkit/agent_input.md#chatkit.agent_input.ThreadItemConverter) includes structured input items in model input. It describes the prompt status and each answer as answered, skipped, or unanswered.

If your model needs a different format, override `ThreadItemConverter.structured_input_to_input` before calling `Runner.run_streamed`.

## @-mentions: tag entities in user messages

Enable @-mentions so users can tag entities (like documents, tickets, or users) instead of pasting raw identifiers. Mentions travel through ChatKit as structured tags so the model can resolve entities instead of guessing from free text.

### Enable as-you-type entity lookup in the composer

To enable entity tagging as @-mentions in the composer, configure [`entities.onTagSearch`](https://openai.github.io/chatkit-js/api/openai/chatkit/type-aliases/entitiesoption/#ontagsearch) as a ChatKit.js option.

It should return a list of [Entity](https://openai.github.io/chatkit-js/api/openai/chatkit/type-aliases/entity/) objects that match the query string.

If you want to hint that @-mentions are available, enable the composer `@` button by setting [`entities.showComposerMenu`](https://openai.github.io/chatkit-js/api/openai/chatkit/type-aliases/entitiesoption/#showcomposermenu). When clicked, it inserts `@` into the composer and opens the tag search menu automatically.

```ts
const chatkit = useChatKit({
  // ...
  entities: {
    onTagSearch: async (query: string) => {
      return [
        {
          id: "article_123",
          title: "The Future of AI",
          group: "Trending",
          icon: "globe",
          data: { type: "article" }
        },
        {
          id: "article_124",
          title: "One weird trick to improve your sleep",
          group: "Trending",
          icon: "globe",
          data: { type: "article" }
        },
      ]
    },
    // Optional: show the "@" button in the composer for added discoverability.
    showComposerMenu: true,
  },
})
```

### Convert tags into model input in your server

Mentions arrive server-side as structured tags. The wire type is
`input_tag`, with your app-specific type in `tag.data`. Override
`ThreadItemConverter.tag_to_message_content` to translate the tag into
model-readable content. Resolve the tag using the authenticated request context.

Example converter method that wraps the tagged entity details in custom markup:

```python
from chatkit.agent_input import ThreadItemConverter
from chatkit.models import UserMessageTagContent
from openai.types.responses import ResponseInputTextParam


class MyThreadItemConverter(ThreadItemConverter):
    def __init__(self, context: RequestContext):
        self.context = context

    async def tag_to_message_content(
        self, tag: UserMessageTagContent
    ) -> ResponseInputTextParam:
        if tag.data.get("type") != "article":
            return ResponseInputTextParam(
                type="input_text", text=f"Tagged reference: {tag.text}"
            )

        article = await fetch_article_summary(tag.id, self.context)
        return ResponseInputTextParam(
            type="input_text",
            text=(
                "<ARTICLE_TAG>\n"
                f"ID: {tag.id}\n"
                f"Title: {article.title}\n"
                f"Summary: {article.summary}\n"
                "</ARTICLE_TAG>"
            ),
        )
```

### Pair mentions with retrieval tool calls

When the referenced content is too large to inline, keep the tag lean (id + short summary) and let the model fetch details via a tool. In your system prompt, tell the assistant to call the retrieval tool when it sees an `ARTICLE_TAG`.

Example tool paired with the converter above:

```python
from agents import Agent, RunContextWrapper, function_tool
from chatkit.agents import AgentContext


@function_tool(description_override="Fetch full article content by id.")
async def fetch_article(ctx: RunContextWrapper[AgentContext], article_id: str):
    article = await load_article_content(article_id, ctx.context.request_context)
    return {
        "title": article.title,
        "content": article.body,
        "url": article.url,
    }


assistant = Agent[AgentContext](
    ...,
    tools=[fetch_article],
)
```

In `tag_to_message_content`, include the id the tool expects (for example, `tag.id` or `tag.data["article_id"]`). The model can then decide to call `fetch_article` to pull the full text instead of relying solely on the brief summary in the tag.

### Prompt the model about mentions

Add short system guidance to help the assistant understand the input item that adds details about the @-mention.

For example:

```
- <ARTICLE_TAG>...</ARTICLE_TAG> is a summary of an article the user referenced.
- Use the content as reference data for questions about the article.
- Do not restate the summary verbatim; answer the user’s question concisely.
- Call the `fetch_article` tool with the article id from the tag when more
  detail is needed or the user asks for specifics not in the summary.
```

Combined with the converter above, the model receives explicit, disambiguated entity context while users keep a rich mention UI.

### Handle clicks and previews

Clicks and hover previews apply to the tagged entities shown in past user messages. Mark an entity as interactive when you return it from `onTagSearch` so the client knows to wire these callbacks:

```ts
{
  id: "article_123",
  title: "The Future of AI",
  group: "Trending",
  icon: "globe",
  interactive: true, // clickable/previewable
  data: { type: "article" }
}
```

- `entities.onClick` fires when a user clicks a tag in the transcript. Handle navigation or open a detail view. See the [onClick option](https://openai.github.io/chatkit-js/api/openai/chatkit/type-aliases/entitiesoption/#onclick).
- `entities.onRequestPreview` runs when the user hovers or taps a tag that has `interactive: true`. Return a `Basic` widget. For server-built previews, render a `Basic` template with `WidgetTemplate.build()` and return its JSON. [onRequestPreview option](https://openai.github.io/chatkit-js/api/openai/chatkit/type-aliases/entitiesoption/#onrequestpreview)
