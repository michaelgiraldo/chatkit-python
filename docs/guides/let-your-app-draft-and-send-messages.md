# Let your app draft and send messages

Use ChatKit’s commands to let your app pre-fill the composer and send messages programmatically for quick replies, “ask again” buttons, or deep links from the rest of your UI.

At a high level:

- `setComposerValue` lets your app draft or edit the pending message.
- `sendUserMessage` lets your app send a message without the user pressing Enter.

## Get ChatKit commands from `useChatKit`

When you call `useChatKit`, destructure commands alongside the `control`
object you pass into `<ChatKit />`. This example uses the same self-hosted API
configuration as the [production guide](prepare-your-app-for-production.md#domain-keys).

```tsx
import {ChatKit, useChatKit} from "@openai/chatkit-react";

export function Inbox({backendUrl, domainKey}: {backendUrl: string; domainKey: string}) {
  const {
    control,
    setComposerValue,
    sendUserMessage,
  } = useChatKit({
    api: {url: backendUrl, domainKey},
  });

  return <ChatKit control={control} />;
}
```

The commands are safe to call as long as ChatKit is not currently loading a thread or streaming a response; combine them with the loading/response state from [**Keep your app in sync with ChatKit**](keep-your-app-in-sync-with-chatkit.md) when you need to guard calls.

## Draft messages with `setComposerValue`

Use `setComposerValue` to pre-fill or update the composer text from your own UI:

- Quick-reply chips that insert a suggested reply.
- “Ask again with more detail” buttons that tweak the last question.
- Deep links from outside the chat that open a specific prompt.

```tsx
function QuickReplies({
  setComposerValue,
}: {
  setComposerValue: (params: {text: string}) => Promise<void>;
}) {
  return (
    <div className="quick-replies">
      <button onClick={() => setComposerValue({text: "Can you summarize this thread?"})}>
        Summarize this thread
      </button>
      <button onClick={() => setComposerValue({text: "Explain this like I'm five."})}>
        Explain like I'm five
      </button>
    </div>
  );
}
```

`setComposerValue` only changes the draft text; the user can still edit it before sending, or you can pair it with `sendUserMessage` to fire immediately.

## Send messages with `sendUserMessage`

Use `sendUserMessage` when your app needs to initiate a turn directly—for example, from a custom toolbar button or a widget action handled on the client.

```tsx
export function Inbox({backendUrl, domainKey}: {backendUrl: string; domainKey: string}) {
  const {
    control,
    sendUserMessage,
  } = useChatKit({
    api: {url: backendUrl, domainKey},
  });

  const handleHelpClick = () => {
    // Send a canned message from a fresh thread
    sendUserMessage({text: "I need help with my billing.", newThread: true});
  };

  return (
    <>
      <button onClick={handleHelpClick}>Contact support</button>
      <ChatKit control={control} />
    </>
  );
}
```

You can also rely on the current active thread when calling `sendUserMessage` without `newThread: true`.

