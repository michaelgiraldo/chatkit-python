# Upgrade from Turnspan 3 to 6

This guide starts at [Turnspan 3](https://github.com/michaelgiraldo/chatkit-python/releases/tag/v1.6.5%2Bturnspan.3).
Update the application pin to the exact Turnspan 6 commit you are adopting.
Keep the previous package pin, lockfile, browser build, and store version
available for rollback. The [pinning guide](pin-the-fork.md) explains how to
verify the package in the artifact that will serve traffic.

## Update the application

Apply each row that describes your integration.

| Integration | Change for Turnspan 6 | Check |
| --- | --- | --- |
| Text chat with the Agents SDK | Pass `resume_from=None` to `AgentContext` when a run starts fresh. For a resumed display workflow, pass the exact saved and authorized `WorkflowItem`. Supply provider response or conversation IDs to the runner through your existing model context path. | Stream a reply, reopen the conversation, and send a follow-up. If your app resumes saved display progress, verify that it loads the intended workflow. |
| Browser tools | Read the tool input from `params.input` and return `{item_id, output}` with the ID from `params.item_id`. Select `continuation="local"` or `continuation="responses"` in `request_client_tool`. Build the next model request from the completed `ClientToolTurn` with `client_tool_continuation_input`. | Complete a new callback with the correct ID after reopening the store. Verify the next model request carries the original model call ID and arguments or the matching provider context. |
| Custom browser-tool store | Add scoped callback registration, completion, and cancellation. Commit the callback write and the corresponding item change together. Preserve `model_continuation` in storage. If you call the shared `validate_pending_claim` helper, remove the `latest_item_id` keyword. | Run the browser-tool [store contract checks](respond-to-user-message.md#check-your-apps-store) with two distinct test users. Verify that history hides cancelled and unregistered pending items before paging. |
| Custom HTTP transport | Use `encode_message` for JSON and `encode_sse` for event streams, or reproduce their browser representation. Keep `model_continuation` server-side. | Inspect a streamed and reloaded tool item for the browser envelope and verify that both omit the model continuation record. |

For a fresh text-only run, the context construction looks like this.

```python
from chatkit.agents import AgentContext

agent_context = AgentContext(
    thread=turn.thread,
    store=self.store,
    request_context=turn.context,
    resume_from=None,
)
```

For browser tools, choose the continuation strategy that matches your existing
model history. With `local`, store and replay the saved native model input
and output items through Turnspan. With `responses`, use the provider-held
response or conversation and send the function outputs it still needs. Use
provider retention consistent with your application’s policy. The [browser
tool guide](update-client-during-response.md#build-the-next-model-request)
shows how to build the next model request for each strategy.

## Rehearse and switch traffic

1. Test the new stack on a copy of the application's store. Read an old text
   thread and a completed tool call, then verify a fresh reply with the Turnspan
   6 build. Keep the old installation and a restorable snapshot available.
2. If you use browser tools, stop opening Turnspan 3 callbacks before the
   switch. Give open calls time to finish on the old stack. End any remaining
   pending interactions in the Turnspan 3 store and have those users reload
   and restart the tool interaction. Take the rollback snapshot after
   confirming it contains no pending browser calls.
3. Switch the browser handler, server, and store implementation together.
   Refresh tabs or expire sessions still using the old browser handler. A
   Turnspan 6 store hides pending legacy rows without a registered invocation.
   Preserve completed history and issue a fresh invocation ID when a user
   restarts an interrupted interaction.
4. Verify that a late Turnspan 3 callback cannot resume a turn. Reissue the
   tool, reopen the server with the same store, complete the callback, and
   verify that a repeat is rejected. Check the input given to the model in the
   same continuation mode your application uses.

## Roll back together

Stop Turnspan 6 from accepting new writes before switching back. Save a copy
of the current data to reconcile changes after the cutover if needed. Restore
the previous browser handler, Python artifact, and store implementation with
the drained Turnspan 3 snapshot or with a reverse migration you have tested on
your own store. Plan for any chat records written after that snapshot. Use a
matching browser session or a versioned route so a stale Turnspan 6 callback
cannot reach the Turnspan 3 handler.

The Turnspan 6 SQLite example stores cancellation in a new callback table.
Turnspan 3 determines callback eligibility from the newest transcript item.
In a rehearsal, reverting only the server on a copy of Turnspan 6 data
exposed a cancelled tool item and accepted its late callback. A schema that
opens successfully can still have different callback behavior. Verify an old
thread, a fresh reply, and a fresh Turnspan 3 callback after restoring the
matching store.

## Reference rehearsal

The offline rehearsal installed the published Turnspan 3 commit
`e0decab` and candidate `17ffe65` in separate Python 3.14.10 environments
with Agents SDK 0.25.0 and OpenAI Python 3.57.0. It used synthetic FastAPI
requests, the two revisions of the reference SQLite store, and no model
requests. It preserved old text and completed tool history. Turnspan 6
rejected the old callback body with HTTP 422 and the obsolete invocation
with HTTP 404. A new callback completed once after the store reopened, and
the repeated request returned HTTP 409. Restoring a drained Turnspan 3
snapshot recovered the old history and a fresh Turnspan 3 reply and tool
interaction.
