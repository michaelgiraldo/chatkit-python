# Widgets

Widgets are structured UI elements the assistant can stream into the conversation. They let you render forms, cards, lists, or other interactive components instead of plain text.

## Representation and delivery

Here’s how a widget is represented from design time through runtime streaming.

| Stage | What it contains |
| --- | --- |
| Working definition | [Widget UI language](#widget-ui-language) plus a schema (Zod/JSON) and example data you author in <https://widgets.chatkit.studio>. |
| Published definition | The exported [`.widget` file](#widget-files) bundling the layout, schema, and sample data. |
| Server runtime (definition only) | [`WidgetTemplate`](../api/chatkit/templates.md#chatkit.templates.WidgetTemplate) instance loaded from the `.widget` file. |
| Server runtime (hydrated) | A [`WidgetRoot`](../api/chatkit/widget_models.md#chatkit.widget_models.WidgetRoot) built from a template or from typed Python builders. |
| Streamed and saved | The same widget tree is serialized to JSON in streamed and stored [`WidgetItem`](../api/chatkit/models.md#chatkit.models.WidgetItem) records. Reload uses that tree directly and preserves nested properties, including client extension fields. |
| Rendered by the client | ChatKit.js deserializes the JSON into typed widget objects (for example, [`Card`](https://openai.github.io/chatkit-js/api/openai/chatkit-react/namespaces/widgets/type-aliases/card/) or [`ListView`](https://openai.github.io/chatkit-js/api/openai/chatkit/namespaces/widgets/type-aliases/listview/)) and renders them; entity previews use [`BasicRoot`](https://openai.github.io/chatkit-js/api/openai/chatkit-react/namespaces/widgets/type-aliases/basicroot/) returned from [`entities.onRequestPreview`](#entity-previews). |
| Sent as model input | A Responses API `Message` produced from a [`WidgetItem`](../api/chatkit/models.md#chatkit.models.WidgetItem) via [`ThreadItemConverter.widget_to_input`](../api/chatkit/agent_input.md#chatkit.agent_input.ThreadItemConverter.widget_to_input). |


## Widget UI language

Widget layouts use a strict, simplified JSX dialect that only allows specific components and props. Explore the available components and their props in <https://widgets.chatkit.studio/components> to see what the renderer supports.

### Containers

Every widget must be wrapped in a root-level container element. For single, self-contained content such as a summary, confirmation, or form, use `<Card>`. For a set of options (for example, restaurants or files), use `<ListView>`. Reserve `<Basic>` for entity previews.

- `<Card>`: Simple card with a light border and plain background; supports confirm and cancel actions.
- `<ListView>`: Scroll-friendly list with built-in “show more” mechanics. Children must be `<ListViewItem>`, and `<ListViewItem>` must only appear as a direct child of `<ListView>`; it has a constrained prop set for row-like layout (`children`, `gap`, `align`, `onClickAction`).
- `<Basic>`: Minimal container only used for entity previews.

## .widget files

Exported `.widget` files are JSON blobs that include the widget template, the expected data schema, and supporting metadata. You can load them server-side and render widgets dynamically with `WidgetTemplate`; see [Build widgets with `WidgetTemplate`](../guides/build-interactive-responses-with-widgets.md#build-widgets-with-widgettemplate) for examples.

## WidgetItem

[`WidgetItem`](../api/chatkit/models.md#chatkit.models.WidgetItem) represents a widget rendered as a [thread item](threads.md#what-are-thread-items) in the chat UI. In addition to a reference to the widget instance, it contains a `copy_text` field that represents the text value copied to the clipboard when the user clicks the copy button below the response.

## Entity previews

The [`entities.onRequestPreview`](https://openai.github.io/chatkit-js/api/openai/chatkit-react/type-aliases/entitiesoption/#onrequestpreview) ChatKit option returns a preview typed as [`BasicRoot`](https://openai.github.io/chatkit-js/api/openai/chatkit-react/namespaces/widgets/type-aliases/basicroot/). On the server, build a `Basic` template using `WidgetTemplate.build()` and return its JSON.


## When to use

- Collect structured input (forms) or present rich results (tables, cards, charts) that text alone cannot convey.
- Present the user with multiple choice options.
- Pair with actions to let users submit selections, confirm steps, or trigger server-side work.
- Mix with text to provide explanation plus an interactive control.

## Related guides

- [Build interactive responses with widgets](../guides/build-interactive-responses-with-widgets.md)
