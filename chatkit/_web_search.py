from collections.abc import Callable, Iterator
from typing import Literal

from openai.types.responses import ResponseFunctionWebSearch, ResponseStreamEvent

from .types import (
    SearchTask,
    ThreadItemAddedEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
    URLSource,
    WorkflowItem,
    WorkflowTaskAdded,
    WorkflowTaskUpdated,
)


def _search_status(
    call: ResponseFunctionWebSearch,
) -> Literal["loading", "complete", "none"]:
    if call.status in ("in_progress", "searching"):
        return "loading"
    if call.status == "completed":
        return "complete"
    return "none"


def _search_task_metadata(
    task: SearchTask, call: ResponseFunctionWebSearch
) -> SearchTask:
    action = getattr(call, "action", None)
    if action is None:
        return task
    queries = list(task.queries)
    sources = {source.url: source for source in task.sources}
    title = task.title
    if action.type == "search":
        if action.query:
            queries.append(action.query)
        queries.extend(action.queries or [])
        urls = [source.url for source in action.sources or []]
    elif action.type in ("open_page", "find_in_page"):
        urls = [action.url] if action.url else []
        title = title or action.url
    else:
        return task

    for url in urls:
        if url.startswith(("https://", "http://")):
            sources.setdefault(url, URLSource(url=url, title=url))

    return task.model_copy(
        update={
            "title": title,
            "title_query": queries[0] if queries else task.title_query,
            "queries": list(dict.fromkeys(queries)),
            "sources": list(sources.values()),
        }
    )


class _WebSearchProgress:
    """Map provider search items to task slots for the duration of one run."""

    def __init__(self, start_workflow: Callable[[], WorkflowItem]) -> None:
        self._start_workflow = start_workflow
        self._locations: dict[str, tuple[str, int]] = {}

    def has_tasks(self, item: WorkflowItem) -> bool:
        return any(
            workflow_id == item.id for workflow_id, _ in self._locations.values()
        )

    def settle(self, item: WorkflowItem) -> None:
        """Clear loading states for search calls that never sent a terminal event."""
        for workflow_id, task_index in self._locations.values():
            if workflow_id == item.id:
                task = item.workflow.tasks[task_index]
                if isinstance(task, SearchTask) and task.status_indicator == "loading":
                    item.workflow.tasks[task_index] = task.model_copy(
                        update={"status_indicator": "none"}
                    )

    def handle(
        self, event: ResponseStreamEvent, workflow: WorkflowItem | None
    ) -> Iterator[ThreadStreamEvent] | None:
        """Return updates for an owned event, or None for a different event family."""
        if (
            event.type == "response.output_item.added"
            and event.item.type == "web_search_call"
        ):
            call = event.item
            return self._update(
                workflow, call.id, _search_status(call), call=call, start=True
            )
        if (
            event.type == "response.output_item.done"
            and event.item.type == "web_search_call"
        ):
            call = event.item
            status = "complete" if call.status == "completed" else "none"
            return self._update(workflow, call.id, status, call=call)
        if (
            event.type == "response.web_search_call.in_progress"
            or event.type == "response.web_search_call.searching"
        ):
            return self._update(workflow, event.item_id, "loading", start=True)
        if event.type == "response.web_search_call.completed":
            return self._update(workflow, event.item_id, "complete")
        return None

    def _update(
        self,
        workflow: WorkflowItem | None,
        item_id: str,
        status: Literal["loading", "complete", "none"],
        *,
        call: ResponseFunctionWebSearch | None = None,
        start: bool = False,
    ) -> Iterator[ThreadStreamEvent]:
        location = self._locations.get(item_id)
        if location is None:
            if not start:
                return
            if workflow is None:
                workflow = self._start_workflow()
                yield ThreadItemAddedEvent(item=workflow.model_copy(deep=True))
            tasks = workflow.workflow.tasks
            task_index = len(tasks)
            task = SearchTask(status_indicator=status)
            if call is not None:
                task = _search_task_metadata(task, call)
            self._locations[item_id] = (workflow.id, task_index)
            tasks.append(task)
            yield ThreadItemUpdatedEvent(
                item_id=workflow.id,
                update=WorkflowTaskAdded(task=task, task_index=task_index),
            )
            return

        workflow_id, task_index = location
        if workflow is None or workflow.id != workflow_id:
            return
        tasks = workflow.workflow.tasks
        previous = tasks[task_index]
        if not isinstance(previous, SearchTask):
            return
        if previous.status_indicator != "loading" and (
            status == "loading" or call is None
        ):
            return
        task = previous.model_copy(update={"status_indicator": status})
        if call is not None:
            task = _search_task_metadata(task, call)
        if task != previous:
            tasks[task_index] = task
            yield ThreadItemUpdatedEvent(
                item_id=workflow_id,
                update=WorkflowTaskUpdated(task=task, task_index=task_index),
            )
