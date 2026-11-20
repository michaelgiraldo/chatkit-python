from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, Field

Handler = Literal["client", "server"]
LoadingBehavior = Literal["auto", "none", "self", "container"]

DEFAULT_HANDLER: Handler = "server"
DEFAULT_LOADING_BEHAVIOR: LoadingBehavior = "auto"


class ActionConfig(BaseModel):
    """Configure a widget action and how the browser dispatches it."""

    type: str = Field(min_length=1)
    payload: Any = None
    handler: Handler = DEFAULT_HANDLER
    loadingBehavior: LoadingBehavior = DEFAULT_LOADING_BEHAVIOR
    streaming: bool = True


class Action[TType: str, TPayload](BaseModel):
    """Describe an action with a required name and an explicitly typed payload."""

    type: TType = Field(frozen=True, min_length=1)
    payload: TPayload

    @classmethod
    def create(
        cls,
        payload: TPayload,
        handler: Handler = DEFAULT_HANDLER,
        loading_behavior: LoadingBehavior = DEFAULT_LOADING_BEHAVIOR,
        streaming: bool = True,
    ) -> ActionConfig:
        action_type = None
        anno = cls.model_fields["type"].annotation
        if get_origin(anno) is Literal:
            lits = get_args(anno)
            if len(lits) == 1 and isinstance(lits[0], str):
                action_type = lits[0]

        if action_type is None:
            raise TypeError(
                "Cannot infer 'type' for this Action[...]. Do not call create() on generic Action."
            )

        return ActionConfig(
            type=action_type,
            payload=payload,
            handler=handler,
            loadingBehavior=loading_behavior,
            streaming=streaming,
        )


class IncomingAction(Action[str, Any]):
    """An action from the browser, where an absent payload means no arguments."""

    payload: Any = None
