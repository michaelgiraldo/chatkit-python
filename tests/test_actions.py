from typing import Literal

import pytest
from pydantic import ValidationError

from chatkit.actions import Action, IncomingAction


def test_typed_action_requires_the_declared_payload():
    action = Action[Literal["save"], dict[str, int]]
    with pytest.raises(ValidationError) as error:
        action.model_validate({"type": "save"})

    (issue,) = error.value.errors()
    assert issue["loc"] == ("payload",)
    assert issue["type"] == "missing"
    assert action.create({"count": 1}).model_dump(exclude_defaults=True) == {
        "type": "save",
        "payload": {"count": 1},
    }


def test_payload_free_browser_action_and_explicit_empty_typed_payload():
    incoming = IncomingAction.model_validate({"type": "save"})
    typed = Action[Literal["save"], None](type="save", payload=None)
    assert (
        incoming.model_dump() == typed.model_dump() == {"type": "save", "payload": None}
    )
