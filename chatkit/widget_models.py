"""Canonical widget trees for JSON, streaming, and storage."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_serializer


class WidgetComponent(BaseModel):
    """A widget node whose JSON properties survive streaming and storage."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)  # pyright: ignore[reportIncompatibleVariableOverride]

    type: str
    id: str | None = None
    key: str | None = None
    children: WidgetComponent | list[WidgetComponent] | None = None

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        # Preserve explicit JSON nulls even when the enclosing event uses
        # exclude_none. Only unset optional node fields are absent on the wire.
        data = {"type": self.type, **(self.model_extra or {})}
        for field in ("id", "key", "children"):
            if field in self.model_fields_set:
                data[field] = getattr(self, field)
        return data


class WidgetRoot(WidgetComponent):
    """The canonical layout used by widget events and saved thread items."""

    type: Literal["Card", "ListView", "Basic"]  # pyright: ignore[reportIncompatibleVariableOverride]
