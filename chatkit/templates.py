"""Render exported ChatKit widget files with Jinja."""

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, Template
from pydantic import BaseModel

from . import widget_models

_jinja_env = Environment(undefined=StrictUndefined)


class WidgetTemplate:
    """Load and render a `.widget` definition into a canonical widget tree."""

    def __init__(self, definition: dict[str, Any]):
        self.version = definition["version"]
        if self.version != "1.0":
            raise ValueError(f"Unsupported widget spec version: {self.version}")

        self.name = definition["name"]
        template = definition["template"]
        self.template = (
            template
            if isinstance(template, Template)
            else _jinja_env.from_string(template)
        )

    @classmethod
    def from_file(cls, path: str | Path) -> WidgetTemplate:
        """Read a widget definition. Relative paths use the working directory."""
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def build(
        self, data: dict[str, Any] | BaseModel | None = None
    ) -> widget_models.WidgetRoot:
        """Render template data into a Card, ListView, or Basic widget tree."""
        context = data.model_dump() if isinstance(data, BaseModel) else data
        return widget_models.WidgetRoot.model_validate_json(
            self.template.render(**(context or {}))
        )
