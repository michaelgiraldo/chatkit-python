import json

from pydantic import TypeAdapter

from chatkit.models import EntitySource, Source
from chatkit.protocol import encode_message


def test_old_entity_preview_is_ignored_and_not_reemitted():
    source = TypeAdapter(Source).validate_json(
        '{"type":"entity","id":"example","title":"Example","preview":"lazy"}'
    )

    assert isinstance(source, EntitySource)
    assert source.id == "example"
    assert "preview" not in json.loads(encode_message(source))
