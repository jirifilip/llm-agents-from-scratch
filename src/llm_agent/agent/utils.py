import base64
from io import BytesIO
from PIL import Image
from typing import Any, TypeVar

T = TypeVar("T")


def select_from_dict(dictionary: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return dict(
        (key, value) for key, value in dictionary.items() if key in keys
    )


def wrap_into_list(list_or_element: T | list[T]) -> list[T]:
    if isinstance(list_or_element, list):
        return list_or_element
    
    return [list_or_element]


def convert_image_to_base64_string(image: Image.Image, image_format: str = "JPEG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format.upper())

    b64_bytes = base64.b64encode(buffer.getvalue())

    return b64_bytes.decode()
