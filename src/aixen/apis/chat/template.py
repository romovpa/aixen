import base64
import inspect
import json
import re
import uuid
from collections.abc import Callable
from functools import wraps
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import jinja2
import requests
from jinja2 import pass_context
from PIL import Image
from pydantic import HttpUrl, TypeAdapter

from aixen.context import File, MixedDictPydanticJSONEncoder, processor

from .generate import chat_generate, chat_generate_structured


def _add_attachment(render_context, content_item):
    attachment_id = str(uuid.uuid4())
    placeholder = f"\0{attachment_id}\0"

    attachments = render_context.get("__attachments", {})
    attachments[attachment_id] = content_item

    return placeholder


def _is_http_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False


@pass_context
def image_filter(
    render_context: jinja2.runtime.Context,
    image: str | HttpUrl | File,
    max_dim: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
):
    if max_dim is not None:
        max_width = max_width or max_dim
        max_height = max_height or max_dim
    need_resize = max_width is not None or max_height is not None

    if isinstance(image, str) and _is_http_url(image):
        image_url = image
        if not need_resize:
            attachment = {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        else:
            img_raw = requests.get(image_url).content
            img = Image.open(BytesIO(img_raw))
            img.thumbnail((max_width or img.width, max_height or img.height))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            attachment = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                },
            }
    else:
        image_file = File(image) if not isinstance(image, File) else image
        with image_file.open("rb") as f:
            img = Image.open(f)
            if need_resize:
                img.thumbnail((max_width or img.width, max_height or img.height))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        attachment = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
            },
        }

    return _add_attachment(render_context, attachment)


def json_filter(obj: Any, **kwargs):
    return json.dumps(obj, cls=MixedDictPydanticJSONEncoder, **kwargs)


chat_template_env = jinja2.Environment()
chat_template_env.filters["image"] = image_filter
chat_template_env.filters["json"] = json_filter


def _split_prompt_into_messages(prompt: str):
    sep_pattern = re.compile(r"(<\|\w+\|>)")
    role_pattern = re.compile(r"<\|(?P<role>\w+)\|>")

    parts = sep_pattern.split(prompt)

    messages = []

    current_role = "system"
    for part in parts:
        part = part
        role_match = role_pattern.match(part)
        if role_match:
            current_role = role_match.group("role")
        else:
            messages.append(
                {
                    "role": current_role,
                    "content": part,
                }
            )

    return messages


def _insert_attachments(render_context: dict, content: str):
    sep_pattern = re.compile("(\0[a-f0-9-]+\0)")
    attachment_pattern = re.compile("\0(?P<attachment_id>[a-f0-9-]+)\0")

    attachments = render_context.get("__attachments", {})

    content_parts = []
    for part in sep_pattern.split(content):
        attachment_match = attachment_pattern.match(part)
        if attachment_match:
            attachment_id = attachment_match.group("attachment_id")
            attachment = attachments[attachment_id]
            content_parts.append(attachment)
        else:
            if part:
                content_parts.append(
                    {
                        "type": "text",
                        "text": part,
                    }
                )

    return content_parts


def _text_part_to_string(content_parts):
    if all(part["type"] == "text" for part in content_parts):
        return "".join(part["text"] for part in content_parts)
    return content_parts


def apply_prompt_template(text: str, variables: dict):
    system_content = []
    messages = []
    for message_template in _split_prompt_into_messages(text):
        role = message_template["role"]
        template = chat_template_env.from_string(message_template["content"])

        render_context = {
            "__attachments": {},
            **variables,
        }

        content = template.render(render_context)

        content_parts = _insert_attachments(render_context, content)

        if role == "system":
            system_content += content_parts
        else:
            messages.append(
                {
                    "role": role,
                    "content": _text_part_to_string(content_parts),
                }
            )

    system = _text_part_to_string(system_content)

    return system, messages


def chat_func(
    func: Callable | None = None,
    model: str | None = None,
    tool_output: bool = True,
    tools: list[Callable] | None = None,
):
    """
    Creates a LLM-calling processor based on a function's signature and docstring.

    >>> @chat_func
    ... def greet(name: str) -> str:
    ...    '''
    ...    Greet the user.
    ...    <|user|>
    ...    Hi, I'm {{ name }}.
    ...    '''

    Dry-run the function to get the system prompt and messages:
    >>> greet.prompt("Alice")
    ('Greet the user.', [{'role': 'user', 'content': "\nHi, I'm Alice."}])

    Call the function to generate a response:
    >>> greet("Alice")
    'Hello, Alice! How can I assist you today?'
    """

    def decorator(func):
        sig = inspect.signature(func)

        def prompt(*args, **kwargs) -> tuple[str, list[dict]]:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            args_dict = bound_args.arguments

            prompt_template = inspect.cleandoc(func.__doc__)
            system, messages = apply_prompt_template(prompt_template, args_dict)

            return system, messages

        prompt.__signature__ = inspect.Signature(
            list(sig.parameters.values()),
            return_annotation=inspect.signature(prompt).return_annotation,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            system, messages = prompt(*args, **kwargs)

            if not messages:
                query_parts = []
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                for name, value in bound_args.arguments.items():
                    if isinstance(value, str):
                        query_parts.append(f"{name}: {value}")
                    else:
                        data = json.dumps(value, cls=MixedDictPydanticJSONEncoder)
                        query_parts.append(f"{name}: {data}")
                query = "\n\n".join(query_parts)
                messages = [
                    {"role": "user", "content": query},
                ]

            if sig.return_annotation is str or not tool_output:
                response = chat_generate(
                    system=system,
                    messages=messages,
                    model_name=model,
                )

                if sig.return_annotation is str:
                    return response
                else:
                    type_adapter = TypeAdapter(sig.return_annotation)
                    return type_adapter.validate_json(response)

            else:
                return chat_generate_structured(
                    system=system,
                    messages=messages,
                    model_name=model,
                    output_type=sig.return_annotation,
                )

        wrapped = processor(wrapper)
        wrapped.prompt = prompt

        return wrapped

    return decorator(func) if func is not None else decorator
