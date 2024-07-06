import importlib.resources as pkg_resources
import inspect
import json
from functools import wraps
from typing import Callable, Optional, TypeVar

import httpx
import pydantic
from openai import OpenAI
from openai.types.chat import ChatCompletion

from aixen.context import (
    Context,
    MixedDictPydanticJSONEncoder,
    Usage,
    get_context,
    processor,
)

DEFAULT_MODEL = "gpt-4o"


# TODO: Add type hints and better structure after the API is finalized
PROVIDERS = {
    "openai": {
        "client": OpenAI,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    "openrouter": {
        "client": OpenAI,
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "groq": {
        "client": OpenAI,
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "fireworks": {
        "client": OpenAI,
        "api_key_env": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1",
    },
}


DEFAULT_PROVIDER = "openai"  # TODO Maybe change to "openrouter"


CHAT_COMPLETION_PRICING = {}
with pkg_resources.open_text(__package__, "chat_pricing.json") as f:
    CHAT_COMPLETION_PRICING = json.load(f)


class ChatCompletionUsage(Usage):
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@processor
def chat_generate(
    system: str,
    messages: list[dict],
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """
    Prompt a chat model (multi-modal LLM) to generate a response.

    Settings:
    ```js
    {
        # Model aliases
        "chat.models": {
            "default": "gpt-4o",
            "fast": "groq:llama-3-7b",
            "smart": "openai:gpt-4-turbo",
            "vision": "openai:gpt-4-turbo-preview",
            "gemini": "openrouter:google/gemini-flash-1.5",
        }
    }
    ```
    """
    context = get_context()
    model_name = _resolve_model(context, model_name)
    client, model_name = _resolve_provider(context, model_name)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": system}, *messages],
        temperature=temperature,
    )
    if response.usage:
        context.record(usage=_estimate_usage_cost(response))
    return response.choices[0].message.content


T = TypeVar("T")


# TODO Merge with chat_generate
@processor
def chat_generate_structured(
    system: str,
    messages: list[dict],
    output_type: type[T],
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> T:
    context = get_context()
    model_name = _resolve_model(context, model_name)
    client, model_name = _resolve_provider(context, model_name)
    Answer = pydantic.create_model("Answer", answer=(output_type, ...))
    response = client.chat.completions.create(
        model=_resolve_model(context, model_name),
        messages=[{"role": "system", "content": system}, *messages],
        temperature=temperature,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "provide_answer",
                    "description": "Provide an answer based on the messages",
                    "parameters": Answer.model_json_schema(),
                },
            }
        ],
        tool_choice="required",
    )
    if response.usage:
        context.record(usage=_estimate_usage_cost(response))
    result = Answer.model_validate_json(
        response.choices[0].message.tool_calls[0].function.arguments
    )
    return result.answer


def chat_func(func: Callable, model: Optional[str] = None) -> Callable:
    """
    Decorator that converts creates LLM-based processor based on function signature
    and docstring.
    """
    sig = inspect.signature(func)
    output_type = sig.return_annotation

    @wraps(func)
    def wrapper(*args, **kwargs):
        system_prompt = func.__doc__

        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        query_parts = []
        for name, value in bound_args.arguments.items():
            if isinstance(value, str):
                query_parts.append(f"{name}: {value}")
            else:
                data = json.dumps(value, cls=MixedDictPydanticJSONEncoder)
                query_parts.append(f"{name}: {data}")
        query = "\n\n".join(query_parts)

        if issubclass(output_type, str):
            return chat_generate(
                system=system_prompt,
                messages=[{"role": "user", "content": query}],
                model_name=model,
            )
        else:
            return chat_generate_structured(
                system=system_prompt,
                messages=[{"role": "user", "content": query}],
                model_name=model,
                output_type=output_type,
            )

    return processor(wrapper)


def _resolve_provider(context: Context, model_name: str) -> tuple[OpenAI, str]:
    """
    Creates a client from the model_name.
    User can specify provider in form of a schema-like prefix.
    >>> model_name = "openai:gpt-4o"
    >>> client, model = _resolve_provider(context, model_name)
    (OpenAI(...), "gpt-4o")
    """
    schema_delim = model_name.find(":")
    if schema_delim >= 0:
        provider_name = model_name[:schema_delim]
        model_name = model_name[schema_delim + 1 :]
    else:
        provider_name = DEFAULT_PROVIDER

    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown chat model provider: {provider_name}")

    provider = PROVIDERS[provider_name]
    client = provider["client"](
        api_key=context.environment.get(provider["api_key_env"]),
        base_url=provider["base_url"],
    )
    return client, model_name


def _resolve_model(context: Context, model_name: Optional[str] = None) -> str:
    """
    Resolves the model name using the aliases from the settings.
    >>> settings = {
    ...     "chat.models": {
    ...         "default_alias": "gpt-4o",
    ...         "gemini_alias": "openrouter:google/gemini-flash-1.5",
    ...     },
    ... }
    >>> context = Context(settings=settings)
    >>> _resolve_model(context, "gemini_alias")
    "openrouter:google/gemini-flash-1.5"
    """
    aliases = context.settings.get("chat.models", {})
    default_model = aliases.get("default", DEFAULT_MODEL)
    if model_name is None:
        return default_model
    if model_name in aliases:
        return aliases[model_name]
    return model_name


def _fetch_openrouter_pricing():
    data = httpx.get("https://openrouter.ai/api/v1/models").json()

    pricing = {}

    for model_card in data["data"]:
        model = model_card["id"]
        model_pricing = {
            key: float(value) if value != "-1" else None
            for key, value in model_card["pricing"].items()
        }
        pricing[model] = model_pricing

        separator = model.find("/")
        if separator != -1:
            alternative_name = model[separator + 1 :]
            pricing[alternative_name] = model_pricing

    with open("src/aixen/apis/chat_pricing.json", "w") as f:
        json.dump(pricing, f, indent=4)


def _estimate_usage_cost(response: ChatCompletion) -> ChatCompletionUsage | None:
    """
    Estimates the cost of the usage based on the response.
    """
    if response.usage is not None and response.model in CHAT_COMPLETION_PRICING:
        model_pricing = CHAT_COMPLETION_PRICING[response.model]

        cost_usd = 0.0

        if model_pricing["prompt"] is not None:
            cost_usd += model_pricing["prompt"] * response.usage.prompt_tokens
        if model_pricing["completion"] is not None:
            cost_usd += model_pricing["completion"] * response.usage.completion_tokens
        if model_pricing["request"] is not None:
            cost_usd += model_pricing["request"]

        return ChatCompletionUsage(
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cost_usd=cost_usd,
        )

    return None
