import contextvars
import inspect
import json
import os
import shutil
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
from dotenv import dotenv_values
from loguru import logger
from pydantic import BaseModel, RootModel
from slugify import slugify

current_context = contextvars.ContextVar("current_context", default=None)
global_context: "Context | None" = None


class File(RootModel[str]):
    @property
    def id(self) -> str:
        return self.root

    def __str__(self) -> str:
        return self.root

    def __repr__(self) -> str:
        return f"File(id={repr(self.root)})"

    @property
    def local_path(self) -> Path:
        context = Context.get_current()
        return context.local_path(self.id)

    def open(self, *args, **kwargs) -> Any:
        context = Context.get_current()
        return context.open(self.id, *args, **kwargs)


class Context:
    """
    External tools used by the agent.

    This implementation will evolve to cover the vision below.

    Purpose of the context:
    - Keep environment variables and secrets (e.g. API keys)
    - Manage file assets and remote storage
    - Track usage and costs for APIs
    - Caching (potentially)

    ## Creating a Context

    >>> context = Context()

    Specifying a name which will be used in the unique context ID:
    >>> context = Context(name="My Pipeline")

    Providing a specific ID (beware of collisions):
    >>> context = Context(id=f"{datetime.now():%d-%H-%M-%S}-my-pipeline")

    Code block with a context:
    >>> with Context() as context:
    ...     output = pipeline(input)

    Setting a global context:
    >>> Context.set_current(context)
    >>> output = pipeline(input)

    ## Environment

    The environment is for host-specific configuration and secrets.
    It includes anything that changes when
    moving code from one machine to another or from staging to production.
    Examples include:
    - API keys
    - Paths

    The environment must NOT contain parameters that guide the pipeline logic.
    Replacing the environment under the same
    settings should lead to identical results of a pipeline (given that the environment
    provides working keys and is subject to randomness in the pipeline).

    By default, the environment is loaded from a `.env` file in the project directory.

    Environment variable should not be exposed externally.
    We assume that it is safe to keep secrets in the environment.


    Usage:
    >>> context.environment['OPENAI_API_KEY']
    'sk-1234567890abcdef'
    >>> context.environment['CACHE_PATH']
    '/path/to/cache'

    ## Settings

    Settings provide global parameters for the pipeline,
    and used to configure the behavior of the pipeline.
    Examples include:
    - default LLM model of choice for the pipeline
    - a Voice ID for text-to-speech

    Setting can be exposed externally. You MUST NOT store secrets in the settings.

    Usage:
    >>> context.settings['chat.default_model']
    'openai:gpt-4-turbo'
    >>> context.settings['elevenlabs.voice_id']
    'josh'

    """

    id: str
    environment: dict[str, str | None]
    settings: dict[str, Any]
    context_dir: str
    calls: list[dict[str, Any]]

    _token: contextvars.Token

    def __init__(
        self,
        id: str | None = None,
        name: str | None = None,
        environment: dict[str, str | None] | str | None = None,
        settings: dict[str, Any] | str | None = None,
        keep_files: bool = False,
    ):
        # Context management
        self._token = None

        # Environment (secrets)
        # First loaded from os.environ, then overridden by .env file or provided dict
        self.environment = {}
        self.environment.update(os.environ)
        if environment is None:
            self.environment.update(dotenv_values())
        elif isinstance(environment, str):
            self.environment.update(dotenv_values(environment))
        else:
            self.environment.update(environment)

        # Settings
        self.settings = {}
        if isinstance(settings, str):
            with open(settings) as file:
                data = json.load(file)
            self.settings.update(data)
        elif settings is not None:
            self.settings.update(settings)

        # Unique Context ID
        # For readability, it is generated with a timestamp and a label.
        # A random string is added to avoid collisions.
        if id is None:
            label_part = ""
            if name is not None:
                label_part = "-" + slugify(
                    name, max_length=50, word_boundary=True, save_order=True
                )
            timestamp_part = datetime.now().strftime("%Y%m%d%H%M%S")
            random_part = str(uuid.uuid4())[:8]
            id = f"{timestamp_part}{label_part}-{random_part}"
        self.id = id

        # File manager
        self.keep_files = keep_files
        cache_path = self.environment.get("CACHE_PATH")  # TODO Rename
        self._temp_dir = None
        if cache_path:
            self.context_dir = os.path.join(cache_path, self.id)
            os.makedirs(self.context_dir, exist_ok=True)
        else:
            self._temp_dir = TemporaryDirectory(
                prefix="context-",
                suffix=f"-{self.id}",
            )
            self.context_dir = self._temp_dir.name
        self._file_index = 0
        logger.debug(f"Created temporary directory at {self.context_dir}")

        # Tracking
        self.calls = []

    def __del__(self):
        if not self.keep_files:
            logger.debug(f"Cleaning up temporary directory at {self.context_dir}")
            if self._temp_dir is not None:
                self._temp_dir.cleanup()
            else:
                shutil.rmtree(self.context_dir)

    def __enter__(self):
        self._token = current_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_context.reset(self._token)

    @staticmethod
    def get_current() -> "Context":
        """
        DEPRECATED Use ai.get_context() instead
        """
        context = current_context.get()
        if context is None:
            if global_context is not None:
                return global_context
            raise RuntimeError("No context is currently set")
        return context

    @staticmethod
    def set_current(context):
        """
        DEPRECATED Use ai.set_context() instead
        """
        global global_context
        global_context = context
        current_context.set(context)

    def save(
        self,
        obj: Any | None = None,
        name: str | None = None,
        ext: str | None = None,
        url: str | None = None,
        share: bool = False,
    ) -> File:
        """
        DEPRECATED: Use ai.save() instead.
        Save an object to the context directory. Returns relative file path (file ID).
        """
        assert not (
            obj is not None and url is not None
        ), "Cannot save both object and URL"

        if name is None:
            file_ext = f".{ext}"
            if ext is None:
                if obj is None:
                    file_ext = ""
                elif isinstance(obj, bytes):
                    file_ext = ".bin"
                elif isinstance(obj, str):
                    file_ext = ".txt"
                else:
                    file_ext = ".json"

            parent = get_parent_processor_call()
            file_label = f"_{parent['function']}" if parent else ""
            name = f"{self._file_index:03d}{file_label}{file_ext}"
            self._file_index += 1

        file_path = os.path.join(self.context_dir, name)
        file_id = os.path.join(self.id, name)

        file_base_dir, _ = os.path.split(file_path)
        os.makedirs(file_base_dir, exist_ok=True)

        if obj is None:
            if url is not None:
                urlretrieve(url, file_path)
            else:
                # No object to save, just ensure the path exists
                pass
        elif isinstance(obj, bytes):
            with open(file_path, "wb") as file:
                file.write(obj)
        elif isinstance(obj, str):
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(obj)
        else:
            with open(file_path, "w") as file:
                json.dump(
                    obj,
                    file,
                    ensure_ascii=False,
                    indent=4,
                    cls=MixedDictPydanticJSONEncoder,
                )

        # TODO Upload file to storage if share=True
        if share:
            raise NotImplementedError("Sharing files is not implemented")
        return File(file_id)

    def local_path(self, file_id: str) -> Path:
        """
        DEPRECATED: Use the File object instead.
        Get the local file path from the file ID.
        """
        separator_index = file_id.find(os.sep)
        parent_dir = file_id[:separator_index] if separator_index >= 0 else ""
        if parent_dir != self.id:
            raise NotImplementedError(
                "Opening files from other contexts is not supported"
            )
        relative_path = (
            file_id[separator_index + 1 :] if separator_index >= 0 else file_id
        )
        return Path(self.context_dir) / relative_path

    def open(self, file_id: str, *args, **kwargs) -> Any:
        """
        DEPRECATED: Use the File object instead.
        Open a file from the project directory.
        """
        file_path = self.local_path(file_id)
        return open(file_path, *args, **kwargs)

    @staticmethod
    def record(**kwargs) -> None:
        """
        Record a key-value pair in the context.
        """
        parent = get_parent_processor_call()
        for key, value in kwargs.items():
            if value is not None:
                record = {
                    "type": key,
                    "value": value,
                    "time": time.time(),
                }
                parent["records"].append(record)

    @property
    def usage_cost_usd(self) -> float:
        total_cost_usd = 0.0
        for call in self.calls:
            for record in call["records"]:
                if record["type"] == "usage":
                    total_cost_usd += record["value"].cost_usd
        return total_cost_usd


class MixedDictPydanticJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, bytes):
            return "[bytes]"
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)


class Usage(BaseModel):
    cost_usd: float | None = None


def urlretrieve(url: str, filename: str) -> None:
    with httpx.stream("GET", url) as response:
        response.raise_for_status()
        with open(filename, "wb") as fout:
            for chunk in response.iter_bytes(chunk_size=8192):
                fout.write(chunk)


def get_parent_processor_call(
    current_call_id: str | None = None,
) -> dict[str, Any] | None:
    frame = inspect.currentframe()
    while frame:
        if frame.f_locals.get("___i_am_a_processor_wrapper"):
            if current_call_id is None or current_call_id != frame.f_locals.get(
                "call_id"
            ):
                return frame.f_locals.get("call_record")
        frame = frame.f_back
    return None


def processor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind arguments to their names
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        named_args = bound_args.arguments

        # Get the context object
        context = Context.get_current()

        # Assign a call ID
        ___i_am_a_processor_wrapper = True  # noqa: F841
        call_id = str(uuid.uuid4())

        # Find the parent call
        parent = get_parent_processor_call(call_id)

        call_index = len(context.calls)
        call_depth = parent["depth"] + 1 if parent else 0
        call_record = {
            "index": call_index,
            "id": call_id,
            "parent": parent["id"] if parent else None,
            "depth": call_depth,
            "module": func.__module__,
            "function": func.__name__,
            "input": named_args,
            "output": None,
            "records": [],
        }
        context.calls.append(call_record)

        call_file = f'calls/{call_index:04d}_{"_" * call_depth}{func.__name__}.json'

        try:
            context.save(call_record, name=call_file)
        except Exception as e:
            logger.error(f"Error saving call record: {e}")

        # Run the function
        call_record["time_start"] = time.time()

        try:
            output = func(*args, **kwargs)
            call_record["output"] = output
        except Exception as e:
            call_record["exception"] = str(e)
            raise e
        finally:
            call_record["time_end"] = time.time()

            try:
                context.save(call_record, name=call_file)
            except Exception as e:
                logger.error(f"Error saving call record: {e}")

        return output

    return wrapper


def save(
    obj: Any | None = None,
    name: str | None = None,
    ext: str | None = None,
    url: str | None = None,
    share: bool = False,
) -> File:
    """
    Save an object to the context directory. Returns relative file path (file ID).
    """
    context = Context.get_current()
    return context.save(obj=obj, name=name, ext=ext, url=url, share=share)


def get_context() -> Context:
    """
    Returns the current context.
    """
    context = current_context.get()
    if context is None:
        if global_context is not None:
            return global_context
        raise RuntimeError("No context is currently set")
    return context


def set_context(context: Context) -> None:
    global global_context
    global_context = context
    current_context.set(context)
