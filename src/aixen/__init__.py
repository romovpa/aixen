from .apis.chat import chat_func as chat_fn
from .context import Context, File, get_context, save, set_context
from .context import processor as fn

try:
    from ._version import version as __version__  # noqa: F401
except ImportError:
    __version__ = None


__all__ = [
    "Context",
    "File",
    "fn",
    "chat_fn",
    "get_context",
    "set_context",
    "save",
]
