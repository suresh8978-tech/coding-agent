"""Shared utilities for coding agent tools."""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def safe_tool(func):
    """Decorator that wraps a tool function with a top-level exception handler.

    Apply this UNDER @tool so the safety net is baked into the function before
    LangChain converts it to a StructuredTool:

        @tool
        @safe_tool
        def my_tool(arg: str) -> str:
            ...

    On any unhandled exception the decorator catches the error, logs it, and
    returns a structured error string so the agent receives a meaningful message
    instead of crashing the graph.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.error(
                "Unhandled exception in tool '%s': %s",
                func.__name__,
                exc,
                exc_info=True,
            )
            return f"Error in tool '{func.__name__}': {exc}"
    return wrapper
