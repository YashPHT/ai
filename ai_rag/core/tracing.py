import functools
import time
from typing import Any, Callable

from .logging_config import get_logger

logger = get_logger(__name__)


def trace_execution(func: Callable) -> Callable:
    """Decorator to trace function execution time and log inputs/outputs."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        func_name = f"{func.__module__}.{func.__qualname__}"

        logger.debug(
            f"Executing {func_name}",
            extra={
                "function": func_name,
                "args": str(args)[:100],
                "kwargs": str(kwargs)[:100],
            },
        )

        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time

            logger.debug(
                f"Completed {func_name} in {elapsed:.4f}s",
                extra={
                    "function": func_name,
                    "elapsed_seconds": elapsed,
                },
            )
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                f"Failed {func_name} after {elapsed:.4f}s: {e}",
                extra={
                    "function": func_name,
                    "elapsed_seconds": elapsed,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise

    return wrapper


class TracingContext:
    """Context manager for tracing execution blocks."""

    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = 0.0

    def __enter__(self) -> "TracingContext":
        self.start_time = time.perf_counter()
        logger.debug(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = time.perf_counter() - self.start_time

        if exc_type is not None:
            logger.error(
                f"Failed {self.operation} after {elapsed:.4f}s: {exc_val}",
                extra={
                    "operation": self.operation,
                    "elapsed_seconds": elapsed,
                    "error": str(exc_val),
                },
            )
        else:
            logger.debug(
                f"Completed {self.operation} in {elapsed:.4f}s",
                extra={
                    "operation": self.operation,
                    "elapsed_seconds": elapsed,
                },
            )
