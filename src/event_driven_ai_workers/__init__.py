"""event-driven-ai-workers: ARQ/Redis background workers for heavy AI workloads."""

from importlib.metadata import version

__version__: str = version("event-driven-ai-workers")

__all__ = ["__version__"]
