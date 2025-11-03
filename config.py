"""Compatibility shim for legacy config imports."""

import warnings
from ai_rag.core.settings import Settings, settings

warnings.warn(
    "Importing from config is deprecated. Use ai_rag.core.settings instead.",
    DeprecationWarning,
    stacklevel=2,
)

RAGConfig = Settings
