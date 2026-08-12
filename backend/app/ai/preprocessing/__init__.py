"""Versioned fabric-image preprocessing.

The legacy :mod:`backend.app.ai.preprocess` module remains the v1
implementation.  This package contains the independently configurable v2
pipeline.
"""

from backend.app.ai.preprocessing.config import PreprocessConfig
from backend.app.ai.preprocessing.pipeline import (
    BatchPreprocessSummary,
    PreprocessResult,
    preprocess_fabric_image,
    preprocess_path,
    process_directory,
)

__all__ = [
    "BatchPreprocessSummary",
    "PreprocessConfig",
    "PreprocessResult",
    "preprocess_fabric_image",
    "preprocess_path",
    "process_directory",
]
