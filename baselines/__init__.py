"""Baseline models for comparison.

Provides wrappers for GPT-2 and other reference models.
"""

from .gpt2 import load_gpt2, GPT2Wrapper
from .registry import get_model, list_models, MODEL_REGISTRY

__all__ = [
    "load_gpt2",
    "GPT2Wrapper",
    "get_model",
    "list_models",
    "MODEL_REGISTRY",
]
