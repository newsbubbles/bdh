"""BDH Benchmarking Suite.

Tools for evaluating and comparing language model architectures.
"""

from .perplexity import evaluate_perplexity, PerplexityResult
from .datasets import load_benchmark_dataset

__all__ = [
    "evaluate_perplexity",
    "PerplexityResult", 
    "load_benchmark_dataset",
]
