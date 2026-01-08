"""Perplexity evaluation for language models.

Supports both BDH models and HuggingFace models (GPT-2, etc.)
"""

import time
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .datasets import BenchmarkDataset, load_benchmark_dataset


@dataclass
class PerplexityResult:
    """Result of perplexity evaluation."""
    model_name: str
    dataset_name: str
    split: str  # 'train', 'val', 'test'
    perplexity: float
    loss: float
    tokens_evaluated: int
    time_seconds: float
    timestamp: str
    
    # Model info
    model_params: Optional[int] = None
    model_config: Optional[dict] = None
    
    # Evaluation config
    block_size: int = 512
    batch_size: int = 32
    device: str = "cuda"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __repr__(self) -> str:
        return (
            f"PerplexityResult({self.model_name} on {self.dataset_name}/{self.split}: "
            f"ppl={self.perplexity:.2f}, loss={self.loss:.4f}, "
            f"{self.tokens_evaluated:,} tokens in {self.time_seconds:.1f}s)"
        )


class BlockDataset(Dataset):
    """Dataset that yields fixed-size blocks for perplexity evaluation."""
    
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size
        # Non-overlapping blocks for clean evaluation
        self.n_blocks = len(data) // (block_size + 1)
    
    def __len__(self) -> int:
        return self.n_blocks
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * (self.block_size + 1)
        chunk = self.data[start : start + self.block_size + 1]
        x = chunk[:-1]  # Input
        y = chunk[1:]   # Target
        return x, y


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataset: BenchmarkDataset | str,
    split: str = "test",
    block_size: int = 512,
    batch_size: int = 32,
    device: str = "auto",
    model_name: Optional[str] = None,
    max_batches: Optional[int] = None,
    forward_fn: Optional[Callable] = None,
) -> PerplexityResult:
    """Evaluate perplexity of a model on a dataset.
    
    Args:
        model: PyTorch model with forward(x, targets) -> (logits, loss)
        dataset: BenchmarkDataset or dataset name string
        split: Which split to evaluate ('train', 'val', 'test')
        block_size: Context length for evaluation
        batch_size: Batch size
        device: Device to use ('auto', 'cuda', 'cpu', 'mps')
        model_name: Name for logging (inferred if not provided)
        max_batches: Limit evaluation to N batches (for quick testing)
        forward_fn: Custom forward function(model, x, y) -> loss
                   Use for models with different interfaces (e.g., HuggingFace)
    
    Returns:
        PerplexityResult with metrics
    """
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset if string
    if isinstance(dataset, str):
        dataset = load_benchmark_dataset(dataset)
    
    # Get the right split
    if split == "train":
        data = dataset.train_data
    elif split == "val":
        data = dataset.val_data
    elif split == "test":
        data = dataset.test_data
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Create dataloader
    block_dataset = BlockDataset(data, block_size)
    dataloader = DataLoader(
        block_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    
    # Infer model name
    if model_name is None:
        model_name = model.__class__.__name__
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    # Get config if available
    model_config = None
    if hasattr(model, 'config'):
        cfg = model.config
        if hasattr(cfg, '__dict__'):
            model_config = {k: v for k, v in cfg.__dict__.items() 
                          if not k.startswith('_')}
    
    # Move model to device and eval mode
    model = model.to(device)
    model.eval()
    
    # Evaluate
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        x = x.to(device)
        y = y.to(device)
        
        # Get loss
        if forward_fn is not None:
            loss = forward_fn(model, x, y)
        else:
            # Default: assume model returns (logits, loss)
            _, loss = model(x, y)
        
        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        n_batches += 1
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return PerplexityResult(
        model_name=model_name,
        dataset_name=dataset.name,
        split=split,
        perplexity=perplexity,
        loss=avg_loss,
        tokens_evaluated=total_tokens,
        time_seconds=elapsed,
        timestamp=datetime.now().isoformat(),
        model_params=n_params,
        model_config=model_config,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
    )


def save_results(
    results: list[PerplexityResult],
    output_dir: Path,
    name: str = "perplexity",
) -> None:
    """Save results to JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON (full detail)
    json_path = output_dir / f"{name}.json"
    with open(json_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"Saved {json_path}")
    
    # CSV (for plotting)
    csv_path = output_dir / f"{name}.csv"
    with open(csv_path, 'w') as f:
        # Header
        fields = ['model_name', 'dataset_name', 'split', 'perplexity', 
                  'loss', 'tokens_evaluated', 'model_params', 'time_seconds']
        f.write(','.join(fields) + '\n')
        # Data
        for r in results:
            d = r.to_dict()
            row = [str(d.get(field, '')) for field in fields]
            f.write(','.join(row) + '\n')
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from bdh import BDH, BDHConfig
    
    # Create tiny model for testing
    config = BDHConfig(
        n_layer=2,
        n_embd=64,
        n_head=2,
        vocab_size=256,
    )
    model = BDH(config)
    
    # Test on Shakespeare
    result = evaluate_perplexity(
        model,
        "shakespeare",
        split="val",
        max_batches=5,
        model_name="bdh_tiny_test",
    )
    print(result)
