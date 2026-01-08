"""Dataset loaders for benchmarking.

Supports WikiText-2, Penn Treebank, and custom text files.
All text is converted to byte-level tokens (0-255) for BDH compatibility.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class BenchmarkDataset:
    """Container for benchmark dataset."""
    name: str
    train_data: torch.Tensor  # (n_tokens,) dtype=torch.long
    val_data: torch.Tensor
    test_data: torch.Tensor
    vocab_size: int = 256  # Byte-level
    
    @property
    def train_tokens(self) -> int:
        return len(self.train_data)
    
    @property 
    def val_tokens(self) -> int:
        return len(self.val_data)
    
    @property
    def test_tokens(self) -> int:
        return len(self.test_data)
    
    def __repr__(self) -> str:
        return (
            f"BenchmarkDataset({self.name}, "
            f"train={self.train_tokens:,}, "
            f"val={self.val_tokens:,}, "
            f"test={self.test_tokens:,})"
        )


def text_to_bytes(text: str) -> torch.Tensor:
    """Convert text to byte-level tensor."""
    return torch.tensor(list(text.encode('utf-8')), dtype=torch.long)


def load_wikitext2(data_dir: Optional[Path] = None) -> BenchmarkDataset:
    """Load WikiText-2 dataset.
    
    Downloads from HuggingFace datasets if not cached.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install datasets: pip install datasets"
        )
    
    print("Loading WikiText-2 from HuggingFace...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Concatenate all text in each split
    train_text = "\n".join(dataset["train"]["text"])
    val_text = "\n".join(dataset["validation"]["text"])
    test_text = "\n".join(dataset["test"]["text"])
    
    return BenchmarkDataset(
        name="wikitext2",
        train_data=text_to_bytes(train_text),
        val_data=text_to_bytes(val_text),
        test_data=text_to_bytes(test_text),
    )


def load_ptb(data_dir: Optional[Path] = None) -> BenchmarkDataset:
    """Load Penn Treebank dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install datasets: pip install datasets"
        )
    
    print("Loading Penn Treebank from HuggingFace...")
    dataset = load_dataset("ptb_text_only")
    
    train_text = "\n".join(dataset["train"]["sentence"])
    val_text = "\n".join(dataset["validation"]["sentence"])
    test_text = "\n".join(dataset["test"]["sentence"])
    
    return BenchmarkDataset(
        name="ptb",
        train_data=text_to_bytes(train_text),
        val_data=text_to_bytes(val_text),
        test_data=text_to_bytes(test_text),
    )


def load_shakespeare(data_dir: Path = Path("data")) -> BenchmarkDataset:
    """Load Shakespeare dataset from local file."""
    # Try both possible filenames
    input_file = data_dir / "shakespeare.txt"
    if not input_file.exists():
        input_file = data_dir / "input.txt"
    if not input_file.exists():
        # Try to download
        import urllib.request
        data_dir.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Shakespeare from {url}...")
        input_file = data_dir / "shakespeare.txt"
        urllib.request.urlretrieve(url, input_file)
        print(f"Saved to {input_file}")
    
    text = input_file.read_text(encoding='utf-8')
    data = text_to_bytes(text)
    
    # Standard 90/5/5 split
    n = len(data)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)
    
    return BenchmarkDataset(
        name="shakespeare",
        train_data=data[:train_end],
        val_data=data[train_end:val_end],
        test_data=data[val_end:],
    )


def load_custom(path: Path) -> BenchmarkDataset:
    """Load custom text file."""
    text = path.read_text(encoding='utf-8')
    data = text_to_bytes(text)
    
    n = len(data)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)
    
    return BenchmarkDataset(
        name=path.stem,
        train_data=data[:train_end],
        val_data=data[train_end:val_end],
        test_data=data[val_end:],
    )


DATASET_REGISTRY = {
    "wikitext2": load_wikitext2,
    "ptb": load_ptb,
    "shakespeare": load_shakespeare,
}


def load_benchmark_dataset(
    name: str,
    data_dir: Optional[Path] = None,
) -> BenchmarkDataset:
    """Load a benchmark dataset by name.
    
    Args:
        name: Dataset name ('wikitext2', 'ptb', 'shakespeare') or path to .txt
        data_dir: Optional data directory
        
    Returns:
        BenchmarkDataset with train/val/test splits
    """
    if name in DATASET_REGISTRY:
        loader = DATASET_REGISTRY[name]
        if name == "shakespeare":
            return loader(data_dir or Path("data"))
        return loader(data_dir)
    
    # Try as file path
    path = Path(name)
    if path.exists() and path.suffix == ".txt":
        return load_custom(path)
    
    raise ValueError(
        f"Unknown dataset: {name}. "
        f"Available: {list(DATASET_REGISTRY.keys())} or path to .txt file"
    )


if __name__ == "__main__":
    # Test loading
    for name in ["shakespeare", "wikitext2"]:
        try:
            ds = load_benchmark_dataset(name)
            print(ds)
        except Exception as e:
            print(f"{name}: {e}")
