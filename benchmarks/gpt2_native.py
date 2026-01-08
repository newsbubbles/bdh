"""GPT-2 native perplexity evaluation.

Evaluates GPT-2 using its own BPE tokenizer for fair comparison.
This gives the "true" GPT-2 perplexity on a dataset.

Note: This is NOT directly comparable to BDH byte-level perplexity!
Byte perplexity and BPE perplexity measure different things:
- Byte PPL: How well the model predicts the next byte
- BPE PPL: How well the model predicts the next subword token

For fair comparison, use bits-per-byte (BPB) or bits-per-character (BPC).
"""

import time
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class GPT2PerplexityResult:
    """Result with both token perplexity and bits-per-byte."""
    model_name: str
    dataset_name: str
    split: str
    
    # Token-level metrics (GPT-2 native)
    token_perplexity: float
    token_loss: float
    tokens_evaluated: int
    
    # Byte-level metrics (for comparison with BDH)
    bits_per_byte: float
    bytes_in_text: int
    
    # Meta
    time_seconds: float
    timestamp: str
    model_params: int


class GPT2TextDataset(Dataset):
    """Dataset that tokenizes text with GPT-2 tokenizer."""
    
    def __init__(self, text: str, tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize entire text
        self.tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)
        self.text_bytes = len(text.encode('utf-8'))
        
        # Create non-overlapping blocks
        self.n_blocks = len(self.tokens) // (block_size + 1)
    
    def __len__(self) -> int:
        return self.n_blocks
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * (self.block_size + 1)
        chunk = self.tokens[start : start + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


@torch.no_grad()
def evaluate_gpt2_native(
    model_name: str = "gpt2",
    text: str = None,
    text_path: Path = None,
    split: str = "test",
    block_size: int = 1024,
    batch_size: int = 8,
    device: str = "auto",
    max_batches: Optional[int] = None,
) -> GPT2PerplexityResult:
    """Evaluate GPT-2 with native tokenization.
    
    Returns both token perplexity and bits-per-byte for comparison.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load text
    if text is None:
        if text_path is None:
            raise ValueError("Must provide text or text_path")
        text = Path(text_path).read_text(encoding='utf-8')
    
    # Load model and tokenizer
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    
    # Create dataset
    dataset = GPT2TextDataset(text, tokenizer, block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
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
        
        outputs = model(x, labels=y)
        loss = outputs.loss
        
        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        n_batches += 1
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    avg_loss = total_loss / total_tokens  # nats per token
    token_perplexity = math.exp(avg_loss)
    
    # Convert to bits per byte for comparison
    # Total bits = total_loss * log2(e) 
    # Bits per byte = total bits / bytes in text
    total_bits = total_loss / math.log(2)  # Convert nats to bits
    bits_per_byte = total_bits / dataset.text_bytes
    
    return GPT2PerplexityResult(
        model_name=model_name,
        dataset_name=str(text_path) if text_path else "custom",
        split=split,
        token_perplexity=token_perplexity,
        token_loss=avg_loss,
        tokens_evaluated=total_tokens,
        bits_per_byte=bits_per_byte,
        bytes_in_text=dataset.text_bytes,
        time_seconds=elapsed,
        timestamp=datetime.now().isoformat(),
        model_params=n_params,
    )


def compute_bdh_bits_per_byte(byte_loss: float) -> float:
    """Convert BDH byte-level cross-entropy loss to bits per byte.
    
    BDH loss is in nats (natural log). Convert to bits:
    bits = nats / log(2)
    
    Since BDH predicts one byte at a time, loss = bits per byte.
    """
    return byte_loss / math.log(2)


if __name__ == "__main__":
    # Test on Shakespeare
    result = evaluate_gpt2_native(
        model_name="gpt2",
        text_path=Path("data/shakespeare.txt"),
        block_size=512,
        batch_size=4,
        max_batches=10,
    )
    
    print(f"\nGPT-2 Results:")
    print(f"  Token Perplexity: {result.token_perplexity:.2f}")
    print(f"  Bits per Byte: {result.bits_per_byte:.4f}")
    print(f"  Tokens evaluated: {result.tokens_evaluated:,}")
    print(f"  Time: {result.time_seconds:.1f}s")
