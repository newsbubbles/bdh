"""GPT-2 baseline wrapper for fair comparison with BDH.

Provides two evaluation modes:
1. Native tokenization (GPT-2's BPE) - standard benchmarks
2. Byte-level (256 vocab) - fair comparison with BDH

Note: Byte-level GPT-2 will have worse perplexity since it wasn't
trained that way, but it's the only fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass 
class GPT2Config:
    """GPT-2 configuration for reference."""
    model_name: str
    n_layer: int
    n_embd: int
    n_head: int
    vocab_size: int
    n_params: int


# GPT-2 model variants
GPT2_CONFIGS = {
    "gpt2": GPT2Config(
        model_name="gpt2",
        n_layer=12,
        n_embd=768,
        n_head=12,
        vocab_size=50257,
        n_params=124_000_000,
    ),
    "gpt2-medium": GPT2Config(
        model_name="gpt2-medium",
        n_layer=24,
        n_embd=1024,
        n_head=16,
        vocab_size=50257,
        n_params=355_000_000,
    ),
    "gpt2-large": GPT2Config(
        model_name="gpt2-large",
        n_layer=36,
        n_embd=1280,
        n_head=20,
        vocab_size=50257,
        n_params=774_000_000,
    ),
    "gpt2-xl": GPT2Config(
        model_name="gpt2-xl",
        n_layer=48,
        n_embd=1600,
        n_head=25,
        vocab_size=50257,
        n_params=1_500_000_000,
    ),
}


class GPT2Wrapper(nn.Module):
    """Wrapper around HuggingFace GPT-2 for benchmarking.
    
    Provides a consistent interface matching BDH:
        forward(idx, targets=None) -> (logits, loss)
    
    Supports two modes:
    - 'native': Use GPT-2's BPE tokenizer (standard benchmark)
    - 'byte': Use byte-level tokens (fair comparison with BDH)
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        mode: Literal["native", "byte"] = "native",
        device: str = "auto",
    ):
        super().__init__()
        
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )
        
        self.model_name = model_name
        self.mode = mode
        self.gpt2_config = GPT2_CONFIGS.get(model_name)
        
        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        
        print(f"Loading {model_name} from HuggingFace...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # For byte mode, we need a projection layer
        # This is NOT a fair comparison since GPT-2 wasn't trained on bytes
        # But it lets us compare on the same data
        if mode == "byte":
            print("Warning: Byte mode - GPT-2 was not trained on bytes!")
            print("This comparison shows architecture efficiency, not training.")
            # We'll compute loss differently in byte mode
        
        self.model.to(device)
        self.model.eval()
    
    @property
    def config(self):
        return self.gpt2_config
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass matching BDH interface.
        
        Args:
            idx: Input token IDs (B, T)
            targets: Target token IDs (B, T) for loss computation
            
        Returns:
            (logits, loss) tuple
        """
        if self.mode == "native":
            # Standard GPT-2 forward
            outputs = self.model(idx, labels=targets)
            logits = outputs.logits
            loss = outputs.loss
        else:
            # Byte mode: idx is bytes (0-255)
            # We can't directly use GPT-2 on bytes since vocab is different
            # Instead, we decode bytes to text, re-tokenize, and compute loss
            # This is slow but fair for comparison
            raise NotImplementedError(
                "Byte mode requires special handling. "
                "Use native mode or the byte_forward_fn helper."
            )
        
        return logits, loss
    
    def parameters(self):
        return self.model.parameters()
    
    def to(self, device):
        self.model.to(device)
        self._device = device
        return self
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self, mode=True):
        self.model.train(mode)
        return self


def gpt2_native_forward(model: GPT2Wrapper, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Forward function for GPT-2 with native tokenization.
    
    Use this with evaluate_perplexity's forward_fn parameter.
    Note: x and y should be GPT-2 BPE tokens, not bytes!
    """
    outputs = model.model(x, labels=y)
    return outputs.loss


def load_gpt2(
    model_name: str = "gpt2",
    device: str = "auto",
) -> GPT2Wrapper:
    """Load a GPT-2 model for benchmarking.
    
    Args:
        model_name: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        device: Device to load on
        
    Returns:
        GPT2Wrapper instance
    """
    return GPT2Wrapper(model_name=model_name, mode="native", device=device)


if __name__ == "__main__":
    # Test loading
    model = load_gpt2("gpt2")
    print(f"Loaded {model.model_name}")
    print(f"Config: {model.config}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
