"""Model registry for benchmarking.

Provides a unified interface to load any model by name.
"""

import sys
from pathlib import Path
from typing import Optional, Callable
import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_bdh(checkpoint: Optional[str] = None, **config_overrides) -> nn.Module:
    """Load BDH model, optionally from checkpoint."""
    from bdh import BDH, BDHConfig
    
    if checkpoint:
        # Load from checkpoint
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model_config" in ckpt:
            config = BDHConfig(**ckpt["model_config"])
        else:
            config = BDHConfig(**config_overrides)
        model = BDH(config)
        # Support both checkpoint formats
        state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
        model.load_state_dict(ckpt[state_dict_key])
        return model
    else:
        # Create fresh model
        config = BDHConfig(**config_overrides)
        return BDH(config)


def _load_gpt2(model_name: str = "gpt2", **kwargs) -> nn.Module:
    """Load GPT-2 from HuggingFace."""
    from .gpt2 import load_gpt2
    return load_gpt2(model_name=model_name, **kwargs)


# Registry maps names to loader functions
MODEL_REGISTRY: dict[str, Callable] = {
    # BDH variants
    "bdh": _load_bdh,
    "bdh_base": lambda **kw: _load_bdh(
        n_layer=6, n_embd=256, n_head=4, **kw
    ),
    "bdh_small": lambda **kw: _load_bdh(
        n_layer=4, n_embd=128, n_head=4, **kw
    ),
    "bdh_medium": lambda **kw: _load_bdh(
        n_layer=8, n_embd=384, n_head=6, **kw
    ),
    
    # GPT-2 variants
    "gpt2": lambda **kw: _load_gpt2("gpt2", **kw),
    "gpt2_124m": lambda **kw: _load_gpt2("gpt2", **kw),
    "gpt2_medium": lambda **kw: _load_gpt2("gpt2-medium", **kw),
    "gpt2_355m": lambda **kw: _load_gpt2("gpt2-medium", **kw),
    "gpt2_large": lambda **kw: _load_gpt2("gpt2-large", **kw),
    "gpt2_xl": lambda **kw: _load_gpt2("gpt2-xl", **kw),
}


def get_model(name: str, **kwargs) -> nn.Module:
    """Load a model by name.
    
    Args:
        name: Model name from registry, or path to BDH checkpoint
        **kwargs: Additional arguments passed to loader
        
    Returns:
        Loaded model
        
    Examples:
        >>> model = get_model("bdh_base")
        >>> model = get_model("gpt2_124m")
        >>> model = get_model("checkpoints/best.pt")  # Load BDH checkpoint
    """
    # Check if it's a checkpoint path
    if name.endswith(".pt"):
        return _load_bdh(checkpoint=name, **kwargs)
    
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {name}. "
            f"Available: {available} or path to .pt checkpoint"
        )
    
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())


if __name__ == "__main__":
    print("Available models:")
    for name in list_models():
        print(f"  - {name}")
