#!/usr/bin/env python3
# Copyright Pathway Technology, Inc.
# Inference script for Baby Dragon Hatchling models

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import bdh


def load_model(checkpoint_path: str, device: str = "auto") -> tuple[bdh.BDH, bdh.BDHConfig, dict]:
    """
    Load a trained BDH model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on (auto, cuda, cpu, mps)
    
    Returns:
        Tuple of (model, config, checkpoint_info)
    """
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device = torch.device(device)
    print(f"Loading model on {device}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct config
    model_config = bdh.BDHConfig(**checkpoint["model_config"])
    
    # Create and load model
    model = bdh.BDH(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Info for display
    info = {
        "step": checkpoint.get("step", "unknown"),
        "val_loss": checkpoint.get("val_loss", "unknown"),
        "best_val_loss": checkpoint.get("best_val_loss", "unknown"),
        "params": sum(p.numel() for p in model.parameters()),
    }
    
    print(f"Loaded model: {info['params']:,} parameters")
    print(f"  Trained for {info['step']} steps")
    print(f"  Val loss: {info['val_loss']}")
    
    return model, model_config, info


@torch.no_grad()
def generate(
    model: bdh.BDH,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = None,
    top_p: float = None,
    device: torch.device = None,
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: Trained BDH model
        prompt: Text prompt to continue
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (None to disable)
        top_p: Nucleus sampling threshold (None to disable)
        device: Device to run on
    
    Returns:
        Generated text including the prompt
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Encode prompt as bytes
    prompt_bytes = prompt.encode("utf-8")
    tokens = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
    
    # Generate
    for _ in range(max_tokens):
        # Get logits for last position
        logits, _ = model(tokens)
        logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        
        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    
    # Decode
    output_bytes = bytes(tokens[0].tolist())
    return output_bytes.decode("utf-8", errors="replace")


def interactive_mode(model: bdh.BDH, args):
    """Interactive generation mode."""
    print("\n" + "="*60)
    print("Interactive mode - type your prompts (Ctrl+C to exit)")
    print("="*60 + "\n")
    
    device = next(model.parameters()).device
    
    while True:
        try:
            prompt = input("Prompt> ")
            if not prompt.strip():
                continue
            
            print("Generating...\n")
            output = generate(
                model,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
            )
            print(output)
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate text with a Baby Dragon Hatchling model")
    
    # Required
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (.pt file)")
    
    # Generation options
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (if not provided, enters interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling (default: disabled)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling threshold (default: disabled)")
    
    # System
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu, mps")
    
    # Output
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Load model
    model, config, info = load_model(args.checkpoint, args.device)
    
    if args.prompt is None:
        # Interactive mode
        interactive_mode(model, args)
    else:
        # Single generation
        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f"\n--- Sample {i+1}/{args.num_samples} ---")
            
            output = generate(
                model,
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            print(output)


if __name__ == "__main__":
    main()
