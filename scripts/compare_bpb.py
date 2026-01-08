#!/usr/bin/env python3
"""Compare models using Bits-Per-Byte (BPB) - the fair metric.

Bits-Per-Byte is the only fair comparison between:
- Byte-level models (BDH): predicts next byte
- BPE models (GPT-2): predicts next subword token

BPB = cross_entropy_loss / log(2)

Lower BPB = better compression = better model.

Usage:
    python scripts/compare_bpb.py --bdh checkpoints/best.pt --gpt2 gpt2 --text data/shakespeare.txt
"""

import argparse
import math
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    print("Please install: pip install matplotlib pandas")
    sys.exit(1)


def evaluate_bdh_bpb(
    checkpoint_path: str,
    text: str,
    block_size: int = 256,
    batch_size: int = 4,
    device: str = "auto",
    max_batches: int = None,
) -> dict:
    """Evaluate BDH and return bits-per-byte."""
    from bdh import BDH, BDHConfig
    from benchmarks.datasets import text_to_bytes
    from torch.utils.data import DataLoader, TensorDataset
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = BDHConfig(**ckpt["model_config"])
    model = BDH(config)
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model = model.to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    
    # Tokenize as bytes
    data = text_to_bytes(text)
    n_bytes = len(data)
    
    # Create batches
    n_blocks = len(data) // (block_size + 1)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, n_blocks, batch_size):
            if max_batches and i // batch_size >= max_batches:
                break
            
            batch_x = []
            batch_y = []
            for j in range(i, min(i + batch_size, n_blocks)):
                start = j * (block_size + 1)
                chunk = data[start : start + block_size + 1]
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])
            
            x = torch.stack(batch_x).to(device)
            y = torch.stack(batch_y).to(device)
            
            _, loss = model(x, y)
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens  # nats per byte
    bpb = avg_loss / math.log(2)  # bits per byte
    ppl = math.exp(avg_loss)
    
    return {
        "model": f"BDH ({Path(checkpoint_path).stem})",
        "type": "byte-level",
        "bpb": bpb,
        "perplexity": ppl,
        "loss": avg_loss,
        "tokens_evaluated": total_tokens,
        "params": n_params,
        "bytes_in_text": n_bytes,
    }


def evaluate_gpt2_bpb(
    model_name: str,
    text: str,
    block_size: int = 512,
    batch_size: int = 4,
    device: str = "auto",
    max_batches: int = None,
) -> dict:
    """Evaluate GPT-2 with native tokenization and return bits-per-byte."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    n_bytes = len(text.encode('utf-8'))
    
    # Tokenize
    tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)
    n_tokens = len(tokens)
    n_blocks = n_tokens // (block_size + 1)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, n_blocks, batch_size):
            if max_batches and i // batch_size >= max_batches:
                break
            
            batch_x = []
            batch_y = []
            for j in range(i, min(i + batch_size, n_blocks)):
                start = j * (block_size + 1)
                chunk = tokens[start : start + block_size + 1]
                batch_x.append(chunk[:-1])
                batch_y.append(chunk[1:])
            
            x = torch.stack(batch_x).to(device)
            y = torch.stack(batch_y).to(device)
            
            outputs = model(x, labels=y)
            loss = outputs.loss
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens  # nats per token
    
    # Convert to bits per byte:
    # total_bits = total_loss / log(2)
    # bpb = total_bits / n_bytes
    total_nats = total_loss
    total_bits = total_nats / math.log(2)
    bpb = total_bits / n_bytes
    
    token_ppl = math.exp(avg_loss)
    
    return {
        "model": f"GPT-2 ({model_name})",
        "type": "BPE",
        "bpb": bpb,
        "perplexity": token_ppl,  # Token perplexity (not comparable to BDH)
        "loss": avg_loss,
        "tokens_evaluated": total_tokens,
        "params": n_params,
        "bytes_in_text": n_bytes,
        "bpe_tokens": n_tokens,
    }


def plot_comparison(results: list[dict], output_dir: Path):
    """Generate comparison charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: BPB comparison (the fair metric)
    ax = axes[0]
    models = [r["model"] for r in results]
    bpbs = [r["bpb"] for r in results]
    colors = ['#2ecc71' if 'BDH' in m else '#3498db' for m in models]
    
    bars = ax.bar(range(len(models)), bpbs, color=colors, edgecolor='black')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Bits Per Byte (lower is better)')
    ax.set_title('Fair Comparison: Bits Per Byte')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, bpb in zip(bars, bpbs):
        ax.annotate(f'{bpb:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right: BPB vs Parameters
    ax = axes[1]
    for r in results:
        color = '#2ecc71' if 'BDH' in r['model'] else '#3498db'
        ax.scatter(r['params'] / 1e6, r['bpb'], s=200, c=color, 
                  edgecolors='black', zorder=5, label=r['model'])
    
    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('Bits Per Byte')
    ax.set_title('Efficiency: BPB vs Model Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'bpb_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def generate_report(results: list[dict], output_dir: Path, text_source: str):
    """Generate markdown report."""
    
    # Sort by BPB
    results_sorted = sorted(results, key=lambda x: x['bpb'])
    
    report = f"""# Bits-Per-Byte Comparison Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Text Source**: {text_source}  

## Why Bits-Per-Byte?

Comparing byte-level models (BDH) to BPE models (GPT-2) using perplexity is **unfair**:
- BDH perplexity: uncertainty over 256 possible bytes
- GPT-2 perplexity: uncertainty over 50,257 possible tokens

**Bits-Per-Byte (BPB)** is the fair metric:
- Measures compression efficiency on the same text
- Lower = better (model compresses text more efficiently)
- 1.0 BPB = 1 bit of information per byte of text

## Results

| Rank | Model | BPB | Params | Type |
|------|-------|-----|--------|------|
"""
    
    for i, r in enumerate(results_sorted, 1):
        report += f"| {i} | {r['model']} | **{r['bpb']:.3f}** | {r['params']/1e6:.1f}M | {r['type']} |\n"
    
    winner = results_sorted[0]
    report += f"""

## Analysis

**Winner**: {winner['model']} with {winner['bpb']:.3f} bits per byte

### Efficiency Comparison

"""
    
    # Compare BDH to GPT-2
    bdh_results = [r for r in results if 'BDH' in r['model']]
    gpt2_results = [r for r in results if 'GPT-2' in r['model']]
    
    if bdh_results and gpt2_results:
        bdh = bdh_results[0]
        gpt2 = gpt2_results[0]
        
        bpb_ratio = bdh['bpb'] / gpt2['bpb']
        param_ratio = gpt2['params'] / bdh['params']
        efficiency = (gpt2['bpb'] / bdh['bpb']) / param_ratio  # BPB improvement per param
        
        report += f"""- BDH BPB: {bdh['bpb']:.3f} vs GPT-2 BPB: {gpt2['bpb']:.3f}
- BPB ratio: {bpb_ratio:.2f}x {'(BDH better)' if bpb_ratio < 1 else '(GPT-2 better)'}
- Parameter ratio: GPT-2 has {param_ratio:.1f}x more parameters
- BPB per million params: BDH={bdh['bpb']/(bdh['params']/1e6):.4f}, GPT-2={gpt2['bpb']/(gpt2['params']/1e6):.4f}
"""
        
        if bpb_ratio < 1:
            improvement = (1 - bpb_ratio) * 100
            report += f"\n**BDH achieves {improvement:.1f}% better compression with {param_ratio:.1f}x fewer parameters!**\n"
        elif bpb_ratio > 1:
            gap = (bpb_ratio - 1) * 100
            report += f"\nGPT-2 achieves {gap:.1f}% better compression, but uses {param_ratio:.1f}x more parameters.\n"
    
    report += """
## Chart

![BPB Comparison](bpb_comparison.png)

## Raw Data

```json
""" + json.dumps(results, indent=2) + """
```
"""
    
    output_path = output_dir / 'bpb_report.md'
    output_path.write_text(report)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare models using Bits-Per-Byte (fair metric)"
    )
    parser.add_argument("--bdh", required=True, help="BDH checkpoint path")
    parser.add_argument("--gpt2", default="gpt2", help="GPT-2 model name")
    parser.add_argument("--text", required=True, help="Text file to evaluate on")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    
    args = parser.parse_args()
    
    # Setup output
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = Path(f"results/bpb_comparison_{timestamp}")
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load text
    text = Path(args.text).read_text(encoding='utf-8')
    print(f"Loaded {len(text):,} characters from {args.text}")
    
    results = []
    
    # Evaluate BDH
    print(f"\n--- Evaluating BDH ---")
    bdh_result = evaluate_bdh_bpb(
        args.bdh, text,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches,
    )
    print(f"  BPB: {bdh_result['bpb']:.4f}")
    results.append(bdh_result)
    
    # Free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Evaluate GPT-2
    print(f"\n--- Evaluating GPT-2 ({args.gpt2}) ---")
    gpt2_result = evaluate_gpt2_bpb(
        args.gpt2, text,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches,
    )
    print(f"  BPB: {gpt2_result['bpb']:.4f}")
    results.append(gpt2_result)
    
    # Generate outputs
    print(f"\n--- Generating Report ---")
    plot_comparison(results, args.output)
    generate_report(results, args.output, args.text)
    
    # Save raw results
    with open(args.output / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
