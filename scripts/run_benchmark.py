#!/usr/bin/env python3
"""Run benchmarks comparing BDH against baselines.

This script orchestrates perplexity evaluation across models and datasets,
generating comparison charts and reports.

Usage:
    # Quick comparison on Shakespeare
    python scripts/run_benchmark.py --models bdh:checkpoints/best.pt gpt2 --dataset shakespeare
    
    # Full benchmark on WikiText-2
    python scripts/run_benchmark.py --models bdh:checkpoints/best.pt gpt2 gpt2_medium --dataset wikitext2
    
    # Custom output
    python scripts/run_benchmark.py --models bdh:checkpoints/best.pt --output results/my_benchmark/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install: pip install pandas matplotlib")
    sys.exit(1)

from benchmarks.perplexity import evaluate_perplexity, PerplexityResult, save_results
from benchmarks.datasets import load_benchmark_dataset
from baselines.registry import get_model, list_models


def parse_model_spec(spec: str) -> tuple[str, Optional[str]]:
    """Parse model specification.
    
    Formats:
        'gpt2' -> ('gpt2', None)
        'bdh:checkpoints/best.pt' -> ('bdh', 'checkpoints/best.pt')
        'checkpoints/best.pt' -> ('bdh', 'checkpoints/best.pt')
    """
    if ':' in spec:
        name, checkpoint = spec.split(':', 1)
        return name, checkpoint
    elif spec.endswith('.pt'):
        return 'bdh', spec
    else:
        return spec, None


def load_model_for_benchmark(spec: str, device: str = "auto"):
    """Load a model from specification string."""
    name, checkpoint = parse_model_spec(spec)
    
    if checkpoint:
        print(f"Loading {name} from {checkpoint}...")
        model = get_model(checkpoint)
    else:
        print(f"Loading {name}...")
        model = get_model(name)
    
    # Get display name
    if checkpoint:
        display_name = f"{name}_{Path(checkpoint).stem}"
    else:
        display_name = name
    
    return model, display_name


def plot_comparison(
    results: list[PerplexityResult],
    output_dir: Path,
    metric: str = "perplexity",
):
    """Generate comparison charts."""
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # Bar chart: Perplexity by model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Perplexity comparison
    ax = axes[0]
    models = df['model_name'].unique()
    x = range(len(models))
    colors = plt.cm.Set2(range(len(models)))
    
    bars = ax.bar(x, [df[df['model_name']==m]['perplexity'].values[0] for m in models],
                  color=colors, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Perplexity (lower is better)')
    ax.set_title(f'Perplexity Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, model in zip(bars, models):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Right: Perplexity vs Parameters (if we have param counts)
    ax = axes[1]
    if df['model_params'].notna().all():
        for i, (_, row) in enumerate(df.iterrows()):
            ax.scatter(row['model_params'] / 1e6, row['perplexity'],
                      s=150, c=[colors[i]], edgecolors='black', zorder=5)
            ax.annotate(row['model_name'],
                       (row['model_params'] / 1e6, row['perplexity']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Parameters (millions)')
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity vs Model Size')
        ax.grid(True, alpha=0.3)
        # ax.set_xscale('log')
    else:
        ax.text(0.5, 0.5, 'Parameter counts not available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Perplexity vs Model Size')
    
    plt.tight_layout()
    
    output_path = output_dir / 'comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def generate_report(
    results: list[PerplexityResult],
    output_dir: Path,
    args: argparse.Namespace,
):
    """Generate markdown comparison report."""
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # Sort by perplexity
    df_sorted = df.sort_values('perplexity')
    
    report = f"""# Benchmark Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset**: {args.dataset}  
**Split**: {args.split}  

## Results

| Rank | Model | Perplexity | Loss | Params | Tokens/sec |
|------|-------|------------|------|--------|------------|
"""
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        params = f"{row['model_params']/1e6:.1f}M" if row['model_params'] else "N/A"
        tps = f"{row['tokens_evaluated']/row['time_seconds']:.0f}" if row['time_seconds'] > 0 else "N/A"
        report += f"| {i} | {row['model_name']} | {row['perplexity']:.2f} | {row['loss']:.4f} | {params} | {tps} |\n"
    
    # Winner analysis
    winner = df_sorted.iloc[0]
    report += f"""

## Analysis

**Winner**: {winner['model_name']} with perplexity {winner['perplexity']:.2f}

### Key Observations

"""
    
    # Compare BDH to GPT-2 if both present
    bdh_results = df[df['model_name'].str.contains('bdh', case=False)]
    gpt2_results = df[df['model_name'].str.contains('gpt2', case=False)]
    
    if len(bdh_results) > 0 and len(gpt2_results) > 0:
        bdh_ppl = bdh_results['perplexity'].values[0]
        bdh_params = bdh_results['model_params'].values[0]
        gpt2_ppl = gpt2_results['perplexity'].min()
        gpt2_params = gpt2_results.loc[gpt2_results['perplexity'].idxmin(), 'model_params']
        
        ppl_ratio = bdh_ppl / gpt2_ppl
        param_ratio = gpt2_params / bdh_params if bdh_params else None
        
        report += f"""- BDH perplexity: {bdh_ppl:.2f} vs GPT-2: {gpt2_ppl:.2f} (ratio: {ppl_ratio:.2f}x)
"""
        if param_ratio:
            report += f"""- GPT-2 has {param_ratio:.1f}x more parameters than BDH
- Perplexity per million params: BDH={bdh_ppl/(bdh_params/1e6):.3f}, GPT-2={gpt2_ppl/(gpt2_params/1e6):.3f}
"""
    
    report += """
## Comparison Chart

![Comparison](comparison.png)

## Configuration

```json
""" + json.dumps({
        'models': args.models,
        'dataset': args.dataset,
        'split': args.split,
        'block_size': args.block_size,
        'batch_size': args.batch_size,
    }, indent=2) + """
```
"""
    
    output_path = output_dir / 'benchmark_report.md'
    output_path.write_text(report)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run BDH benchmarks against baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", "-m", nargs="+", required=True,
        help="Models to benchmark (e.g., 'gpt2', 'bdh:checkpoints/best.pt')")
    parser.add_argument(
        "--dataset", "-d", default="shakespeare",
        help="Dataset: shakespeare, wikitext2, ptb, or path to .txt")
    parser.add_argument(
        "--split", default="test",
        help="Dataset split: train, val, test")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory")
    parser.add_argument(
        "--block-size", type=int, default=512,
        help="Context length for evaluation")
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size")
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Limit to N batches (for quick testing)")
    parser.add_argument(
        "--device", default="auto",
        help="Device: auto, cuda, cpu")
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name in list_models():
            print(f"  - {name}")
        print("\nOr specify a checkpoint: bdh:checkpoints/best.pt")
        return
    
    # Setup output directory
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = Path(f"results/benchmark_{timestamp}")
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"BDH Benchmark Suite")
    print(f"{'='*60}")
    print(f"Models: {args.models}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Load dataset once
    print(f"Loading dataset: {args.dataset}")
    dataset = load_benchmark_dataset(args.dataset)
    print(f"  {dataset}\n")
    
    # Run benchmarks
    results = []
    
    for model_spec in args.models:
        print(f"\n--- Benchmarking: {model_spec} ---")
        try:
            model, display_name = load_model_for_benchmark(model_spec, args.device)
            
            result = evaluate_perplexity(
                model=model,
                dataset=dataset,
                split=args.split,
                block_size=args.block_size,
                batch_size=args.batch_size,
                device=args.device,
                model_name=display_name,
                max_batches=args.max_batches,
            )
            
            print(f"  ✓ {result}")
            results.append(result)
            
            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("\nNo successful benchmarks. Exiting.")
        return
    
    # Save results
    print(f"\n--- Saving Results ---")
    save_results(results, args.output)
    
    # Generate visualizations
    print(f"\n--- Generating Charts ---")
    plot_comparison(results, args.output)
    
    # Generate report
    print(f"\n--- Generating Report ---")
    generate_report(results, args.output, args)
    
    print(f"\n{'='*60}")
    print(f"✅ Benchmark complete! Results in {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
