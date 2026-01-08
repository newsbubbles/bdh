#!/usr/bin/env python3
"""Visualize training logs from BDH training runs.

Generates publication-quality charts showing:
- Loss curves (train vs val)
- Perplexity curves
- Overfitting analysis

Usage:
    python scripts/plot_training.py checkpoints/train_log.csv
    python scripts/plot_training.py checkpoints/train_log.csv --output results/charts/
"""

import argparse
from pathlib import Path
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("Please install: pip install pandas matplotlib")
    sys.exit(1)


def setup_style():
    """Set up publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


def load_training_log(path: Path) -> pd.DataFrame:
    """Load training log CSV."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    print(f"Columns: {list(df.columns)}")
    return df


def plot_loss_curves(df: pd.DataFrame, output_dir: Path, title_suffix: str = ""):
    """Plot train and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot losses
    ax.plot(df['step'], df['train_loss'], 'b-', linewidth=2, 
            marker='o', markersize=6, label='Train Loss')
    ax.plot(df['step'], df['val_loss'], 'r-', linewidth=2,
            marker='s', markersize=6, label='Val Loss')
    
    # Find best val loss point
    best_idx = df['val_loss'].idxmin()
    best_step = df.loc[best_idx, 'step']
    best_val = df.loc[best_idx, 'val_loss']
    
    ax.axvline(x=best_step, color='green', linestyle='--', alpha=0.7, 
               label=f'Best Val @ step {best_step}')
    ax.scatter([best_step], [best_val], color='green', s=100, zorder=5, 
               marker='*', edgecolors='black')
    
    # Styling
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (Cross-Entropy)')
    ax.set_title(f'BDH Training: Loss Curves{title_suffix}')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = output_dir / 'loss_curves.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def plot_perplexity_curves(df: pd.DataFrame, output_dir: Path, title_suffix: str = ""):
    """Plot train and validation perplexity curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot perplexity
    ax.plot(df['step'], df['train_ppl'], 'b-', linewidth=2,
            marker='o', markersize=6, label='Train Perplexity')
    ax.plot(df['step'], df['val_ppl'], 'r-', linewidth=2,
            marker='s', markersize=6, label='Val Perplexity')
    
    # Find best val perplexity point
    best_idx = df['val_ppl'].idxmin()
    best_step = df.loc[best_idx, 'step']
    best_val = df.loc[best_idx, 'val_ppl']
    
    ax.axvline(x=best_step, color='green', linestyle='--', alpha=0.7,
               label=f'Best Val @ step {best_step}')
    ax.scatter([best_step], [best_val], color='green', s=100, zorder=5,
               marker='*', edgecolors='black')
    
    # Styling
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Perplexity')
    ax.set_title(f'BDH Training: Perplexity{title_suffix}')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)
    
    # Log scale for perplexity often more readable
    # ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = output_dir / 'perplexity_curves.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def plot_overfitting_analysis(df: pd.DataFrame, output_dir: Path, title_suffix: str = ""):
    """Plot overfitting gap (val_loss - train_loss) over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Gap over time
    gap = df['val_loss'] - df['train_loss']
    colors = ['green' if g < 0.5 else 'orange' if g < 1.0 else 'red' for g in gap]
    
    ax1.bar(df['step'], gap, color=colors, alpha=0.7, width=df['step'].diff().mean() * 0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Mild overfit')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Severe overfit')
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Generalization Gap (Val - Train Loss)')
    ax1.set_title(f'Overfitting Analysis{title_suffix}')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Right: Train vs Val scatter
    ax2.scatter(df['train_loss'], df['val_loss'], c=df['step'], 
                cmap='viridis', s=100, edgecolors='black')
    
    # Perfect generalization line
    max_val = max(df['train_loss'].max(), df['val_loss'].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect generalization')
    
    # Colorbar for steps
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Training Step')
    
    ax2.set_xlabel('Train Loss')
    ax2.set_ylabel('Val Loss')
    ax2.set_title('Train vs Val Loss (colored by step)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'overfitting_analysis.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_summary(df: pd.DataFrame, output_dir: Path, title_suffix: str = ""):
    """Create a combined summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Loss curves
    ax = axes[0, 0]
    ax.plot(df['step'], df['train_loss'], 'b-o', linewidth=2, markersize=5, label='Train')
    ax.plot(df['step'], df['val_loss'], 'r-s', linewidth=2, markersize=5, label='Val')
    best_idx = df['val_loss'].idxmin()
    ax.axvline(x=df.loc[best_idx, 'step'], color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top right: Perplexity curves
    ax = axes[0, 1]
    ax.plot(df['step'], df['train_ppl'], 'b-o', linewidth=2, markersize=5, label='Train')
    ax.plot(df['step'], df['val_ppl'], 'r-s', linewidth=2, markersize=5, label='Val')
    ax.axvline(x=df.loc[best_idx, 'step'], color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom left: Overfitting gap
    ax = axes[1, 0]
    gap = df['val_loss'] - df['train_loss']
    colors = ['green' if g < 0.5 else 'orange' if g < 1.0 else 'red' for g in gap]
    ax.bar(df['step'], gap, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Val - Train Loss')
    ax.set_title('Generalization Gap')
    ax.grid(True, alpha=0.3)
    
    # Bottom right: Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    
    # Compute stats
    best_val_loss = df['val_loss'].min()
    best_val_ppl = df['val_ppl'].min()
    best_step = df.loc[df['val_loss'].idxmin(), 'step']
    final_train_loss = df['train_loss'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    final_gap = final_val_loss - final_train_loss
    total_steps = df['step'].max()
    
    stats_text = f"""
    Training Summary
    {'='*40}
    
    Best Validation:
      • Step: {best_step:,}
      • Val Loss: {best_val_loss:.4f}
      • Val Perplexity: {best_val_ppl:.2f}
    
    Final State (step {total_steps:,}):
      • Train Loss: {final_train_loss:.4f}
      • Val Loss: {final_val_loss:.4f}
      • Gap: {final_gap:.4f} {'⚠️ OVERFIT' if final_gap > 1.0 else '✓ OK' if final_gap < 0.5 else '⚡ Mild'}
    
    Recommendation:
      {'Use best.pt checkpoint (early stopping)' if final_gap > 0.5 else 'Model generalizes well'}
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'BDH Training Summary{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'training_summary.png'
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def generate_markdown_report(df: pd.DataFrame, output_dir: Path, log_path: Path):
    """Generate a markdown report summarizing the training run."""
    
    best_idx = df['val_loss'].idxmin()
    best_step = df.loc[best_idx, 'step']
    best_val_loss = df.loc[best_idx, 'val_loss']
    best_val_ppl = df.loc[best_idx, 'val_ppl']
    
    final_train_loss = df['train_loss'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    
    report = f"""# Training Report

**Source**: `{log_path}`  
**Generated**: Auto-generated by `plot_training.py`

## Summary

| Metric | Value |
|--------|-------|
| Total Steps | {df['step'].max():,} |
| Best Val Loss | {best_val_loss:.4f} (step {best_step}) |
| Best Val Perplexity | {best_val_ppl:.2f} |
| Final Train Loss | {final_train_loss:.4f} |
| Final Val Loss | {final_val_loss:.4f} |
| Generalization Gap | {final_val_loss - final_train_loss:.4f} |

## Observations

{'⚠️ **Overfitting detected!** Val loss increased significantly after step ' + str(best_step) + '. Use `best.pt` checkpoint.' if (final_val_loss - best_val_loss) > 0.5 else '✅ Training looks healthy.'}

## Charts

### Loss Curves
![Loss Curves](loss_curves.png)

### Perplexity
![Perplexity](perplexity_curves.png)

### Overfitting Analysis
![Overfitting](overfitting_analysis.png)

### Full Summary
![Summary](training_summary.png)

## Raw Data

```
{df.to_string()}
```
"""
    
    output_path = output_dir / 'training_report.md'
    output_path.write_text(report)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BDH training logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_training.py checkpoints/train_log.csv
  python scripts/plot_training.py checkpoints/train_log.csv --output results/my_run/
  python scripts/plot_training.py checkpoints/train_log.csv --title "Shakespeare Run"
        """
    )
    parser.add_argument("log_file", type=Path, help="Path to train_log.csv")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: same as log file)")
    parser.add_argument("--title", "-t", type=str, default="",
                        help="Title suffix for charts")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.log_file.exists():
        print(f"Error: {args.log_file} not found")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output or args.log_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Title suffix
    title_suffix = f" ({args.title})" if args.title else ""
    
    # Setup
    setup_style()
    
    # Load data
    df = load_training_log(args.log_file)
    
    # Generate all plots
    print(f"\nGenerating charts in {output_dir}/")
    plot_loss_curves(df, output_dir, title_suffix)
    plot_perplexity_curves(df, output_dir, title_suffix)
    plot_overfitting_analysis(df, output_dir, title_suffix)
    plot_combined_summary(df, output_dir, title_suffix)
    generate_markdown_report(df, output_dir, args.log_file)
    
    print(f"\n✅ Done! Check {output_dir}/ for outputs.")


if __name__ == "__main__":
    main()
