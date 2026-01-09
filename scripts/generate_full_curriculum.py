#!/usr/bin/env python3
"""Generate the complete BDH curriculum dataset.

This is the ONE script to run to generate all training data.
Designed to work in Colab or locally.

Usage:
    python scripts/generate_full_curriculum.py
    python scripts/generate_full_curriculum.py --quick  # Smaller dataset for testing
    python scripts/generate_full_curriculum.py --stats  # Just show statistics

Output structure:
    data/curriculum/
    ├── phase1_primitives/
    │   └── phase1_primitives.jsonl
    ├── phase2_composition/
    │   └── phase2_composition.jsonl
    ├── phase3_algorithms/
    │   └── phase3_algorithms.jsonl
    ├── phase4_systems/
    │   └── phase4_systems.jsonl
    ├── phase5_language/
    │   ├── phase5_train.jsonl
    │   └── phase5_val.jsonl
    └── curriculum_manifest.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add scripts to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "data_pipeline"))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Full curriculum targets (for real training)
FULL_CONFIG = {
    "phase1": {"variants_per_template": 100, "target_examples": 10_000},
    "phase2": {"variants_per_template": 150, "target_examples": 20_000},
    "phase3": {"variants_per_template": 100, "target_examples": 30_000},
    "phase4": {"target_examples": 15_000},
    "phase5": {"chunk_size": 256, "max_chunks": None},  # Full WikiText-2
}

# Quick config for testing (smaller but representative)
QUICK_CONFIG = {
    "phase1": {"variants_per_template": 20, "target_examples": 2_000},
    "phase2": {"variants_per_template": 30, "target_examples": 3_000},
    "phase3": {"variants_per_template": 20, "target_examples": 5_000},
    "phase4": {"target_examples": 2_000},
    "phase5": {"chunk_size": 256, "max_chunks": 20_000},
}

# Training step ratios (how much to train on each phase relative to corpus size)
# This ensures balanced learning across phases
TRAINING_RATIOS = {
    "phase1": 1.5,   # Oversample primitives (foundation)
    "phase2": 1.2,   # Slightly oversample composition
    "phase3": 1.0,   # Standard ratio for algorithms
    "phase4": 1.5,   # Oversample systems (complex, fewer examples)
    "phase5": 0.8,   # Undersample language (largest corpus)
}


# =============================================================================
# PHASE GENERATORS
# =============================================================================

def generate_phase1(output_dir: Path, config: dict, force: bool = False) -> dict:
    """Generate Phase 1: Primitives - Simple functions with heavy comments."""
    from phase1_generator import save_phase1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "phase1_primitives.jsonl"
    
    if jsonl_path.exists() and not force:
        print(f"  Phase 1 exists ({count_lines(jsonl_path):,} examples), skipping...")
        return load_stats(output_dir / "phase1_stats.json")
    
    print(f"  Generating Phase 1 (target: {config.get('target_examples', 10000):,} examples)...")
    return save_phase1(
        output_dir,
        variants_per_template=config.get("variants_per_template", 50)
    )


def generate_phase2(output_dir: Path, config: dict, force: bool = False) -> dict:
    """Generate Phase 2: Composition - Loops, recursion, control flow."""
    from phase2_generator import save_phase2
    
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "phase2_composition.jsonl"
    
    if jsonl_path.exists() and not force:
        print(f"  Phase 2 exists ({count_lines(jsonl_path):,} examples), skipping...")
        return load_stats(output_dir / "phase2_stats.json")
    
    print(f"  Generating Phase 2 (target: {config.get('target_examples', 20000):,} examples)...")
    return save_phase2(
        output_dir,
        variants_per_template=config.get("variants_per_template", 100)
    )


def generate_phase3(output_dir: Path, config: dict, force: bool = False) -> dict:
    """Generate Phase 3: Algorithms - Classic CS algorithms."""
    from phase3_algorithms import save_phase3
    
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "phase3_algorithms.jsonl"
    
    if jsonl_path.exists() and not force:
        print(f"  Phase 3 exists ({count_lines(jsonl_path):,} examples), skipping...")
        return load_stats(output_dir / "phase3_stats.json")
    
    print(f"  Generating Phase 3 (target: {config.get('target_examples', 30000):,} examples)...")
    return save_phase3(
        output_dir,
        variants_per_template=config.get("variants_per_template", 50)
    )


def generate_phase4(output_dir: Path, config: dict, force: bool = False) -> dict:
    """Generate Phase 4: Systems - OOP, design patterns, real-world code."""
    from phase4_systems import save_phase4
    
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "phase4_systems.jsonl"
    
    if jsonl_path.exists() and not force:
        print(f"  Phase 4 exists ({count_lines(jsonl_path):,} examples), skipping...")
        return load_stats(output_dir / "phase4_stats.json")
    
    print(f"  Generating Phase 4 (target: {config.get('target_examples', 15000):,} examples)...")
    return save_phase4(output_dir)


def generate_phase5(output_dir: Path, config: dict, force: bool = False) -> dict:
    """Generate Phase 5: Language - WikiText-2 natural language."""
    from phase5_language import save_phase5_hf
    
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "phase5_train.jsonl"
    
    if jsonl_path.exists() and not force:
        print(f"  Phase 5 exists ({count_lines(jsonl_path):,} examples), skipping...")
        return load_stats(output_dir / "phase5_stats.json")
    
    max_chunks = config.get("max_chunks")
    if max_chunks:
        print(f"  Generating Phase 5 (max {max_chunks:,} chunks)...")
    else:
        print(f"  Generating Phase 5 (full WikiText-2)...")
    
    return save_phase5_hf(
        output_dir,
        chunk_size=config.get("chunk_size", 256),
        max_chunks=max_chunks
    )


# =============================================================================
# UTILITIES
# =============================================================================

def count_lines(path: Path) -> int:
    """Count lines in a file."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def load_stats(path: Path) -> dict:
    """Load stats JSON file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_manifest(data_dir: Path, stats: dict) -> None:
    """Save curriculum manifest with all phase statistics."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "phases": stats,
        "training_ratios": TRAINING_RATIOS,
        "total_examples": sum(s.get("total", s.get("examples", 0)) for s in stats.values()),
        "total_tokens": sum(s.get("tokens", 0) for s in stats.values()),
    }
    
    # Calculate recommended epochs per phase based on ratios
    base_epochs = 5
    manifest["recommended_epochs"] = {
        phase: int(base_epochs * ratio) 
        for phase, ratio in TRAINING_RATIOS.items()
    }
    
    with open(data_dir / "curriculum_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def print_stats(data_dir: Path) -> None:
    """Print curriculum statistics."""
    manifest_path = data_dir / "curriculum_manifest.json"
    
    if not manifest_path.exists():
        print("No manifest found. Run generation first.")
        return
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print("\n" + "="*70)
    print("BDH CURRICULUM STATISTICS")
    print("="*70)
    print(f"Generated: {manifest.get('generated_at', 'unknown')}")
    print()
    
    total_examples = 0
    total_tokens = 0
    
    for phase_name, stats in manifest.get("phases", {}).items():
        examples = stats.get("total", stats.get("examples", 0))
        tokens = stats.get("tokens", 0)
        total_examples += examples
        total_tokens += tokens
        
        ratio = TRAINING_RATIOS.get(phase_name, 1.0)
        rec_epochs = manifest.get("recommended_epochs", {}).get(phase_name, 5)
        
        print(f"{phase_name.upper()}")
        print(f"  Examples: {examples:>10,}")
        print(f"  Tokens:   {tokens:>10,}")
        print(f"  Ratio:    {ratio:>10.1f}x")
        print(f"  Epochs:   {rec_epochs:>10}")
        print()
    
    print("-"*70)
    print(f"TOTAL: {total_examples:,} examples, {total_tokens:,} tokens")
    print(f"       (~{total_tokens/1_000_000:.1f}M tokens)")
    print("="*70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate the complete BDH curriculum dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_full_curriculum.py           # Full dataset
  python scripts/generate_full_curriculum.py --quick   # Smaller test dataset
  python scripts/generate_full_curriculum.py --stats   # Show statistics
  python scripts/generate_full_curriculum.py --force   # Regenerate all
"""
    )
    parser.add_argument("--quick", action="store_true",
                        help="Generate smaller dataset for testing")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics only")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if data exists")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                        help="Generate specific phase only")
    parser.add_argument("--data-dir", type=Path, default=Path("data/curriculum"),
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Just show stats
    if args.stats:
        print_stats(args.data_dir)
        return
    
    # Select config
    config = QUICK_CONFIG if args.quick else FULL_CONFIG
    config_name = "QUICK" if args.quick else "FULL"
    
    print("\n" + "#"*70)
    print(f"BDH CURRICULUM GENERATION ({config_name})")
    print("#"*70 + "\n")
    
    # Ensure base directory exists
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase generators
    generators = [
        ("phase1", "phase1_primitives", generate_phase1),
        ("phase2", "phase2_composition", generate_phase2),
        ("phase3", "phase3_algorithms", generate_phase3),
        ("phase4", "phase4_systems", generate_phase4),
        ("phase5", "phase5_language", generate_phase5),
    ]
    
    all_stats = {}
    
    for phase_key, dir_name, gen_func in generators:
        phase_num = int(phase_key[-1])
        
        # Skip if not the requested phase
        if args.phase and args.phase != phase_num:
            continue
        
        print(f"\n{'='*60}")
        print(f"PHASE {phase_num}: {dir_name}")
        print("="*60)
        
        output_dir = args.data_dir / dir_name
        phase_config = config.get(phase_key, {})
        
        try:
            stats = gen_func(output_dir, phase_config, force=args.force)
            all_stats[phase_key] = stats
            
            examples = stats.get("total", stats.get("examples", 0))
            tokens = stats.get("tokens", 0)
            print(f"  ✓ Generated {examples:,} examples ({tokens:,} tokens)")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_stats[phase_key] = {"error": str(e)}
    
    # Save manifest
    if not args.phase:  # Only save full manifest if generating all phases
        save_manifest(args.data_dir, all_stats)
    
    # Print final stats
    print_stats(args.data_dir)
    
    print("\n✓ Curriculum generation complete!")
    print(f"  Data directory: {args.data_dir.absolute()}")
    print(f"  Manifest: {args.data_dir / 'curriculum_manifest.json'}")


if __name__ == "__main__":
    main()
