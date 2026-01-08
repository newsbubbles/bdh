# BDH Benchmarking Results Summary

## Key Finding: Fair Comparison with Bits-Per-Byte

### The Problem with Perplexity

Comparing BDH (byte-level) to GPT-2 (BPE) using perplexity is **unfair**:
- BDH perplexity: uncertainty over 256 possible bytes
- GPT-2 perplexity: uncertainty over 50,257 possible tokens

### The Solution: Bits-Per-Byte (BPB)

BPB measures compression efficiency on the same text:
- Lower = better (model compresses text more efficiently)
- 1.0 BPB = 1 bit of information per byte of text
- Allows fair comparison across tokenization schemes

## Results on Shakespeare (Validation Set)

### Bits-Per-Byte Comparison

| Model | BPB | Params | BPB/M params |
|-------|-----|--------|-------------|
| GPT-2 (124M) | **0.369** | 124.4M | 0.0030 |
| BDH (best.pt) | 1.759 | 25.3M | 0.0695 |

### Analysis

1. **GPT-2 wins on raw BPB** (0.369 vs 1.759 = 4.77x better)
2. **But GPT-2 has 4.9x more parameters**
3. **Per-parameter efficiency**: GPT-2 is still more efficient

### Why GPT-2 Wins

1. **Pre-training**: GPT-2 was trained on 40GB of internet text
2. **BPE tokenization**: More efficient encoding than raw bytes
3. **Architecture maturity**: Transformer is highly optimized

### Why This Is Still Promising for BDH

1. **BDH was only trained on 1MB of Shakespeare** (tiny dataset!)
2. **BDH has novel architecture** (sparse activations, multiplicative gating)
3. **BDH overfits quickly** - needs more data/regularization
4. **25M params vs 124M** - BDH is 5x smaller

## Training Observations

### A100 Colab Run

| Metric | Value |
|--------|-------|
| Best Val Loss | 1.53 (step 500) |
| Best Val PPL | 4.64 |
| Final Val Loss | 3.06 (step 2500) |
| Overfitting | Severe after step 500 |

### Overfitting Analysis

```
Step  Train_Loss  Val_Loss  Gap
500   1.17        1.53      0.36  ✓ OK
1000  0.67        1.82      1.15  ⚠️ Mild
1500  0.24        2.45      2.21  ❌ Severe
2000  0.12        2.87      2.75  ❌ Severe
2500  0.09        3.06      2.97  ❌ Severe
```

**Recommendation**: Use `best.pt` (early stopping at step 500)

## Next Steps

### 1. Fix Overfitting
- [ ] Train on WikiText-2 (2M tokens vs 1M)
- [ ] Increase dropout (currently 0.1)
- [ ] Add weight decay
- [ ] Data augmentation?

### 2. Scale Comparison
- [ ] Train BDH at GPT-2 scale (124M params)
- [ ] Compare at equal compute budget
- [ ] Test on multiple datasets

### 3. Architecture Ablations
- [ ] ReLU vs GELU vs SwiGLU
- [ ] With/without multiplicative gating
- [ ] Different sparsity levels

## Files Generated

```
results/
├── training_viz/
│   ├── loss_curves.png
│   ├── perplexity_curves.png
│   ├── overfitting_analysis.png
│   ├── training_summary.png
│   └── training_report.md
├── bdh_vs_gpt2_shakespeare/
│   ├── comparison.png
│   ├── perplexity.json
│   ├── perplexity.csv
│   └── benchmark_report.md
└── bpb_shakespeare/
    ├── bpb_comparison.png
    ├── bpb_report.md
    └── results.json
```

## Benchmark Infrastructure Created

```
benchmarks/
├── __init__.py
├── perplexity.py      # Perplexity evaluation
├── datasets.py        # Dataset loaders
└── gpt2_native.py     # GPT-2 native eval

baselines/
├── __init__.py
├── gpt2.py           # GPT-2 wrapper
└── registry.py       # Model registry

scripts/
├── plot_training.py   # Training visualization
├── run_benchmark.py   # Main benchmark runner
└── compare_bpb.py     # Fair BPB comparison
```
