# BDH Benchmarking Framework

## Current Status (Session Update)

### Training Results from A100 Colab Run
- **Model**: BDH 6-layer, 256d, 4 heads, 25.3M params
- **Dataset**: Shakespeare (byte-level)
- **Best checkpoint**: `best.pt` at step 500, val_loss=1.53
- **Final checkpoint**: `final.pt` at step 3000, val_loss=3.25 (overfit)

### Key Observations
1. Model learns Shakespeare dialogue format quickly
2. Severe overfitting after ~500 steps on small dataset
3. Generation quality decent but has nonsense words
4. Need more data (WikiText-2) and regularization

## Three Immediate Goals

### 1. Fix Overfitting
- [ ] Train on WikiText-2 (larger, more diverse)
- [ ] Increase dropout
- [ ] Data augmentation?
- [ ] Early stopping

### 2. Perplexity Benchmark vs GPT-2
- [ ] Implement `benchmarks/perplexity.py`
- [ ] Load GPT-2 from HuggingFace
- [ ] Compare on same test data
- [ ] Fair comparison: same tokenization? Or native?

### 3. Training Visualization
- [ ] Plot loss curves from CSV
- [ ] Create `scripts/plot_training.py`
- [ ] Save charts to `results/`

## Benchmark Metrics

| Metric | Description | Priority |
|--------|-------------|----------|
| Perplexity | exp(cross-entropy loss) | HIGH |
| Params | Total parameters | HIGH |
| Tokens/sec | Throughput | MEDIUM |
| Memory | Peak GPU memory | MEDIUM |
| MAUVE | Generation quality | LOW (later) |

## Baselines to Compare

| Model | Params | Notes |
|-------|--------|-------|
| GPT-2 (124M) | 124M | Standard baseline |
| GPT-2 (355M) | 355M | Larger baseline |
| BDH (current) | 25M | Our model |
| LLaMA-tiny | ~25M | Fair size comparison |

## File Structure

```
benchmarks/
├── __init__.py
├── perplexity.py      # Perplexity evaluation
├── datasets.py        # Dataset loaders
└── efficiency.py      # FLOPs, memory, speed

baselines/
├── __init__.py
├── gpt2.py           # GPT-2 wrapper
└── registry.py       # Model registry

results/
└── {experiment}/
    ├── metrics.json
    ├── metrics.csv
    └── charts/

scripts/
├── plot_training.py  # Visualize training logs
└── run_benchmark.py  # Main benchmark runner
```
