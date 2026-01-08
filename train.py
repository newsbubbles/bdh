#!/usr/bin/env python3
# Copyright Pathway Technology, Inc.
# Refactored training script with checkpointing, evaluation, and multi-dataset support

import argparse
import csv
import dataclasses
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import bdh
import numpy as np
import requests
import torch
import torch.nn.functional as F

# =============================================================================
# Configuration
# =============================================================================

@dataclasses.dataclass
class TrainConfig:
    """Training configuration - separate from model config."""
    # Data
    dataset: str = "shakespeare"  # shakespeare, wikitext2, or path to .txt file
    block_size: int = 512
    batch_size: int = 32
    
    # Training
    max_iters: int = 3000
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 50
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1000
    save_best: bool = True
    resume_from: str = None  # Path to checkpoint to resume from
    
    # Logging
    log_interval: int = 100
    log_file: str = None  # CSV log file path
    
    # System
    device: str = "auto"  # auto, cuda, cpu, mps
    dtype: str = "auto"   # auto, float32, bfloat16, float16
    compile_model: bool = True
    seed: int = 1337


def get_device(config: TrainConfig) -> torch.device:
    """Determine the best available device."""
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config.device)


def get_dtype(config: TrainConfig, device: torch.device) -> tuple:
    """Determine dtype and setup autocast context."""
    if config.dtype == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype_str = "bfloat16"
        elif device.type == "cuda":
            dtype_str = "float16"
        else:
            dtype_str = "float32"
    else:
        dtype_str = config.dtype
    
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_str]
    
    if device.type == "cuda":
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    else:
        ctx = nullcontext()
    
    return dtype_str, ptdtype, ctx


# =============================================================================
# Data Loading
# =============================================================================

class DataLoader:
    """Abstract data loader for byte-level text data."""
    
    def __init__(self, data_dir: Path, block_size: int, batch_size: int, device: torch.device):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.data_dir = data_dir
        
        # Load train/val splits
        self.train_data = None
        self.val_data = None
        self._load_data()
    
    def _load_data(self):
        raise NotImplementedError
    
    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        
        return x, y
    
    @property
    def train_tokens(self) -> int:
        return len(self.train_data)
    
    @property
    def val_tokens(self) -> int:
        return len(self.val_data)


class ShakespeareDataLoader(DataLoader):
    """Tiny Shakespeare dataset loader."""
    
    DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    def _load_data(self):
        input_file = self.data_dir / "shakespeare.txt"
        
        # Download if needed
        if not input_file.exists():
            print(f"Downloading Shakespeare dataset to {input_file}...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            response = requests.get(self.DATA_URL)
            input_file.write_text(response.text)
        
        # Load as bytes
        data = np.memmap(input_file, dtype=np.uint8, mode="r")
        n = len(data)
        
        # 90/10 train/val split
        self.train_data = data[:int(0.9 * n)]
        self.val_data = data[int(0.9 * n):]
        
        print(f"Shakespeare: {self.train_tokens:,} train tokens, {self.val_tokens:,} val tokens")


class WikiText2DataLoader(DataLoader):
    """WikiText-2 dataset loader."""
    
    DATA_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
    
    def _load_data(self):
        wikitext_dir = self.data_dir / "wikitext-2-raw"
        train_file = wikitext_dir / "wiki.train.raw"
        val_file = wikitext_dir / "wiki.valid.raw"
        
        # Download and extract if needed
        if not train_file.exists():
            print(f"Downloading WikiText-2 dataset...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            import zipfile
            import io
            
            response = requests.get(self.DATA_URL)
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(self.data_dir)
            
            print(f"Extracted to {wikitext_dir}")
        
        # Load as bytes
        self.train_data = np.frombuffer(train_file.read_bytes(), dtype=np.uint8)
        self.val_data = np.frombuffer(val_file.read_bytes(), dtype=np.uint8)
        
        print(f"WikiText-2: {self.train_tokens:,} train tokens, {self.val_tokens:,} val tokens")


class TextFileDataLoader(DataLoader):
    """Generic text file loader."""
    
    def __init__(self, file_path: str, block_size: int, batch_size: int, device: torch.device):
        self.file_path = Path(file_path)
        super().__init__(self.file_path.parent, block_size, batch_size, device)
    
    def _load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        data = np.frombuffer(self.file_path.read_bytes(), dtype=np.uint8)
        n = len(data)
        
        # 90/10 split
        self.train_data = data[:int(0.9 * n)]
        self.val_data = data[int(0.9 * n):]
        
        print(f"Custom data: {self.train_tokens:,} train tokens, {self.val_tokens:,} val tokens")


def get_data_loader(config: TrainConfig, device: torch.device) -> DataLoader:
    """Factory function to get the appropriate data loader."""
    data_dir = Path("data")
    
    if config.dataset == "shakespeare":
        return ShakespeareDataLoader(data_dir, config.block_size, config.batch_size, device)
    elif config.dataset == "wikitext2":
        return WikiText2DataLoader(data_dir, config.block_size, config.batch_size, device)
    elif os.path.exists(config.dataset):
        return TextFileDataLoader(config.dataset, config.block_size, config.batch_size, device)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model_config: bdh.BDHConfig,
    train_config: TrainConfig,
    step: int,
    val_loss: float,
    best_val_loss: float,
    path: Path,
):
    """Save a training checkpoint."""
    # Handle compiled model
    model_state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    
    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "model_config": dataclasses.asdict(model_config),
        "train_config": dataclasses.asdict(train_config),
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    device: torch.device,
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.amp.GradScaler = None,
) -> dict:
    """Load a training checkpoint."""
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if model is not None:
        # Handle compiled model
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        target.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # Restore RNG states for reproducibility
    if checkpoint.get("rng_state") is not None:
        torch.set_rng_state(checkpoint["rng_state"])
    if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    
    return checkpoint


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: DataLoader, eval_iters: int, ctx) -> dict:
    """Evaluate the model on train and val splits."""
    model.eval()
    results = {}
    
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, y = data_loader.get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
        
        avg_loss = sum(losses) / len(losses)
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        
        results[f"{split}_loss"] = avg_loss
        results[f"{split}_ppl"] = perplexity
    
    model.train()
    return results


# =============================================================================
# Logging
# =============================================================================

class TrainLogger:
    """Simple CSV logger for training metrics."""
    
    def __init__(self, log_file: Path = None):
        self.log_file = log_file
        self.rows = []
        self.fieldnames = None
    
    def log(self, metrics: dict):
        """Log a row of metrics."""
        if self.fieldnames is None:
            self.fieldnames = list(metrics.keys())
        
        self.rows.append(metrics)
        
        if self.log_file:
            self._write_csv()
    
    def _write_csv(self):
        with open(self.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


# =============================================================================
# Training Loop
# =============================================================================

def train(model_config: bdh.BDHConfig, train_config: TrainConfig):
    """Main training function."""
    
    # Setup
    torch.manual_seed(train_config.seed)
    device = get_device(train_config)
    dtype_str, ptdtype, ctx = get_dtype(train_config, device)
    
    print(f"Using device: {device} with dtype {dtype_str}")
    
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Data
    data_loader = get_data_loader(train_config, device)
    
    # Model
    model = bdh.BDH(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if train_config.compile_model and hasattr(torch, "compile"):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    
    # Scaler for mixed precision
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype_str == "float16"))
    
    # Resume from checkpoint if specified
    start_step = 0
    best_val_loss = float("inf")
    
    if train_config.resume_from:
        checkpoint = load_checkpoint(
            Path(train_config.resume_from),
            device,
            model,
            optimizer,
            scaler,
        )
        start_step = checkpoint["step"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed from step {start_step}, best_val_loss={best_val_loss:.4f}")
    
    # Logging
    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = Path(train_config.log_file) if train_config.log_file else checkpoint_dir / "train_log.csv"
    logger = TrainLogger(log_file)
    
    # Save config
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_config": dataclasses.asdict(model_config),
            "train_config": dataclasses.asdict(train_config),
        }, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Training loop
    model.train()
    x, y = data_loader.get_batch("train")
    
    loss_acc = 0.0
    loss_steps = 0
    t0 = time.time()
    
    for step in range(start_step, train_config.max_iters):
        # Forward pass
        with ctx:
            logits, loss = model(x, y)
        
        # Prefetch next batch
        x, y = data_loader.get_batch("train")
        
        # Backward pass
        loss_acc += loss.item()
        loss_steps += 1
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if step > 0 and step % train_config.log_interval == 0:
            dt = time.time() - t0
            avg_loss = loss_acc / loss_steps
            tokens_per_sec = (loss_steps * train_config.batch_size * train_config.block_size) / dt
            
            print(f"Step {step:6d}/{train_config.max_iters} | loss {avg_loss:.4f} | {tokens_per_sec:.0f} tok/s")
            
            loss_acc = 0.0
            loss_steps = 0
            t0 = time.time()
        
        # Evaluation
        if step > 0 and step % train_config.eval_interval == 0:
            eval_results = evaluate(model, data_loader, train_config.eval_iters, ctx)
            
            print(f"  Eval: train_loss={eval_results['train_loss']:.4f}, "
                  f"val_loss={eval_results['val_loss']:.4f}, "
                  f"val_ppl={eval_results['val_ppl']:.2f}")
            
            # Log metrics
            logger.log({
                "step": step,
                "train_loss": eval_results["train_loss"],
                "val_loss": eval_results["val_loss"],
                "train_ppl": eval_results["train_ppl"],
                "val_ppl": eval_results["val_ppl"],
            })
            
            # Save best model
            if train_config.save_best and eval_results["val_loss"] < best_val_loss:
                best_val_loss = eval_results["val_loss"]
                save_checkpoint(
                    model, optimizer, scaler,
                    model_config, train_config,
                    step, eval_results["val_loss"], best_val_loss,
                    checkpoint_dir / "best.pt",
                )
        
        # Regular checkpoint
        if step > 0 and step % train_config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scaler,
                model_config, train_config,
                step, loss.item(), best_val_loss,
                checkpoint_dir / f"step_{step:06d}.pt",
            )
    
    # Final checkpoint
    final_eval = evaluate(model, data_loader, train_config.eval_iters, ctx)
    save_checkpoint(
        model, optimizer, scaler,
        model_config, train_config,
        train_config.max_iters, final_eval["val_loss"], best_val_loss,
        checkpoint_dir / "final.pt",
    )
    
    print(f"\nTraining complete!")
    print(f"  Final val_loss: {final_eval['val_loss']:.4f}")
    print(f"  Final val_ppl: {final_eval['val_ppl']:.2f}")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    
    # Generate a sample
    print("\nGenerating sample...")
    model.eval()
    prompt = "To be or "
    prompt_tokens = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = model.generate(prompt_tokens, max_new_tokens=100, top_k=3)
    
    output_text = bytes(output[0].tolist()).decode("utf-8", errors="replace")
    print(f"Prompt: {prompt}")
    print(f"Output: {output_text}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Baby Dragon Hatchling model")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        help="Dataset: shakespeare, wikitext2, or path to .txt file")
    
    # Model config
    parser.add_argument("--n-layer", type=int, default=6, help="Number of layers")
    parser.add_argument("--n-embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--mlp-mult", type=int, default=128, help="MLP internal dimension multiplier")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training config
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--block-size", type=int, default=512, help="Context length")
    parser.add_argument("--max-iters", type=int, default=3000, help="Maximum training iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping (0 to disable)")
    
    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--eval-iters", type=int, default=50, help="Iterations per evaluation")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--no-save-best", action="store_true", help="Disable saving best model")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--log-file", type=str, default=None, help="CSV log file path")
    
    # System
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu, mps")
    parser.add_argument("--dtype", type=str, default="auto", help="Dtype: auto, float32, bfloat16, float16")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build configs from args
    model_config = bdh.BDHConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mlp_mult,
        dropout=args.dropout,
        vocab_size=256,  # Byte-level
    )
    
    train_config = TrainConfig(
        dataset=args.dataset,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        save_best=not args.no_save_best,
        resume_from=args.resume,
        log_interval=args.log_interval,
        log_file=args.log_file,
        device=args.device,
        dtype=args.dtype,
        compile_model=not args.no_compile,
        seed=args.seed,
    )
    
    train(model_config, train_config)


if __name__ == "__main__":
    main()
