# Copyright 2025 Pathway Technology, Inc.
#
# Hierarchical BDH - MEGABYTE-inspired multi-scale architecture
# Global model processes patch-level representations
# Local model refines byte-level predictions within patches

import dataclasses
import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclasses.dataclass
class HierarchicalBDHConfig:
    """Configuration for Hierarchical BDH.
    
    Size presets:
        tiny:  ~15M params - for testing
        small: ~50M params - Colab T4 friendly  
        base:  ~120M params - A100/V100
        large: ~350M params - Multi-GPU
    """
    # Patch configuration
    patch_size: int = 8  # P: bytes per patch
    max_seq_len: int = 1024  # T: max sequence length (must be divisible by patch_size)
    
    # Global model (processes patches)
    global_n_layer: int = 6
    global_n_embd: int = 512
    global_n_head: int = 8
    global_mlp_mult: int = 128  # MLP internal dim multiplier
    
    # Local model (processes bytes within patches)
    local_n_layer: int = 4
    local_n_embd: int = 256
    local_n_head: int = 4
    local_mlp_mult: int = 128
    
    # Shared
    vocab_size: int = 256
    dropout: float = 0.1
    
    # Whether to use BDH-style attention or standard attention
    use_bdh_attention: bool = True
    
    def __post_init__(self):
        assert self.max_seq_len % self.patch_size == 0, \
            f"max_seq_len ({self.max_seq_len}) must be divisible by patch_size ({self.patch_size})"
    
    @classmethod
    def tiny(cls, **kwargs) -> 'HierarchicalBDHConfig':
        """~10M params - for testing."""
        return cls(
            patch_size=4,
            global_n_layer=3, global_n_embd=128, global_n_head=4, global_mlp_mult=16,
            local_n_layer=2, local_n_embd=64, local_n_head=2, local_mlp_mult=16,
            **kwargs
        )
    
    @classmethod
    def small(cls, **kwargs) -> 'HierarchicalBDHConfig':
        """~25M params - match original BDH, Colab T4 friendly."""
        return cls(
            patch_size=8,
            global_n_layer=4, global_n_embd=224, global_n_head=4, global_mlp_mult=40,
            local_n_layer=2, local_n_embd=160, local_n_head=4, local_mlp_mult=40,
            **kwargs
        )
    
    @classmethod
    def base(cls, **kwargs) -> 'HierarchicalBDHConfig':
        """~50M params - A100/V100."""
        return cls(
            patch_size=8,
            global_n_layer=6, global_n_embd=256, global_n_head=8, global_mlp_mult=48,
            local_n_layer=3, local_n_embd=192, local_n_head=4, local_mlp_mult=48,
            **kwargs
        )
    
    @classmethod
    def large(cls, **kwargs) -> 'HierarchicalBDHConfig':
        """~100M params - Multi-GPU."""
        return cls(
            patch_size=8,
            global_n_layer=8, global_n_embd=384, global_n_head=8, global_mlp_mult=64,
            local_n_layer=4, local_n_embd=256, local_n_head=8, local_mlp_mult=64,
            **kwargs
        )


# =============================================================================
# ROTARY POSITION ENCODING
# =============================================================================

def get_freqs(n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    """Get rotary embedding frequencies."""
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (
        1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


def rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding."""
    v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
    phases = (phases % 1) * (2 * math.pi)
    phases_cos = torch.cos(phases)
    phases_sin = torch.sin(phases)
    return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)


# =============================================================================
# BDH ATTENTION LAYER
# =============================================================================

class BDHAttention(nn.Module):
    """BDH-style attention with bottleneck and sparse activations.
    
    Shape:
        Input: (B, nh, T, N) where N = D * mlp_mult // nh
        Output: (B, nh, T, N)
    """
    def __init__(self, n_embd: int, n_head: int, mlp_mult: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd * mlp_mult // n_head
        
        self.freqs = nn.Buffer(
            get_freqs(self.head_dim, theta=2**16, dtype=torch.float32).view(1, 1, 1, -1)
        )
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Causal attention with RoPE.
        
        Args:
            Q, K: (B, nh, T, N) - queries and keys (typically same)
            V: (B, 1, T, D) or (B, nh, T, N) - values
        """
        _, _, T, _ = Q.size()
        
        # Compute rotary phases
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        
        # Apply RoPE
        QR = rope(r_phases, Q)
        KR = rope(r_phases, K) if K is not Q else QR
        
        # Causal attention (strictly lower triangular)
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


# =============================================================================
# BDH BLOCK (single layer)
# =============================================================================

class BDHBlock(nn.Module):
    """Single BDH layer with bottleneck attention and gated MLP.
    
    Shape:
        Input: (B, T, D)
        Output: (B, T, D)
    """
    def __init__(self, n_embd: int, n_head: int, mlp_mult: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        N = n_embd * mlp_mult // n_head  # Bottleneck dimension per head
        
        # Encoder/decoder for bottleneck
        self.encoder = nn.Parameter(torch.zeros((n_head, n_embd, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((n_head, n_embd, N)).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros((n_head * N, n_embd)).normal_(std=0.02))
        
        # Attention and normalization
        self.attn = BDHAttention(n_embd, n_head, mlp_mult)
        self.ln = nn.LayerNorm(n_embd, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        B, T, D = x.size()
        nh = self.n_head
        N = D * self.encoder.size(-1) // D  # Recover N from encoder shape
        N = self.encoder.size(-1)
        
        # Add head dimension: (B, T, D) -> (B, 1, T, D)
        x_4d = x.unsqueeze(1)
        
        # Encode to bottleneck: (B, 1, T, D) @ (nh, D, N) -> (B, nh, T, N)
        x_latent = x_4d @ self.encoder
        x_sparse = F.relu(x_latent)
        
        # Attention in bottleneck space
        yKV = self.attn(Q=x_sparse, K=x_sparse, V=x_4d)
        yKV = self.ln(yKV)
        
        # Gated MLP
        y_latent = yKV @ self.encoder_v
        y_sparse = F.relu(y_latent)
        xy_sparse = x_sparse * y_sparse  # Gating
        xy_sparse = self.drop(xy_sparse)
        
        # Decode: (B, nh, T, N) -> (B, T, D)
        yMLP = xy_sparse.transpose(1, 2).reshape(B, T, nh * N) @ self.decoder
        y = self.ln(yMLP)
        
        return self.ln(x + y)


# =============================================================================
# STANDARD TRANSFORMER BLOCK (for comparison)
# =============================================================================

class TransformerBlock(nn.Module):
    """Standard transformer block with multi-head attention.
    
    Used when use_bdh_attention=False for comparison.
    """
    def __init__(self, n_embd: int, n_head: int, mlp_mult: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        
        # Self-attention
        qkv = self.qkv(self.ln1(x)).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(attn))
        
        # MLP
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# PATCH EMBEDDER
# =============================================================================

class PatchEmbedder(nn.Module):
    """Embed byte patches into global model dimension.
    
    Takes P consecutive byte embeddings and projects to global dimension.
    
    Shape:
        Input: (B, T) byte indices
        Output: (B, T/P, D_G) patch embeddings
    """
    def __init__(self, vocab_size: int, patch_size: int, local_dim: int, global_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.byte_embed = nn.Embedding(vocab_size, local_dim)
        self.patch_proj = nn.Linear(patch_size * local_dim, global_dim, bias=False)
        self.ln = nn.LayerNorm(global_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed patches.
        
        Args:
            x: (B, T) byte indices
        Returns:
            (B, T/P, D_G) patch embeddings
        """
        B, T = x.size()
        P = self.patch_size
        assert T % P == 0, f"Sequence length {T} not divisible by patch size {P}"
        
        # Embed bytes: (B, T) -> (B, T, D_L)
        byte_emb = self.byte_embed(x)
        
        # Reshape to patches: (B, T, D_L) -> (B, T/P, P*D_L)
        byte_emb = byte_emb.view(B, T // P, P * byte_emb.size(-1))
        
        # Project to global dim: (B, T/P, P*D_L) -> (B, T/P, D_G)
        patch_emb = self.patch_proj(byte_emb)
        return self.ln(patch_emb)


# =============================================================================
# GLOBAL MODEL
# =============================================================================

class GlobalModel(nn.Module):
    """Global model operating on patch-level representations.
    
    Shape:
        Input: (B, num_patches, D_G)
        Output: (B, num_patches, D_G)
    """
    def __init__(self, config: HierarchicalBDHConfig):
        super().__init__()
        self.config = config
        
        if config.use_bdh_attention:
            self.layers = nn.ModuleList([
                BDHBlock(
                    n_embd=config.global_n_embd,
                    n_head=config.global_n_head,
                    mlp_mult=config.global_mlp_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.global_n_layer)
            ])
        else:
            self.layers = nn.ModuleList([
                TransformerBlock(
                    n_embd=config.global_n_embd,
                    n_head=config.global_n_head,
                    mlp_mult=config.global_mlp_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.global_n_layer)
            ])
        
        self.ln_out = nn.LayerNorm(config.global_n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.ln_out(x)


# =============================================================================
# LOCAL MODEL
# =============================================================================

class LocalModel(nn.Module):
    """Local model operating on bytes within each patch.
    
    Receives global context from PREVIOUS patches (shifted by 1) to maintain
    proper causal ordering. This prevents information leakage where bytes
    could "see" future bytes through the global context.
    
    MEGABYTE-style causality:
        - Bytes in patch 0 receive learned start embedding (no prior context)
        - Bytes in patch i receive global context from patch i-1
    
    Shape:
        Input bytes: (B, T) 
        Input global: (B, num_patches, D_G)
        Output: (B, T, D_L)
    """
    def __init__(self, config: HierarchicalBDHConfig):
        super().__init__()
        self.config = config
        P = config.patch_size
        
        # Byte embedding (shared with patch embedder in full model)
        self.byte_embed = nn.Embedding(config.vocab_size, config.local_n_embd)
        
        # Project global context to local dimension
        self.global_proj = nn.Linear(config.global_n_embd, config.local_n_embd, bias=False)
        
        # Learned embedding for first patch (no prior global context)
        self.start_embed = nn.Parameter(torch.zeros(1, 1, config.local_n_embd))
        nn.init.normal_(self.start_embed, std=0.02)
        
        # Position embedding within patch (0 to P-1)
        self.pos_embed = nn.Embedding(P, config.local_n_embd)
        
        if config.use_bdh_attention:
            self.layers = nn.ModuleList([
                BDHBlock(
                    n_embd=config.local_n_embd,
                    n_head=config.local_n_head,
                    mlp_mult=config.local_mlp_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.local_n_layer)
            ])
        else:
            self.layers = nn.ModuleList([
                TransformerBlock(
                    n_embd=config.local_n_embd,
                    n_head=config.local_n_head,
                    mlp_mult=config.local_mlp_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.local_n_layer)
            ])
        
        self.ln_out = nn.LayerNorm(config.local_n_embd)
    
    def forward(self, x: torch.Tensor, global_ctx: torch.Tensor) -> torch.Tensor:
        """Process bytes with global context from PREVIOUS patches.
        
        Critical for causality: global context is shifted by 1 patch so that
        bytes in patch i only see global information from patches 0..i-1.
        
        Args:
            x: (B, T) byte indices
            global_ctx: (B, num_patches, D_G) global representations
        Returns:
            (B, T, D_L) local representations
        """
        B, T = x.size()
        P = self.config.patch_size
        num_patches = T // P
        D_L = self.config.local_n_embd
        
        # Embed bytes: (B, T) -> (B, T, D_L)
        byte_emb = self.byte_embed(x)
        
        # Project global context: (B, num_patches, D_G) -> (B, num_patches, D_L)
        global_proj = self.global_proj(global_ctx)
        
        # CRITICAL FIX: Shift global context by 1 patch for causality
        # - Patch 0 bytes get start_embed (no prior context)
        # - Patch i bytes get global context from patch i-1
        # Shape: (B, num_patches, D_L) -> (B, num_patches, D_L)
        start_expanded = self.start_embed.expand(B, 1, D_L)  # (B, 1, D_L)
        global_shifted = torch.cat([
            start_expanded,           # First patch gets learned start embedding
            global_proj[:, :-1, :]    # Remaining patches get previous patch's context
        ], dim=1)  # (B, num_patches, D_L)
        
        # Expand shifted global to byte level: (B, num_patches, D_L) -> (B, T, D_L)
        # Each patch's bytes receive the PREVIOUS patch's global context
        global_expanded = global_shifted.unsqueeze(2).expand(B, num_patches, P, D_L)
        global_expanded = global_expanded.reshape(B, T, D_L)
        
        # Position within patch: (P,) -> (1, T, D_L)
        pos_ids = torch.arange(P, device=x.device).repeat(num_patches)
        pos_emb = self.pos_embed(pos_ids).unsqueeze(0)
        
        # Combine: byte embedding + shifted global context + position
        h = byte_emb + global_expanded + pos_emb
        
        # Process through local layers
        # Reshape to process each patch independently for efficiency
        # (B, T, D_L) -> (B * num_patches, P, D_L)
        h = h.view(B * num_patches, P, D_L)
        
        for layer in self.layers:
            h = layer(h)
        
        # Reshape back: (B * num_patches, P, D_L) -> (B, T, D_L)
        h = h.view(B, T, D_L)
        
        return self.ln_out(h)


# =============================================================================
# HIERARCHICAL BDH MODEL
# =============================================================================

class HierarchicalBDH(nn.Module):
    """Hierarchical BDH with global (patch) and local (byte) models.
    
    Architecture:
        1. Patch Embedder: bytes -> patch embeddings
        2. Global Model: patch-level attention across sequence
        3. Local Model: byte-level attention within patches (with global context)
        4. LM Head: predict next byte
    
    Shape:
        Input: (B, T) byte indices
        Output: (B, T, vocab_size) logits
    """
    def __init__(self, config: HierarchicalBDHConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embedder = PatchEmbedder(
            vocab_size=config.vocab_size,
            patch_size=config.patch_size,
            local_dim=config.local_n_embd,
            global_dim=config.global_n_embd,
        )
        
        # Global model (patch-level)
        self.global_model = GlobalModel(config)
        
        # Local model (byte-level)
        self.local_model = LocalModel(config)
        
        # LM head
        self.lm_head = nn.Linear(config.local_n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie byte embeddings between patch embedder and local model
        self.local_model.byte_embed.weight = self.patch_embedder.byte_embed.weight
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Forward pass.
        
        Args:
            idx: (B, T) byte indices
            targets: (B, T) target byte indices (optional)
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar (if targets provided)
        """
        B, T = idx.size()
        P = self.config.patch_size
        
        # Pad sequence to be divisible by patch size
        if T % P != 0:
            pad_len = P - (T % P)
            idx = F.pad(idx, (0, pad_len), value=0)
            if targets is not None:
                targets = F.pad(targets, (0, pad_len), value=-100)  # Ignore padding in loss
            T = idx.size(1)
        
        # 1. Embed patches: (B, T) -> (B, T/P, D_G)
        patch_emb = self.patch_embedder(idx)
        
        # 2. Global model: (B, T/P, D_G) -> (B, T/P, D_G)
        global_out = self.global_model(patch_emb)
        
        # 3. Local model: (B, T), (B, T/P, D_G) -> (B, T, D_L)
        local_out = self.local_model(idx, global_out)
        
        # 4. LM head: (B, T, D_L) -> (B, T, vocab_size)
        logits = self.lm_head(local_out)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.
        
        Args:
            idx: (B, T) conditioning sequence
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling (None for full distribution)
        Returns:
            (B, T + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            'patch_embedder': count(self.patch_embedder),
            'global_model': count(self.global_model),
            'local_model': count(self.local_model),
            'lm_head': count(self.lm_head),
            'total': count(self),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_hierarchical_bdh(
    size: Literal['tiny', 'small', 'base', 'large'] = 'small',
    **config_overrides
) -> HierarchicalBDH:
    """Create a HierarchicalBDH model with preset size.
    
    Args:
        size: Model size preset
        **config_overrides: Override any config parameter
    
    Returns:
        HierarchicalBDH model
    """
    config_cls = {
        'tiny': HierarchicalBDHConfig.tiny,
        'small': HierarchicalBDHConfig.small,
        'base': HierarchicalBDHConfig.base,
        'large': HierarchicalBDHConfig.large,
    }[size]
    
    config = config_cls(**config_overrides)
    return HierarchicalBDH(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    # Test all size configurations
    for size in ['tiny', 'small', 'base', 'large']:
        print(f"\n{'='*60}")
        print(f"Testing {size.upper()} configuration")
        print('='*60)
        
        model = create_hierarchical_bdh(size)
        config = model.config
        
        print(f"\nConfig:")
        print(f"  Patch size: {config.patch_size}")
        print(f"  Global: {config.global_n_layer}L x {config.global_n_embd}D x {config.global_n_head}H")
        print(f"  Local:  {config.local_n_layer}L x {config.local_n_embd}D x {config.local_n_head}H")
        
        params = model.count_parameters()
        print(f"\nParameters:")
        for name, count in params.items():
            print(f"  {name}: {count:,} ({count/1e6:.1f}M)")
        
        # Test forward pass
        B, T = 2, 512
        x = torch.randint(0, 256, (B, T))
        targets = torch.randint(0, 256, (B, T))
        
        logits, loss = model(x, targets)
        print(f"\nForward pass:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Test generation
        prompt = torch.randint(0, 256, (1, 16))
        generated = model.generate(prompt, max_new_tokens=32)
        print(f"\nGeneration:")
        print(f"  Prompt: {prompt.shape}")
        print(f"  Generated: {generated.shape}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
