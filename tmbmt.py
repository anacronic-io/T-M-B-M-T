"""
T-M-B-M-T: A Symmetric Hybrid Architecture for Thermodynamic Reasoning

Implementation based on the paper:
"T-M-B-M-T: A Symmetric Hybrid Architecture for Thermodynamic Reasoning on TPUs"

Architecture: Transformer-Mamba-Boltzmann-Mamba-Transformer
- Phase 1: Transformer Encoder
- Bridge 1: Semantic → Temporal
- Phase 2: Mamba Encoder (SSM)
- Bridge 2: Temporal → Energetic
- Phase 3: GB-RBM Core (Gaussian-Bernoulli Restricted Boltzmann Machine)
- Bridge 3: Energetic → Temporal
- Phase 4: Mamba Decoder (SSM)
- Bridge 4: Temporal → Semantic
- Phase 5: Transformer Decoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from dataclasses import dataclass
from typing import Optional, Tuple
from einops import rearrange, repeat


@dataclass
class TMBMTConfig:
    """Configuration for T-M-B-M-T model."""
    # Model dimensions
    vocab_size: int = 32000
    d_transformer: int = 4096      # D_t: Transformer hidden dim
    d_ssm: int = 2048              # D_s: SSM/Mamba hidden dim
    d_rbm: int = 1024              # D_r: RBM visible units
    n_hidden_rbm: int = 2048       # K: RBM hidden units

    # Architecture
    n_transformer_layers: int = 24
    n_mamba_layers: int = 16
    n_heads: int = 32

    # SSM parameters
    ssm_state_size: int = 16       # N: State space dimension
    ssm_conv_size: int = 4
    ssm_expand: int = 2

    # RBM parameters
    gibbs_steps: int = 8           # K: Number of Gibbs sampling steps

    # Training
    max_seq_len: int = 8192
    dropout: float = 0.1

    # Loss coefficients
    beta: float = 0.1              # Consistency loss weight
    gamma: float = 0.01            # KL loss weight


# =============================================================================
# Spectral Normalization Utilities
# =============================================================================

def spectral_norm_power_iteration(W: torch.Tensor, u: torch.Tensor, n_iters: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Power iteration for spectral norm computation."""
    for _ in range(n_iters):
        v = F.normalize(W.t() @ u, dim=0)
        u = F.normalize(W @ v, dim=0)
    sigma = u @ W @ v
    return sigma, u


class SpectralNormLinear(nn.Module):
    """Linear layer with dynamic spectral normalization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer('u', F.normalize(torch.randn(out_features), dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma, u_new = spectral_norm_power_iteration(self.weight, self.u)
        self.u.copy_(u_new)
        W_normalized = self.weight / sigma
        return F.linear(x, W_normalized, self.bias)


# =============================================================================
# Mamba / SSM Block
# =============================================================================

class MambaBlock(nn.Module):
    """
    Selective State Space Model (Mamba) block.

    Implements the discretized LTI system:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t

    With input-dependent selectivity via delta, B, C projections.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, delta

        # A is learned but constrained to be negative (for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Convolution
        x_ssm = rearrange(x_ssm, 'b l d -> b d l')
        x_ssm = self.conv1d(x_ssm)[:, :, :seq_len]
        x_ssm = rearrange(x_ssm, 'b d l -> b l d')
        x_ssm = F.silu(x_ssm)

        # SSM parameters from input
        x_dbl = self.x_proj(x_ssm)
        delta, B, C = x_dbl[..., :1], x_dbl[..., 1:self.d_state+1], x_dbl[..., self.d_state+1:]
        delta = F.softplus(delta)

        # Discretization (ZOH)
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(delta * A)  # (batch, seq_len, d_state)
        B_bar = delta * B

        # Selective scan (simplified sequential version)
        y = self._selective_scan(x_ssm, A_bar, B_bar, C)

        # Skip connection and output
        y = y + x_ssm * self.D
        y = y * F.silu(z)

        return self.out_proj(y)

    def _selective_scan(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Sequential selective scan (for clarity; use parallel scan in production)."""
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[-1]

        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            h = A[:, t:t+1, :] * h + B[:, t:t+1, :] * x[:, t:t+1, :].unsqueeze(-1)
            y_t = (h * C[:, t:t+1, :].unsqueeze(2)).sum(dim=-1)
            outputs.append(y_t)

        return torch.cat(outputs, dim=1)


class MambaEncoder(nn.Module):
    """Stack of Mamba blocks for sequence compression."""

    def __init__(self, config: TMBMTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=config.d_ssm,
                d_state=config.ssm_state_size,
                d_conv=config.ssm_conv_size,
                expand=config.ssm_expand
            )
            for _ in range(config.n_mamba_layers)
        ])
        self.norm = nn.LayerNorm(config.d_ssm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x)


class MambaDecoder(nn.Module):
    """Stack of Mamba blocks for sequence expansion."""

    def __init__(self, config: TMBMTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=config.d_ssm,
                d_state=config.ssm_state_size,
                d_conv=config.ssm_conv_size,
                expand=config.ssm_expand
            )
            for _ in range(config.n_mamba_layers)
        ])
        self.norm = nn.LayerNorm(config.d_ssm)

    def forward(self, x: torch.Tensor, init_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x)


# =============================================================================
# Gaussian-Bernoulli RBM
# =============================================================================

class GaussianBernoulliRBM(nn.Module):
    """
    Gaussian-Bernoulli Restricted Boltzmann Machine.

    Energy function:
        E(v,h) = sum_i (v_i - b_i)^2 / (2*sigma_i^2)
               - sum_j c_j * h_j
               - sum_ij (v_i / sigma_i) * W_ij * h_j

    Uses Gumbel-Softmax for differentiable sampling and spectral normalization for stability.
    """

    def __init__(self, n_visible: int, n_hidden: int, sigma_init: float = 1.0):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Weights with spectral normalization
        self.W = SpectralNormLinear(n_visible, n_hidden, bias=False)

        # Biases
        self.visible_bias = nn.Parameter(torch.zeros(n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden))

        # Learnable variance for visible units
        self.log_sigma = nn.Parameter(torch.full((n_visible,), math.log(sigma_init)))

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute energy E(v, h)."""
        sigma = self.sigma

        # Gaussian term for visible units
        gaussian_term = ((v - self.visible_bias) ** 2 / (2 * sigma ** 2)).sum(dim=-1)

        # Hidden bias term
        hidden_term = (h * self.hidden_bias).sum(dim=-1)

        # Interaction term
        v_normalized = v / sigma
        interaction = (h * self.W(v_normalized)).sum(dim=-1)

        return gaussian_term - hidden_term - interaction

    def sample_hidden(self, v: torch.Tensor, tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given visible units using Gumbel-Softmax.

        Args:
            v: Visible units (batch, n_visible)
            tau: Temperature for Gumbel-Softmax

        Returns:
            h_sample: Sampled hidden units (soft during training)
            h_prob: Probability of h=1
        """
        v_normalized = v / self.sigma
        logits = self.W(v_normalized) + self.hidden_bias

        # Gumbel-Softmax for differentiable sampling
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            h_sample = torch.sigmoid((logits + gumbel_noise) / tau)
        else:
            h_sample = torch.sigmoid(logits)

        h_prob = torch.sigmoid(logits)
        return h_sample, h_prob

    def sample_visible(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units.

        Args:
            h: Hidden units (batch, n_hidden)

        Returns:
            v_sample: Sampled visible units
            v_mean: Mean of visible distribution
        """
        # Mean-field reconstruction
        v_mean = self.W.weight.t() @ h.unsqueeze(-1)
        v_mean = v_mean.squeeze(-1) * self.sigma + self.visible_bias

        # For continuous visible units, we use the mean (or add Gaussian noise)
        v_sample = v_mean

        return v_sample, v_mean

    def gibbs_step(self, v: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """One step of Gibbs sampling: v → h → v'."""
        h, _ = self.sample_hidden(v, tau)
        v_new, _ = self.sample_visible(h)
        return v_new

    def forward(self, v_in: torch.Tensor, k_steps: int = 8, tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run K steps of Gibbs sampling for energy minimization.

        Args:
            v_in: Initial visible state (batch, n_visible)
            k_steps: Number of Gibbs sampling steps
            tau: Temperature for Gumbel-Softmax

        Returns:
            v_opt: Optimized visible state
            energies: Energy at each step (for monitoring)
            h_final: Final hidden state
        """
        v = v_in
        energies = []

        for k in range(k_steps):
            h, _ = self.sample_hidden(v, tau)
            v_new, _ = self.sample_visible(h)

            # Track energy
            with torch.no_grad():
                h_hard = (h > 0.5).float()
                e = self.energy(v, h_hard)
                energies.append(e.mean().item())

            v = v_new

        h_final, _ = self.sample_hidden(v, tau)
        return v, torch.tensor(energies), h_final


# =============================================================================
# Bridges
# =============================================================================

class Bridge1(nn.Module):
    """
    Bridge 1: Semantic → Temporal

    Transforms Transformer output to Mamba input with GLU gating.
    X_ssm = GLU(LayerNorm(W1 * H_trans))
    """

    def __init__(self, d_transformer: int, d_ssm: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_transformer)
        self.proj = nn.Linear(d_transformer, d_ssm * 2)  # *2 for GLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)  # GLU


class Bridge2(nn.Module):
    """
    Bridge 2: Temporal → Energetic

    Attention-weighted pooling to collapse sequence into thought vector.
    v_in = sum_t alpha_t * h_t
    alpha_t = softmax(q^T * W2 * h_t)
    """

    def __init__(self, d_ssm: int, d_rbm: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_ssm) * 0.02)
        self.key_proj = nn.Linear(d_ssm, d_ssm)
        self.value_proj = nn.Linear(d_ssm, d_rbm)

    def forward(self, h_mamba: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_mamba: (batch, seq_len, d_ssm)
        Returns:
            v_in: (batch, d_rbm)
        """
        # Attention weights
        keys = self.key_proj(h_mamba)  # (batch, seq_len, d_ssm)
        attn_logits = torch.einsum('d,bsd->bs', self.query, keys)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Weighted pooling
        values = self.value_proj(h_mamba)  # (batch, seq_len, d_rbm)
        v_in = torch.einsum('bs,bsd->bd', attn_weights, values)

        return v_in


class Bridge3(nn.Module):
    """
    Bridge 3: Energetic → Temporal

    Initializes Mamba decoder state from optimized RBM representation.
    h_0^dec = tanh(W3 * v_opt + b3)
    """

    def __init__(self, d_rbm: int, d_ssm: int):
        super().__init__()
        self.proj = nn.Linear(d_rbm, d_ssm)

    def forward(self, v_opt: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            v_opt: (batch, d_rbm)
            seq_len: Target sequence length
        Returns:
            h_init: (batch, seq_len, d_ssm)
        """
        h_init = torch.tanh(self.proj(v_opt))
        # Expand to sequence
        h_init = h_init.unsqueeze(1).expand(-1, seq_len, -1)
        return h_init


class Bridge4(nn.Module):
    """
    Bridge 4: Temporal → Semantic

    Expands Mamba output back to Transformer dimension with residual.
    H_trans = H_mamba + LayerNorm(W4 * H_mamba)
    """

    def __init__(self, d_ssm: int, d_transformer: int):
        super().__init__()
        self.proj = nn.Linear(d_ssm, d_transformer)
        self.norm = nn.LayerNorm(d_transformer)
        self.residual_proj = nn.Linear(d_ssm, d_transformer)

    def forward(self, h_mamba: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(h_mamba)
        x = self.proj(h_mamba)
        x = self.norm(x)
        return residual + x


# =============================================================================
# Transformer Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.n_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h s d -> b s (h d)')

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(self, config: TMBMTConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_transformer)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_transformer)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config.d_transformer, config.n_heads, config.dropout)
            for _ in range(config.n_transformer_layers)
        ])
        self.norm = nn.LayerNorm(config.d_transformer)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)

        x = self.embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Transformer decoder stack with causal masking."""

    def __init__(self, config: TMBMTConfig):
        super().__init__()
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_transformer)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config.d_transformer, config.n_heads, config.dropout)
            for _ in range(config.n_transformer_layers)
        ])
        self.norm = nn.LayerNorm(config.d_transformer)
        self.lm_head = nn.Linear(config.d_transformer, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)

        x = x + self.pos_embed(positions)
        x = self.dropout(x)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


# =============================================================================
# Full T-M-B-M-T Model
# =============================================================================

class TMBMT(nn.Module):
    """
    T-M-B-M-T: Transformer-Mamba-Boltzmann-Mamba-Transformer

    A symmetric hybrid architecture with energy-based deliberation.
    """

    def __init__(self, config: TMBMTConfig):
        super().__init__()
        self.config = config

        # Phase 1: Transformer Encoder
        self.transformer_encoder = TransformerEncoder(config)

        # Bridge 1: Semantic → Temporal
        self.bridge1 = Bridge1(config.d_transformer, config.d_ssm)

        # Phase 2: Mamba Encoder
        self.mamba_encoder = MambaEncoder(config)

        # Bridge 2: Temporal → Energetic
        self.bridge2 = Bridge2(config.d_ssm, config.d_rbm)

        # Phase 3: GB-RBM Core
        self.rbm = GaussianBernoulliRBM(config.d_rbm, config.n_hidden_rbm)

        # Bridge 3: Energetic → Temporal
        self.bridge3 = Bridge3(config.d_rbm, config.d_ssm)

        # Phase 4: Mamba Decoder
        self.mamba_decoder = MambaDecoder(config)

        # Bridge 4: Temporal → Semantic
        self.bridge4 = Bridge4(config.d_ssm, config.d_transformer)

        # Phase 5: Transformer Decoder
        self.transformer_decoder = TransformerDecoder(config)

        # Temperature for Gumbel-Softmax (annealed during training)
        self.register_buffer('tau', torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass through the full T-M-B-M-T architecture.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            labels: Target token IDs for loss computation

        Returns:
            Dictionary containing logits, loss components, and diagnostics
        """
        batch, seq_len = input_ids.shape

        # Phase 1: Transformer Encoder
        h_trans_enc = self.transformer_encoder(input_ids)

        # Bridge 1: Semantic → Temporal
        h_ssm = self.bridge1(h_trans_enc)

        # Phase 2: Mamba Encoder
        h_mamba_enc = self.mamba_encoder(h_ssm)

        # Bridge 2: Temporal → Energetic (collapse to thought vector)
        v_in = self.bridge2(h_mamba_enc)

        # Phase 3: GB-RBM Core (energy minimization)
        v_opt, energies, h_hidden = self.rbm(
            v_in,
            k_steps=self.config.gibbs_steps,
            tau=self.tau.item()
        )

        # Bridge 3: Energetic → Temporal
        h_init = self.bridge3(v_opt, seq_len)

        # Phase 4: Mamba Decoder
        h_mamba_dec = self.mamba_decoder(h_init)

        # Bridge 4: Temporal → Semantic
        h_trans_dec = self.bridge4(h_mamba_dec)

        # Phase 5: Transformer Decoder
        logits = self.transformer_decoder(h_trans_dec)

        # Compute losses
        output = {
            'logits': logits,
            'v_in': v_in,
            'v_opt': v_opt,
            'energies': energies,
        }

        if labels is not None:
            # NLL Loss (cross-entropy)
            loss_nll = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

            # Consistency Loss: encourage encoder to produce near-optimal states
            loss_consistency = F.mse_loss(v_in, v_opt.detach())

            # KL Loss: regularize hidden unit distribution
            h_prob = torch.sigmoid(self.rbm.W(v_opt / self.rbm.sigma) + self.rbm.hidden_bias)
            loss_kl = (h_prob * torch.log(h_prob + 1e-10) +
                      (1 - h_prob) * torch.log(1 - h_prob + 1e-10)).mean()

            # Total loss
            loss_total = (
                loss_nll +
                self.config.beta * loss_consistency +
                self.config.gamma * loss_kl
            )

            output.update({
                'loss': loss_total,
                'loss_nll': loss_nll,
                'loss_consistency': loss_consistency,
                'loss_kl': loss_kl,
            })

        return output

    def set_temperature(self, tau: float):
        """Set Gumbel-Softmax temperature."""
        self.tau.fill_(tau)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()

        for _ in range(max_new_tokens):
            # Forward pass
            output = self(input_ids)
            logits = output['logits'][:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# =============================================================================
# Training Utilities
# =============================================================================

class TMBMTTrainer:
    """Training loop for T-M-B-M-T with temperature annealing."""

    def __init__(
        self,
        model: TMBMT,
        optimizer: torch.optim.Optimizer,
        tau_start: float = 1.0,
        tau_end: float = 0.5,
        anneal_steps: int = 100000
    ):
        self.model = model
        self.optimizer = optimizer
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.anneal_steps = anneal_steps
        self.step = 0

    def get_tau(self) -> float:
        """Compute current temperature based on step."""
        if self.step >= self.anneal_steps:
            return self.tau_end
        progress = self.step / self.anneal_steps
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        """Execute one training step."""
        self.model.train()

        # Update temperature
        tau = self.get_tau()
        self.model.set_temperature(tau)

        # Forward pass
        output = self.model(input_ids, labels)
        loss = output['loss']

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.step += 1

        return {
            'loss': loss.item(),
            'loss_nll': output['loss_nll'].item(),
            'loss_consistency': output['loss_consistency'].item(),
            'loss_kl': output['loss_kl'].item(),
            'tau': tau,
            'energies': output['energies'].tolist(),
        }


# =============================================================================
# Example Usage
# =============================================================================

def create_small_model() -> TMBMT:
    """Create a small model for testing."""
    config = TMBMTConfig(
        vocab_size=32000,
        d_transformer=512,
        d_ssm=256,
        d_rbm=128,
        n_hidden_rbm=256,
        n_transformer_layers=4,
        n_mamba_layers=4,
        n_heads=8,
        ssm_state_size=8,
        gibbs_steps=4,
        max_seq_len=512,
    )
    return TMBMT(config)


if __name__ == "__main__":
    # Test the model
    print("Creating T-M-B-M-T model...")
    model = create_small_model()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    labels = torch.randint(0, 32000, (batch_size, seq_len))

    print("\nRunning forward pass...")
    output = model(input_ids, labels)

    print(f"Logits shape: {output['logits'].shape}")
    print(f"Total loss: {output['loss'].item():.4f}")
    print(f"NLL loss: {output['loss_nll'].item():.4f}")
    print(f"Consistency loss: {output['loss_consistency'].item():.4f}")
    print(f"KL loss: {output['loss_kl'].item():.4f}")
    print(f"Energy trajectory: {output['energies'].tolist()}")

    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, 32000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated sequence shape: {generated.shape}")

    print("\nT-M-B-M-T model test complete!")
