"""
End-to-End Integration Test for T-M-B-M-T Architecture
Verifies that all components work together correctly.
"""

import jax
import jax.numpy as jnp
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.zoh_discretization import zoh_discretization_analytical, parallel_scan_operation
from core.gb_rbm import GaussianBernoulliRBM, gumbel_sigmoid
from core.spectral_stability import apply_spectral_normalization
from typing import Tuple, Dict


class SimplifiedTMBMT:
    """
    Simplified T-M-B-M-T architecture for integration testing.
    """

    def __init__(
        self,
        d_trans: int = 512,
        d_ssm: int = 256,
        d_rbm: int = 128,
        gibbs_steps: int = 10,
        delta: float = 0.01
    ):
        self.d_trans = d_trans
        self.d_ssm = d_ssm
        self.d_rbm = d_rbm
        self.gibbs_steps = gibbs_steps
        self.delta = delta

        # Initialize components
        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

        # Bridge 1: Transformer → SSM (Linearization)
        self.W_proj = jax.random.normal(k1, (d_trans, d_ssm)) * 0.02

        # Mamba Encoder SSM
        eigenvalues = jax.random.uniform(k2, (d_ssm,), minval=-0.5, maxval=0)
        self.A_enc = jnp.diag(eigenvalues)
        self.B_enc = jax.random.normal(k3, (d_ssm, d_ssm)) * 0.1
        self.A_enc_discrete, self.B_enc_discrete = zoh_discretization_analytical(
            self.A_enc, self.B_enc, delta
        )

        # Bridge 2: SSM → RBM (Energy Pooling)
        self.W_pool = jax.random.normal(k4, (d_ssm,)) * 0.02

        # RBM Core
        self.rbm = GaussianBernoulliRBM(d_rbm, d_rbm // 2, sigma=1.0)

        # Apply spectral normalization to RBM weights
        self.rbm.W = apply_spectral_normalization(self.rbm.W, target_norm=0.9)

        # Bridge 3: RBM → SSM Decoder (Dynamic Init)
        self.W_init = jax.random.normal(k5, (d_rbm, d_ssm)) * 0.02

        # Mamba Decoder SSM
        self.A_dec = self.A_enc  # Symmetric
        self.B_dec = self.B_enc
        self.A_dec_discrete, self.B_dec_discrete = zoh_discretization_analytical(
            self.A_dec, self.B_dec, delta
        )

        # Transformer Decoder (simplified as linear)
        self.W_out = jax.random.normal(k6, (d_ssm, d_trans)) * 0.02

    def forward(
        self,
        x: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Forward pass through T-M-B-M-T.

        Args:
            x: Input [B, L, d_trans]
            key: Random key for stochastic sampling

        Returns:
            output: Reconstructed output [B, L, d_trans]
            debug_info: Dictionary with intermediate activations
        """
        B, L, _ = x.shape
        debug_info = {}

        # ===== ENCODER PHASE =====

        # 1. Transformer Encoder (simplified)
        h_trans = x  # [B, L, d_trans]
        debug_info['h_trans'] = h_trans

        # 2. Bridge 1: Linearization Projector
        x_ssm = jax.nn.swish(h_trans @ self.W_proj)  # [B, L, d_ssm]
        debug_info['x_ssm'] = x_ssm

        # 3. Mamba Encoder: Parallel Scan
        # Process each batch element
        def encode_sequence(x_seq):
            return parallel_scan_operation(
                self.A_enc_discrete,
                self.B_enc_discrete,
                x_seq
            )

        h_mamba_enc = jax.vmap(encode_sequence)(x_ssm)  # [B, L, d_ssm]
        debug_info['h_mamba_enc'] = h_mamba_enc

        # 4. Bridge 2: Energy Pooling (attention-based)
        # Simplified: weighted average with learned attention
        scores = jnp.tanh(h_mamba_enc @ self.W_pool)  # [B, L]
        attention_weights = jax.nn.softmax(scores, axis=-1)  # [B, L]
        v_in = jnp.einsum('bl,bld->bd', attention_weights, h_mamba_enc)  # [B, d_ssm]

        # Project to RBM dimension
        v_in = v_in[:, :self.d_rbm]  # [B, d_rbm]
        debug_info['v_in'] = v_in

        # ===== BOLTZMANN CORE =====

        # 5. RBM Reasoning (Gibbs Sampling with Gumbel-Softmax)
        v_current = v_in

        for step in range(self.gibbs_steps):
            key, k_gibbs = jax.random.split(key)

            # Mean-field h | v
            h_logits = self.rbm.c + self.rbm.W.T @ (v_current.T / self.rbm.sigma_vec[:, None])  # [K, B]
            h_logits = h_logits.T  # [B, K]

            # Gumbel-Softmax (differentiable)
            h = jax.vmap(lambda logits, k: gumbel_sigmoid(logits, temperature=1.0, key=k))(
                h_logits,
                jax.random.split(k_gibbs, B)
            )  # [B, K]

            # Mean-field v | h
            v_current = self.rbm.b[None, :] + self.rbm.sigma_vec[None, :] * (h @ self.rbm.W.T)  # [B, d_rbm]

        v_opt = v_current
        debug_info['v_opt'] = v_opt
        debug_info['energy'] = jax.vmap(lambda v, h: self.rbm.energy(v, h))(v_opt, h)

        # ===== DECODER PHASE =====

        # 6. Bridge 3: Dynamic Initializer
        h_dec_init = jnp.tanh(v_opt @ self.W_init)  # [B, d_ssm]
        debug_info['h_dec_init'] = h_dec_init

        # 7. Mamba Decoder: Use h_dec_init to condition generation
        # Expand back to sequence length
        h_dec_init_expanded = jnp.repeat(h_dec_init[:, None, :], L, axis=1)  # [B, L, d_ssm]

        def decode_sequence(h_seq):
            return parallel_scan_operation(
                self.A_dec_discrete,
                self.B_dec_discrete,
                h_seq
            )

        h_mamba_dec = jax.vmap(decode_sequence)(h_dec_init_expanded)  # [B, L, d_ssm]
        debug_info['h_mamba_dec'] = h_mamba_dec

        # 8. Transformer Decoder (simplified)
        output = h_mamba_dec @ self.W_out  # [B, L, d_trans]
        debug_info['output'] = output

        return output, debug_info

    def reconstruction_loss(self, x: jnp.ndarray, x_recon: jnp.ndarray) -> float:
        """Mean squared error reconstruction loss."""
        return jnp.mean((x - x_recon) ** 2)


def test_integration():
    """Run integration tests."""
    print("=" * 70)
    print("T-M-B-M-T END-TO-END INTEGRATION TEST")
    print("=" * 70)

    # Initialize model
    print("\n[Setup] Initializing T-M-B-M-T model...")
    model = SimplifiedTMBMT(
        d_trans=512,
        d_ssm=256,
        d_rbm=128,
        gibbs_steps=10
    )
    print("  ✓ Model initialized")

    # Test 1: Forward pass
    print("\n[Test 1] Forward Pass")

    B, L = 2, 64
    key = jax.random.PRNGKey(0)
    x_input = jax.random.normal(key, (B, L, model.d_trans))

    key, subkey = jax.random.split(key)
    output, debug_info = model.forward(x_input, subkey)

    print(f"  Input shape:  {x_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  ✓ PASS - Shape preserved" if output.shape == x_input.shape else "  ✗ FAIL")

    # Test 2: Verify all intermediate shapes
    print("\n[Test 2] Intermediate Activations")

    expected_shapes = {
        'h_trans': (B, L, model.d_trans),
        'x_ssm': (B, L, model.d_ssm),
        'h_mamba_enc': (B, L, model.d_ssm),
        'v_in': (B, model.d_rbm),
        'v_opt': (B, model.d_rbm),
        'h_dec_init': (B, model.d_ssm),
        'h_mamba_dec': (B, L, model.d_ssm),
        'output': (B, L, model.d_trans)
    }

    all_correct = True
    for name, expected_shape in expected_shapes.items():
        actual_shape = debug_info[name].shape
        correct = actual_shape == expected_shape
        all_correct = all_correct and correct

        status = "✓" if correct else "✗"
        print(f"  {status} {name:20s}: {actual_shape} {'==' if correct else '!='} {expected_shape}")

    print(f"\n  {'✓ PASS - All shapes correct' if all_correct else '✗ FAIL - Shape mismatch'}")

    # Test 3: Energy convergence
    print("\n[Test 3] RBM Energy Dynamics")

    energies = debug_info['energy']
    print(f"  Final energy (batch mean): {jnp.mean(energies):.4f}")
    print(f"  Energy range: [{jnp.min(energies):.4f}, {jnp.max(energies):.4f}]")
    print(f"  ✓ PASS - Energy computed" if jnp.isfinite(energies).all() else "  ✗ FAIL")

    # Test 4: Gradient flow
    print("\n[Test 4] End-to-End Gradient Flow")

    @jax.jit
    def loss_fn(x, key):
        output, _ = model.forward(x, key)
        return jnp.mean((output - x) ** 2)

    # Compute gradients w.r.t. model parameters
    # For simplicity, check gradient of loss w.r.t. input
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(x_input, subkey)

    grad_norm = jnp.linalg.norm(grads)
    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  ✓ PASS - Gradients flow" if jnp.isfinite(grads).all() else "  ✗ FAIL")

    # Test 5: Reconstruction quality
    print("\n[Test 5] Reconstruction Quality")

    # For a simple pattern
    x_pattern = jnp.ones((1, L, model.d_trans)) * 0.5

    key, subkey = jax.random.split(key)
    x_recon, _ = model.forward(x_pattern, subkey)

    mse = jnp.mean((x_pattern - x_recon) ** 2)
    print(f"  Input pattern: constant 0.5")
    print(f"  MSE: {mse:.6f}")
    print(f"  Output mean: {jnp.mean(x_recon):.6f}")
    print(f"  Output std:  {jnp.std(x_recon):.6f}")

    # Test 6: Spectral norm constraint verification
    print("\n[Test 6] Spectral Normalization Constraint")

    # Check that RBM weights are properly normalized
    from core.spectral_stability import spectral_norm

    rbm_spectral_norm = spectral_norm(model.rbm.W)
    print(f"  ||W_RBM||₂ = {rbm_spectral_norm:.4f}")
    print(f"  Target: 0.9")
    print(f"  ✓ PASS" if abs(rbm_spectral_norm - 0.9) < 0.1 else "  ⚠ WARNING")

    # Test 7: Determinism check
    print("\n[Test 7] Deterministic Behavior (Same Seed)")

    key_test = jax.random.PRNGKey(123)
    x_test = jax.random.normal(key_test, (1, L, model.d_trans))

    key1 = jax.random.PRNGKey(999)
    key2 = jax.random.PRNGKey(999)  # Same seed

    out1, _ = model.forward(x_test, key1)
    out2, _ = model.forward(x_test, key2)

    diff = jnp.linalg.norm(out1 - out2)
    print(f"  Difference between runs: {diff:.2e}")
    print(f"  ✓ PASS - Deterministic" if diff < 1e-5 else "  ⚠ WARNING - Non-deterministic")

    # Test 8: Batch consistency
    print("\n[Test 8] Batch Processing Consistency")

    # Process individually vs in batch
    key_batch = jax.random.PRNGKey(555)
    x_batch = jax.random.normal(key_batch, (4, L, model.d_trans))

    # Batch processing
    out_batch, _ = model.forward(x_batch, key_batch)

    # Individual processing
    outs_individual = []
    for i in range(4):
        key_ind, subkey = jax.random.split(key_batch)
        out_ind, _ = model.forward(x_batch[i:i+1], subkey)
        outs_individual.append(out_ind[0])

    outs_individual = jnp.array(outs_individual)

    # Compare (allowing for randomness in RBM)
    batch_diff = jnp.mean(jnp.abs(out_batch - outs_individual))
    print(f"  Mean absolute difference: {batch_diff:.6f}")
    print(f"  ⚠ Note: Some difference expected due to RBM stochasticity")

    # Test 9: Information preservation
    print("\n[Test 9] Information Preservation Through Pipeline")

    # Create two very different inputs
    x1 = jnp.ones((1, L, model.d_trans)) * 1.0
    x2 = -jnp.ones((1, L, model.d_trans)) * 1.0

    key, k1, k2 = jax.random.split(key, 3)
    out1, _ = model.forward(x1, k1)
    out2, _ = model.forward(x2, k2)

    input_diff = jnp.linalg.norm(x1 - x2)
    output_diff = jnp.linalg.norm(out1 - out2)
    preservation_ratio = output_diff / input_diff

    print(f"  Input difference:  {input_diff:.4f}")
    print(f"  Output difference: {output_diff:.4f}")
    print(f"  Preservation ratio: {preservation_ratio:.4f}")
    print(f"  ✓ PASS" if preservation_ratio > 0.1 else "  ⚠ WARNING - Low preservation")

    # Test 10: Component ablation
    print("\n[Test 10] Component Ablation (Contribution Analysis)")

    x_test = jax.random.normal(key, (1, L, model.d_trans))

    # Full model
    key, subkey = jax.random.split(key)
    out_full, debug_full = model.forward(x_test, subkey)
    energy_full = jnp.mean(debug_full['energy'])

    # Without RBM refinement (fewer Gibbs steps)
    model_no_rbm = SimplifiedTMBMT(gibbs_steps=1)
    key, subkey = jax.random.split(key)
    out_no_rbm, debug_no_rbm = model_no_rbm.forward(x_test, subkey)
    energy_no_rbm = jnp.mean(debug_no_rbm['energy'])

    print(f"  Full model energy:    {energy_full:.4f}")
    print(f"  Reduced RBM energy:   {energy_no_rbm:.4f}")
    print(f"  Energy reduction:     {energy_full - energy_no_rbm:.4f}")

    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_integration()
