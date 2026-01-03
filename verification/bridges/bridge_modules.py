"""
Bridge Module Implementations and Verification
Implements the three critical bridges in the T-M-B-M-T architecture.
"""

import jax
import jax.numpy as jnp
from typing import Tuple
import flax.linen as nn
from functools import partial


class LinearizationProjector(nn.Module):
    """
    Bridge 1: Transformer → Mamba
    Transforms rich semantic representation into SSM latent space.
    """
    ssm_dim: int
    use_glu: bool = True

    @nn.compact
    def __call__(self, H_trans: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            H_trans: Transformer output [B, L, D_trans]

        Returns:
            X_ssm: SSM input [B, L, D_ssm]
        """
        # Layer normalization
        x = nn.LayerNorm()(H_trans)

        if self.use_glu:
            # Gated Linear Unit for feature selection
            # Split into two paths
            gate = nn.Dense(self.ssm_dim, name='proj_gate')(x)
            value = nn.Dense(self.ssm_dim, name='proj_value')(x)

            # Swish activation (as per paper)
            gate = jax.nn.swish(gate)

            # Element-wise gating
            X_ssm = gate * value
        else:
            # Simple linear projection
            X_ssm = nn.Dense(self.ssm_dim, name='proj_linear')(x)
            X_ssm = jax.nn.swish(X_ssm)

        return X_ssm


class EnergyPooling(nn.Module):
    """
    Bridge 2: Mamba Encoder → RBM
    Collapses temporal dimension using attention weighted by energetic importance.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, h_mamba: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            h_mamba: Mamba encoder output [B, L, D]

        Returns:
            v_in: Visible units for RBM [B, D]
        """
        B, L, D = h_mamba.shape

        # Query vector for attention
        q = nn.Dense(self.hidden_dim, name='attention_query')(h_mamba)  # [B, L, hidden]

        # Importance scores
        W = self.param('attention_W', nn.initializers.normal(0.02), (self.hidden_dim,))
        scores = jnp.tanh(q) @ W  # [B, L]

        # Softmax normalization
        attention_weights = jax.nn.softmax(scores, axis=-1)  # [B, L]

        # Weighted sum
        v_in = jnp.einsum('bl,bld->bd', attention_weights, h_mamba)  # [B, D]

        return v_in


class DynamicInitializer(nn.Module):
    """
    Bridge 3: RBM → Mamba Decoder
    Uses purified thought vector to initialize decoder memory.
    """
    decoder_dim: int

    @nn.compact
    def __call__(self, v_opt: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            v_opt: Optimized visible units from RBM [B, D_rbm]

        Returns:
            h_dec_0: Initial decoder state [B, D_decoder]
        """
        # Linear projection
        h = nn.Dense(self.decoder_dim, name='init_projection')(v_opt)

        # Tanh activation for bounded initialization
        h_dec_0 = jnp.tanh(h)

        return h_dec_0


def verify_bridges():
    """Verification tests for bridge modules."""
    print("=" * 70)
    print("BRIDGE MODULES VERIFICATION")
    print("=" * 70)

    key = jax.random.PRNGKey(42)
    B, L, D_trans, D_ssm, D_rbm, D_decoder = 4, 128, 512, 256, 128, 256

    # Test 1: Linearization Projector
    print("\n[Test 1] Linearization Projector (Bridge 1)")

    # Initialize module
    bridge1 = LinearizationProjector(ssm_dim=D_ssm, use_glu=True)
    H_trans = jax.random.normal(key, (B, L, D_trans))

    # Initialize parameters
    variables = bridge1.init(key, H_trans)
    params = variables['params']

    # Forward pass
    X_ssm = bridge1.apply(variables, H_trans)

    print(f"  Input shape:  {H_trans.shape}")
    print(f"  Output shape: {X_ssm.shape}")
    print(f"  Expected:     ({B}, {L}, {D_ssm})")

    # Verify shape
    shape_correct = X_ssm.shape == (B, L, D_ssm)
    print(f"  ✓ PASS - Shape correct" if shape_correct else "  ✗ FAIL")

    # Verify GLU gating reduces dimensionality properly
    print(f"  Output range: [{jnp.min(X_ssm):.4f}, {jnp.max(X_ssm):.4f}]")
    print(f"  Output mean:  {jnp.mean(X_ssm):.4f}")
    print(f"  Output std:   {jnp.std(X_ssm):.4f}")

    # Verify differentiability
    def loss_fn(params, x):
        out = bridge1.apply({'params': params}, x)
        return jnp.sum(out ** 2)

    grads = jax.grad(loss_fn)(params, H_trans)
    grad_norm = jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(x ** 2),
        grads,
        0.0
    )
    print(f"  Gradient norm: {jnp.sqrt(grad_norm):.4f}")
    print(f"  ✓ PASS - Differentiable" if jnp.isfinite(grad_norm) else "  ✗ FAIL")

    # Test 2: Energy Pooling
    print("\n[Test 2] Energy Pooling (Bridge 2)")

    bridge2 = EnergyPooling(hidden_dim=256)
    h_mamba = jax.random.normal(key, (B, L, D_ssm))

    variables2 = bridge2.init(key, h_mamba)
    params2 = variables2['params']

    v_in = bridge2.apply(variables2, h_mamba)

    print(f"  Input shape:  {h_mamba.shape}")
    print(f"  Output shape: {v_in.shape}")
    print(f"  Expected:     ({B}, {D_ssm})")

    shape_correct = v_in.shape == (B, D_ssm)
    print(f"  ✓ PASS - Shape correct" if shape_correct else "  ✗ FAIL")

    # Verify attention mechanism
    # Test that different sequences give different pooling
    h_mamba_2 = jax.random.normal(jax.random.PRNGKey(123), (B, L, D_ssm))
    v_in_2 = bridge2.apply(variables2, h_mamba_2)

    difference = jnp.linalg.norm(v_in - v_in_2)
    print(f"  Difference for different inputs: {difference:.4f}")
    print(f"  ✓ PASS - Attention is input-dependent" if difference > 0.1 else "  ⚠ WARNING")

    # Verify temporal collapse (no temporal dimension in output)
    print(f"  Temporal dimension collapsed: {v_in.ndim == 2}")
    print(f"  ✓ PASS - Temporal collapse" if v_in.ndim == 2 else "  ✗ FAIL")

    # Test 3: Dynamic Initializer
    print("\n[Test 3] Dynamic Initializer (Bridge 3)")

    bridge3 = DynamicInitializer(decoder_dim=D_decoder)
    v_opt = jax.random.normal(key, (B, D_rbm))

    variables3 = bridge3.init(key, v_opt)
    params3 = variables3['params']

    h_dec_0 = bridge3.apply(variables3, v_opt)

    print(f"  Input shape:  {v_opt.shape}")
    print(f"  Output shape: {h_dec_0.shape}")
    print(f"  Expected:     ({B}, {D_decoder})")

    shape_correct = h_dec_0.shape == (B, D_decoder)
    print(f"  ✓ PASS - Shape correct" if shape_correct else "  ✗ FAIL")

    # Verify bounded output (tanh should keep values in [-1, 1])
    output_bounded = (jnp.min(h_dec_0) >= -1.0) and (jnp.max(h_dec_0) <= 1.0)
    print(f"  Output range: [{jnp.min(h_dec_0):.4f}, {jnp.max(h_dec_0):.4f}]")
    print(f"  ✓ PASS - Bounded by tanh" if output_bounded else "  ✗ FAIL")

    # Test 4: Information flow through bridges
    print("\n[Test 4] End-to-End Information Flow")

    # Simulate full bridge chain
    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    # Input: Transformer output
    H_trans_test = jax.random.normal(k1, (B, L, D_trans))

    # Bridge 1: Transformer → SSM
    bridge1_vars = bridge1.init(k2, H_trans_test)
    X_ssm_test = bridge1.apply(bridge1_vars, H_trans_test)

    # Simulate Mamba encoding (identity for test)
    h_mamba_test = X_ssm_test  # [B, L, D_ssm]

    # Bridge 2: SSM → RBM
    bridge2_vars = bridge2.init(k3, h_mamba_test)
    v_in_test = bridge2.apply(bridge2_vars, h_mamba_test)  # [B, D_ssm]

    # Simulate RBM processing (identity for test)
    v_opt_test = v_in_test

    # Pad to RBM dimension if needed
    if v_opt_test.shape[-1] != D_rbm:
        padding = D_rbm - v_opt_test.shape[-1]
        v_opt_test = jnp.pad(v_opt_test, ((0, 0), (0, padding)))

    # Bridge 3: RBM → Decoder
    bridge3_vars = bridge3.init(k4, v_opt_test)
    h_dec_0_test = bridge3.apply(bridge3_vars, v_opt_test)

    print(f"  Input:  {H_trans_test.shape}")
    print(f"  → SSM:  {X_ssm_test.shape}")
    print(f"  → RBM:  {v_in_test.shape}")
    print(f"  → Dec:  {h_dec_0_test.shape}")

    # Verify gradient flows through entire chain
    def full_chain_loss(params1, params2, params3, x):
        x1 = bridge1.apply({'params': params1}, x)
        x2 = bridge2.apply({'params': params2}, x1)

        # Pad if necessary
        if x2.shape[-1] != D_rbm:
            padding = D_rbm - x2.shape[-1]
            x2 = jnp.pad(x2, ((0, 0), (0, padding)))

        x3 = bridge3.apply({'params': params3}, x2)
        return jnp.sum(x3 ** 2)

    grads_chain = jax.grad(full_chain_loss, argnums=(0, 1, 2))(
        bridge1_vars['params'],
        bridge2_vars['params'],
        bridge3_vars['params'],
        H_trans_test
    )

    # Check all gradients are finite
    all_finite = all(
        jax.tree_util.tree_reduce(
            lambda acc, x: acc and jnp.isfinite(x).all(),
            g,
            True
        )
        for g in grads_chain
    )

    print(f"  ✓ PASS - Gradients flow through all bridges" if all_finite else "  ✗ FAIL")

    # Test 5: Verify information preservation
    print("\n[Test 5] Information Preservation Analysis")

    # Create inputs with different patterns
    H_pattern1 = jnp.ones((1, L, D_trans))
    H_pattern2 = -jnp.ones((1, L, D_trans))

    out1 = bridge1.apply(bridge1_vars, H_pattern1)
    out2 = bridge1.apply(bridge1_vars, H_pattern2)

    out1_pooled = bridge2.apply(bridge2_vars, out1)
    out2_pooled = bridge2.apply(bridge2_vars, out2)

    # Different inputs should produce different outputs
    diff_ssm = jnp.linalg.norm(out1 - out2)
    diff_pooled = jnp.linalg.norm(out1_pooled - out2_pooled)

    print(f"  Difference after Bridge 1: {diff_ssm:.4f}")
    print(f"  Difference after Bridge 2: {diff_pooled:.4f}")

    info_preserved = (diff_ssm > 0.1) and (diff_pooled > 0.1)
    print(f"  ✓ PASS - Information preserved" if info_preserved else "  ⚠ WARNING")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    verify_bridges()
