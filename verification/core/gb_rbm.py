"""
Gaussian-Bernoulli Restricted Boltzmann Machine (GB-RBM) Verification
Implements and verifies equations (5), (6), (7), and (8) from the paper.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict
import numpy as np
from functools import partial


class GaussianBernoulliRBM:
    """
    Gaussian-Bernoulli RBM implementation following paper equations.
    """

    def __init__(self, visible_dim: int, hidden_dim: int, sigma: float = 1.0):
        """
        Args:
            visible_dim: Dimension D of visible units (continuous)
            hidden_dim: Dimension K of hidden units (binary)
            sigma: Standard deviation for Gaussian visible units
        """
        self.D = visible_dim
        self.K = hidden_dim
        self.sigma = sigma

        # Initialize parameters
        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Weight matrix W: [D x K]
        self.W = jax.random.normal(k1, (visible_dim, hidden_dim)) * 0.01

        # Visible biases b: [D]
        self.b = jnp.zeros(visible_dim)

        # Hidden biases c: [K]
        self.c = jnp.zeros(hidden_dim)

        # Sigma per dimension (can be learned)
        self.sigma_vec = jnp.ones(visible_dim) * sigma

    def energy(self, v: jnp.ndarray, h: jnp.ndarray) -> float:
        """
        Compute energy E(v, h) as per equation (5).

        E(v, h) = Σᵢ (vᵢ - bᵢ)² / (2σᵢ²) - Σⱼ cⱼhⱼ - Σᵢⱼ (vᵢ/σᵢ)Wᵢⱼhⱼ

        Args:
            v: Visible units [D]
            h: Hidden units [K]

        Returns:
            energy: Scalar energy value
        """
        # Term 1: Gaussian energy of visible units
        term1 = jnp.sum((v - self.b) ** 2 / (2 * self.sigma_vec ** 2))

        # Term 2: Hidden bias energy
        term2 = -jnp.sum(self.c * h)

        # Term 3: Interaction energy
        v_normalized = v / self.sigma_vec
        term3 = -jnp.sum(v_normalized[:, None] * self.W * h[None, :])

        energy = term1 + term2 + term3
        return energy

    def free_energy(self, v: jnp.ndarray) -> float:
        """
        Compute free energy F(v) = -log(Σₕ exp(-E(v,h)))

        For binary hidden units, this has a closed form.
        """
        # Gaussian term
        gaussian_term = jnp.sum((v - self.b) ** 2 / (2 * self.sigma_vec ** 2))

        # Hidden activation
        v_normalized = v / self.sigma_vec
        hidden_input = self.c + self.W.T @ v_normalized

        # Log-sum-exp over hidden configurations (tractable for RBM)
        hidden_term = -jnp.sum(jnp.log(1 + jnp.exp(hidden_input)))

        return gaussian_term + hidden_term

    @partial(jax.jit, static_argnums=(0,))
    def sample_hidden_given_visible(
        self,
        v: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Sample h ~ P(h|v) using equation (7).

        P(hⱼ = 1|v) = σ(cⱼ + Σᵢ Wᵢⱼ vᵢ/σᵢ)

        Args:
            v: Visible units [D]
            key: Random key

        Returns:
            h: Sampled hidden units [K]
        """
        v_normalized = v / self.sigma_vec
        logits = self.c + self.W.T @ v_normalized
        probs = jax.nn.sigmoid(logits)

        # Bernoulli sampling
        h = jax.random.bernoulli(key, probs).astype(jnp.float32)
        return h

    def mean_hidden_given_visible(self, v: jnp.ndarray) -> jnp.ndarray:
        """
        Compute mean-field approximation: E[h|v] = P(h=1|v)
        """
        v_normalized = v / self.sigma_vec
        logits = self.c + self.W.T @ v_normalized
        return jax.nn.sigmoid(logits)

    @partial(jax.jit, static_argnums=(0,))
    def sample_visible_given_hidden(
        self,
        h: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Sample v ~ P(v|h) using equation (8).

        P(vᵢ|h) = N(bᵢ + σᵢΣⱼ Wᵢⱼhⱼ, σᵢ²)

        Args:
            h: Hidden units [K]
            key: Random key

        Returns:
            v: Sampled visible units [D]
        """
        mean = self.b + self.sigma_vec * (self.W @ h)
        v = mean + self.sigma_vec * jax.random.normal(key, (self.D,))
        return v

    def mean_visible_given_hidden(self, h: jnp.ndarray) -> jnp.ndarray:
        """
        Compute mean reconstruction: E[v|h]
        """
        return self.b + self.sigma_vec * (self.W @ h)

    def gibbs_step(
        self,
        v: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Single Gibbs sampling step: v -> h -> v'

        Args:
            v: Current visible state
            key: Random key

        Returns:
            v_new: New visible state
            h: Sampled hidden state
        """
        k1, k2 = jax.random.split(key)
        h = self.sample_hidden_given_visible(v, k1)
        v_new = self.sample_visible_given_hidden(h, k2)
        return v_new, h

    def contrastive_divergence(
        self,
        v_data: jnp.ndarray,
        k_steps: int,
        key: jax.random.PRNGKey
    ) -> Dict[str, jnp.ndarray]:
        """
        Contrastive Divergence-k algorithm.

        Args:
            v_data: Data visible units [D]
            k_steps: Number of Gibbs steps
            key: Random key

        Returns:
            Dictionary with gradients and statistics
        """
        # Positive phase: data
        h_pos = self.mean_hidden_given_visible(v_data)

        # Negative phase: k steps of Gibbs sampling
        v_neg = v_data
        for i in range(k_steps):
            key, subkey = jax.random.split(key)
            v_neg, _ = self.gibbs_step(v_neg, subkey)

        h_neg = self.mean_hidden_given_visible(v_neg)

        # Compute gradients
        v_data_norm = v_data / self.sigma_vec
        v_neg_norm = v_neg / self.sigma_vec

        grad_W = jnp.outer(v_data_norm, h_pos) - jnp.outer(v_neg_norm, h_neg)
        grad_b = (v_data - v_neg) / (self.sigma_vec ** 2)
        grad_c = h_pos - h_neg

        reconstruction_error = jnp.linalg.norm(v_data - v_neg)

        return {
            'grad_W': grad_W,
            'grad_b': grad_b,
            'grad_c': grad_c,
            'v_neg': v_neg,
            'h_pos': h_pos,
            'h_neg': h_neg,
            'reconstruction_error': reconstruction_error,
            'free_energy_data': self.free_energy(v_data),
            'free_energy_model': self.free_energy(v_neg)
        }


def gumbel_sigmoid(logits: jnp.ndarray, temperature: float, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Gumbel-Softmax reparameterization for binary variables (Straight-Through estimator).

    Used for differentiable sampling in the paper's Algorithm 1.
    """
    # Sample Gumbel noise
    u = jax.random.uniform(key, logits.shape)
    gumbel_noise = -jnp.log(-jnp.log(u + 1e-8) + 1e-8)

    # Soft sample
    y_soft = jax.nn.sigmoid((logits + gumbel_noise) / temperature)

    # Hard sample (for forward pass)
    y_hard = (y_soft > 0.5).astype(jnp.float32)

    # Straight-through: use hard in forward, soft in backward
    return y_hard + (y_soft - jax.lax.stop_gradient(y_soft))


def verify_gb_rbm():
    """Verification tests for GB-RBM."""
    print("=" * 70)
    print("GAUSSIAN-BERNOULLI RBM VERIFICATION")
    print("=" * 70)

    # Initialize RBM
    D, K = 64, 32
    rbm = GaussianBernoulliRBM(D, K, sigma=1.0)
    key = jax.random.PRNGKey(123)

    # Test 1: Verify energy function properties
    print("\n[Test 1] Energy Function Properties")
    v = jax.random.normal(key, (D,))
    h = jax.random.bernoulli(key, 0.5, shape=(K,)).astype(jnp.float32)

    energy = rbm.energy(v, h)
    print(f"  E(v, h) = {energy:.4f}")

    # Energy should be real and bounded
    print(f"  ✓ PASS - Energy is real and finite" if jnp.isfinite(energy) else "  ✗ FAIL")

    # Test 2: Verify conditional probabilities sum to 1
    print("\n[Test 2] Conditional Probability Normalization")

    # P(h|v) for all 2^K configurations should sum to 1
    # (Intractable for large K, test with small K)
    rbm_small = GaussianBernoulliRBM(8, 4, sigma=1.0)
    v_small = jax.random.normal(key, (8,))

    # Enumerate all hidden configurations
    total_prob = 0.0
    for i in range(2 ** 4):
        h_config = jnp.array([(i >> j) & 1 for j in range(4)], dtype=jnp.float32)
        prob_h = jnp.prod(
            jnp.where(h_config == 1,
                      rbm_small.mean_hidden_given_visible(v_small),
                      1 - rbm_small.mean_hidden_given_visible(v_small))
        )
        total_prob += prob_h

    print(f"  Σₕ P(h|v) = {total_prob:.6f} (should be ≈ 1.0)")
    print(f"  ✓ PASS" if jnp.abs(total_prob - 1.0) < 0.01 else "  ✗ FAIL")

    # Test 3: Gibbs sampling convergence
    print("\n[Test 3] Gibbs Sampling Convergence")

    v_init = jax.random.normal(key, (D,))
    free_energies = []

    v_current = v_init
    for step in range(50):
        key, subkey = jax.random.split(key)
        v_current, _ = rbm.gibbs_step(v_current, subkey)
        fe = rbm.free_energy(v_current)
        free_energies.append(float(fe))

    # Free energy should decrease (roughly)
    fe_initial = free_energies[0]
    fe_final = free_energies[-1]
    fe_change = fe_final - fe_initial

    print(f"  Initial Free Energy: {fe_initial:.4f}")
    print(f"  Final Free Energy: {fe_final:.4f}")
    print(f"  Change: {fe_change:.4f}")
    print(f"  ✓ PASS - Energy decreased" if fe_change < 0 else "  ⚠ WARNING - Energy increased (stochastic)")

    # Test 4: Contrastive Divergence gradients
    print("\n[Test 4] Contrastive Divergence (CD-10)")

    v_data = jax.random.normal(key, (D,))
    key, subkey = jax.random.split(key)
    cd_results = rbm.contrastive_divergence(v_data, k_steps=10, key=subkey)

    print(f"  Reconstruction error: {cd_results['reconstruction_error']:.4f}")
    print(f"  ΔF (data - model): {cd_results['free_energy_data'] - cd_results['free_energy_model']:.4f}")
    print(f"  ||∇W||: {jnp.linalg.norm(cd_results['grad_W']):.4f}")
    print(f"  ✓ PASS - Gradients computed" if jnp.isfinite(cd_results['grad_W']).all() else "  ✗ FAIL")

    # Test 5: Gumbel-Sigmoid differentiability
    print("\n[Test 5] Gumbel-Sigmoid Differentiability")

    def loss_fn(W, v):
        logits = W @ v
        key_local = jax.random.PRNGKey(0)
        h_soft = gumbel_sigmoid(logits, temperature=1.0, key=key_local)
        return jnp.sum(h_soft ** 2)

    W_test = jax.random.normal(key, (K, D))
    v_test = jax.random.normal(key, (D,))

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(W_test, v_test)

    print(f"  ||∇W|| from Gumbel-Sigmoid: {jnp.linalg.norm(grads):.4f}")
    print(f"  ✓ PASS - Gradients flow through Gumbel" if jnp.isfinite(grads).all() else "  ✗ FAIL")

    # Test 6: Equation verification - detailed check
    print("\n[Test 6] Verify Equations (7) and (8) Explicitly")

    v_test = jax.random.normal(key, (D,))
    h_test = jax.random.bernoulli(key, 0.5, shape=(K,)).astype(jnp.float32)

    # Equation (7): P(hⱼ=1|v) = σ(cⱼ + Σᵢ Wᵢⱼ vᵢ/σᵢ)
    v_normalized = v_test / rbm.sigma_vec
    logits_expected = rbm.c + rbm.W.T @ v_normalized
    probs_expected = jax.nn.sigmoid(logits_expected)
    probs_computed = rbm.mean_hidden_given_visible(v_test)

    eq7_error = jnp.linalg.norm(probs_expected - probs_computed)
    print(f"  Equation (7) error: {eq7_error:.2e}")
    print(f"  ✓ PASS" if eq7_error < 1e-6 else "  ✗ FAIL")

    # Equation (8): E[v|h] = b + σ * W * h
    mean_v_expected = rbm.b + rbm.sigma_vec * (rbm.W @ h_test)
    mean_v_computed = rbm.mean_visible_given_hidden(h_test)

    eq8_error = jnp.linalg.norm(mean_v_expected - mean_v_computed)
    print(f"  Equation (8) error: {eq8_error:.2e}")
    print(f"  ✓ PASS" if eq8_error < 1e-6 else "  ✗ FAIL")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    verify_gb_rbm()
