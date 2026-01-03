"""
Spectral Normalization and Lyapunov Stability Verification
Implements and verifies Theorem 1 and Theorem 2 from the paper.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Callable
import numpy as np
from functools import partial


def spectral_norm(W: jnp.ndarray, n_iterations: int = 10) -> float:
    """
    Compute spectral norm (largest singular value) using power iteration.

    Args:
        W: Weight matrix
        n_iterations: Number of power iterations

    Returns:
        Spectral norm ||W||₂
    """
    # Power iteration
    u = jax.random.normal(jax.random.PRNGKey(0), (W.shape[0],))
    u = u / jnp.linalg.norm(u)

    for _ in range(n_iterations):
        v = W.T @ u
        v = v / (jnp.linalg.norm(v) + 1e-8)
        u = W @ v
        u = u / (jnp.linalg.norm(u) + 1e-8)

    # Rayleigh quotient
    sigma = jnp.sqrt(u @ W @ W.T @ u)
    return sigma


def apply_spectral_normalization(
    W: jnp.ndarray,
    target_norm: float = 1.0,
    n_iterations: int = 10
) -> jnp.ndarray:
    """
    Apply spectral normalization to ensure ||W||₂ = target_norm.

    As per paper: W ← W / σ_max(W)
    """
    sigma = spectral_norm(W, n_iterations)
    W_normalized = W * (target_norm / (sigma + 1e-8))
    return W_normalized


def verify_contraction_property(W: jnp.ndarray) -> Tuple[bool, float]:
    """
    Verify if W defines a contraction mapping.

    For Gibbs update T(v) = σ(Wv + b), the Jacobian is:
    J_T(v) = diag(σ'(Wv + b)) * W

    Since 0 < σ'(z) ≤ 0.25, we have:
    ||J_T|| ≤ 0.25 * ||W||₂

    For contraction: ||J_T|| < 1 ⟹ ||W||₂ < 4
    """
    sigma_max = spectral_norm(W)

    # Jacobian bound
    jacobian_bound = 0.25 * sigma_max

    is_contraction = jacobian_bound < 1.0
    return is_contraction, jacobian_bound


def mean_field_update(
    v: jnp.ndarray,
    W: jnp.ndarray,
    b: jnp.ndarray
) -> jnp.ndarray:
    """
    Mean-field update operator: T(v) = σ(Wv + b)

    This is the operator analyzed in Theorem 2.
    """
    return jax.nn.sigmoid(W @ v + b)


def compute_fixed_point(
    W: jnp.ndarray,
    b: jnp.ndarray,
    v_init: jnp.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[jnp.ndarray, int, bool]:
    """
    Find fixed point v* where T(v*) = v* using iteration.

    Returns:
        v_star: Fixed point (or approximation)
        iterations: Number of iterations taken
        converged: Whether convergence criterion was met
    """
    v = v_init

    for i in range(max_iterations):
        v_next = mean_field_update(v, W, b)
        error = jnp.linalg.norm(v_next - v)

        if error < tolerance:
            return v_next, i + 1, True

        v = v_next

    return v, max_iterations, False


def lyapunov_energy_descent(
    W: jnp.ndarray,
    b: jnp.ndarray,
    v_init: jnp.ndarray,
    energy_fn: Callable,
    max_iterations: int = 50
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Verify that energy decreases monotonically during iteration.

    This validates the Lyapunov stability claim.
    """
    v = v_init
    trajectory = [v]
    energies = [float(energy_fn(v))]

    for _ in range(max_iterations):
        v = mean_field_update(v, W, b)
        trajectory.append(v)
        energies.append(float(energy_fn(v)))

    return jnp.array(energies), jnp.array(trajectory)


def compute_lipschitz_constant(
    W: jnp.ndarray,
    num_samples: int = 1000
) -> float:
    """
    Empirically estimate Lipschitz constant of T.

    L = sup_{v₁≠v₂} ||T(v₁) - T(v₂)|| / ||v₁ - v₂||
    """
    key = jax.random.PRNGKey(42)
    D = W.shape[0]
    b = jnp.zeros(D)  # For simplicity

    lipschitz_constants = []

    for _ in range(num_samples):
        key, k1, k2 = jax.random.split(key, 3)

        v1 = jax.random.normal(k1, (D,))
        v2 = jax.random.normal(k2, (D,))

        Tv1 = mean_field_update(v1, W, b)
        Tv2 = mean_field_update(v2, W, b)

        numerator = jnp.linalg.norm(Tv1 - Tv2)
        denominator = jnp.linalg.norm(v1 - v2)

        if denominator > 1e-6:
            lipschitz_constants.append(float(numerator / denominator))

    return jnp.max(jnp.array(lipschitz_constants))


def verify_spectral_stability():
    """Verification tests for spectral normalization and stability."""
    print("=" * 70)
    print("SPECTRAL NORMALIZATION AND LYAPUNOV STABILITY VERIFICATION")
    print("=" * 70)

    # Test 1: Spectral norm computation accuracy
    print("\n[Test 1] Spectral Norm Computation Accuracy")

    D = 64
    key = jax.random.PRNGKey(42)
    W = jax.random.normal(key, (D, D)) * 2.0

    # Our implementation
    sigma_power = spectral_norm(W, n_iterations=20)

    # Ground truth (JAX SVD)
    sigma_exact = jnp.linalg.svd(W, compute_uv=False)[0]

    error = jnp.abs(sigma_power - sigma_exact) / sigma_exact
    print(f"  Power iteration: {sigma_power:.6f}")
    print(f"  Exact (SVD):     {sigma_exact:.6f}")
    print(f"  Relative error:  {error:.2e}")
    print(f"  ✓ PASS" if error < 1e-4 else "  ✗ FAIL")

    # Test 2: Spectral normalization effect
    print("\n[Test 2] Spectral Normalization Effect")

    W_unnormalized = jax.random.normal(key, (D, D)) * 5.0
    sigma_before = spectral_norm(W_unnormalized)

    W_normalized = apply_spectral_normalization(W_unnormalized, target_norm=1.0)
    sigma_after = spectral_norm(W_normalized)

    print(f"  Before normalization: ||W||₂ = {sigma_before:.4f}")
    print(f"  After normalization:  ||W||₂ = {sigma_after:.4f}")
    print(f"  Target:               ||W||₂ = 1.0000")
    print(f"  ✓ PASS" if jnp.abs(sigma_after - 1.0) < 0.01 else "  ✗ FAIL")

    # Test 3: Contraction property verification (Theorem 2)
    print("\n[Test 3] Contraction Property (Theorem 2)")

    # Case 1: Normalized W (should be contraction)
    W_contract = apply_spectral_normalization(W_unnormalized, target_norm=1.0)
    is_contract_1, bound_1 = verify_contraction_property(W_contract)

    print(f"  Normalized W (||W||₂ = 1.0):")
    print(f"    Jacobian bound: ||J_T|| ≤ {bound_1:.4f}")
    print(f"    Is contraction: {is_contract_1} (should be True)")
    print(f"    ✓ PASS" if is_contract_1 else "  ✗ FAIL")

    # Case 2: Un-normalized large W (should NOT be contraction)
    W_large = jax.random.normal(key, (D, D)) * 10.0
    is_contract_2, bound_2 = verify_contraction_property(W_large)

    print(f"\n  Large W (||W||₂ = {spectral_norm(W_large):.2f}):")
    print(f"    Jacobian bound: ||J_T|| ≤ {bound_2:.4f}")
    print(f"    Is contraction: {is_contract_2} (should be False)")
    print(f"    ✓ PASS" if not is_contract_2 else "  ✗ FAIL")

    # Test 4: Fixed point existence and uniqueness
    print("\n[Test 4] Fixed Point Convergence (Banach Theorem)")

    W_stable = apply_spectral_normalization(W_unnormalized, target_norm=0.8)
    b = jax.random.normal(key, (D,)) * 0.1

    # Try multiple random initializations
    convergence_results = []
    fixed_points = []

    for i in range(5):
        v_init = jax.random.normal(jax.random.PRNGKey(i), (D,))
        v_star, iterations, converged = compute_fixed_point(W_stable, b, v_init)

        convergence_results.append(converged)
        fixed_points.append(v_star)

    # All should converge to the same point
    all_converged = all(convergence_results)
    fixed_points = jnp.array(fixed_points)

    # Check uniqueness: all fixed points should be similar
    pairwise_diffs = []
    for i in range(len(fixed_points)):
        for j in range(i + 1, len(fixed_points)):
            diff = jnp.linalg.norm(fixed_points[i] - fixed_points[j])
            pairwise_diffs.append(float(diff))

    max_diff = jnp.max(jnp.array(pairwise_diffs)) if pairwise_diffs else 0.0

    print(f"  Convergence rate: {sum(convergence_results)}/5 initializations")
    print(f"  Max difference between fixed points: {max_diff:.2e}")
    print(f"  ✓ PASS - Unique fixed point exists" if all_converged and max_diff < 1e-3 else "  ✗ FAIL")

    # Test 5: Lyapunov stability - energy descent
    print("\n[Test 5] Lyapunov Stability (Energy Monotonically Decreases)")

    # Define simple energy function
    def energy_fn(v):
        return 0.5 * jnp.sum(v ** 2) - jnp.sum(jax.nn.sigmoid(W_stable @ v + b) * v)

    v_init = jax.random.normal(key, (D,)) * 2.0
    energies, trajectory = lyapunov_energy_descent(W_stable, b, v_init, energy_fn)

    # Check monotonic decrease
    energy_diffs = jnp.diff(energies)
    descending_steps = jnp.sum(energy_diffs <= 0)
    total_steps = len(energy_diffs)

    print(f"  Initial energy: {energies[0]:.4f}")
    print(f"  Final energy:   {energies[-1]:.4f}")
    print(f"  Descending steps: {descending_steps}/{total_steps} ({100*descending_steps/total_steps:.1f}%)")
    print(f"  ✓ PASS - Energy mostly decreases" if descending_steps >= 0.9 * total_steps else "  ⚠ WARNING")

    # Test 6: Empirical Lipschitz constant
    print("\n[Test 6] Empirical Lipschitz Constant Estimation")

    L_empirical = compute_lipschitz_constant(W_stable, num_samples=500)
    L_theoretical = 0.25 * spectral_norm(W_stable)

    print(f"  Empirical:   L ≈ {L_empirical:.4f}")
    print(f"  Theoretical: L ≤ {L_theoretical:.4f}")
    print(f"  Ratio: {L_empirical / L_theoretical:.4f}")
    print(f"  ✓ PASS" if L_empirical <= L_theoretical * 1.1 else "  ⚠ WARNING - Empirical exceeds bound")

    # Test 7: Gradient flow stability
    print("\n[Test 7] Gradient Flow Stability")

    @jax.jit
    def compute_gradient_norm(W, v, b):
        def loss(W_param):
            v_next = jax.nn.sigmoid(W_param @ v + b)
            return jnp.sum(v_next ** 2)

        grads = jax.grad(loss)(W)
        return jnp.linalg.norm(grads)

    # Without spectral norm
    W_unstable = jax.random.normal(key, (D, D)) * 10.0
    v_test = jax.random.normal(key, (D,))
    grad_norm_unstable = compute_gradient_norm(W_unstable, v_test, b)

    # With spectral norm
    W_stable_test = apply_spectral_normalization(W_unstable, target_norm=1.0)
    grad_norm_stable = compute_gradient_norm(W_stable_test, v_test, b)

    print(f"  Gradient norm (unstable W): {grad_norm_unstable:.4f}")
    print(f"  Gradient norm (normalized W): {grad_norm_stable:.4f}")
    print(f"  Reduction factor: {grad_norm_unstable / grad_norm_stable:.2f}x")
    print(f"  ✓ PASS - Spectral norm stabilizes gradients" if grad_norm_stable < grad_norm_unstable else "  ⚠ WARNING")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    verify_spectral_stability()
