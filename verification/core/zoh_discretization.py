"""
Zero-Order Hold (ZOH) Discretization Verification
Implements and verifies equations (3) and (4) from the paper.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from typing import Tuple
import numpy as np


def zoh_discretization_analytical(
    A: jnp.ndarray,
    B: jnp.ndarray,
    delta: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Analytically compute ZOH discretization as per equations (3) and (4).

    Args:
        A: Continuous state matrix (N x N)
        B: Continuous input matrix (N x M)
        delta: Discretization step size

    Returns:
        A_discrete: exp(delta * A)
        B_discrete: (delta*A)^{-1} * (exp(delta*A) - I) * delta*B
    """
    N = A.shape[0]
    I = jnp.eye(N)

    # Equation (3): A_discrete = exp(Δ*A)
    delta_A = delta * A
    A_discrete = expm(delta_A)

    # Equation (4): B_discrete = (ΔA)^{-1} * (exp(ΔA) - I) * ΔB
    # For diagonal A (as stated in paper), this simplifies
    if jnp.allclose(A, jnp.diag(jnp.diag(A))):
        # Diagonal case: closed form
        eigenvalues = jnp.diag(A)
        # exp(lambda * delta) - 1 / lambda
        # Handle near-zero eigenvalues carefully
        def safe_divide(lam):
            return jnp.where(
                jnp.abs(lam) > 1e-8,
                (jnp.exp(delta * lam) - 1) / lam,
                delta  # L'Hopital limit as lambda -> 0
            )
        scaling = jax.vmap(safe_divide)(eigenvalues)
        B_discrete = jnp.diag(scaling) @ B
    else:
        # General case: use matrix inverse
        delta_A_inv = jnp.linalg.pinv(delta_A)
        B_discrete = delta_A_inv @ (A_discrete - I) @ (delta * B)

    return A_discrete, B_discrete


def zoh_discretization_series(
    A: jnp.ndarray,
    B: jnp.ndarray,
    delta: float,
    n_terms: int = 10
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute ZOH using Taylor series approximation (eq 3).

    A_discrete ≈ sum_{k=0}^N (ΔA)^k / k!
    """
    N = A.shape[0]
    I = jnp.eye(N)
    delta_A = delta * A

    # Series approximation
    A_discrete = I.copy()
    term = I.copy()

    for k in range(1, n_terms + 1):
        term = term @ delta_A / k
        A_discrete = A_discrete + term

    # B using same approach
    if jnp.allclose(A, jnp.diag(jnp.diag(A))):
        eigenvalues = jnp.diag(A)
        def safe_divide(lam):
            return jnp.where(
                jnp.abs(lam) > 1e-8,
                (jnp.exp(delta * lam) - 1) / lam,
                delta
            )
        scaling = jax.vmap(safe_divide)(eigenvalues)
        B_discrete = jnp.diag(scaling) @ B
    else:
        delta_A_inv = jnp.linalg.pinv(delta_A)
        B_discrete = delta_A_inv @ (A_discrete - I) @ (delta * B)

    return A_discrete, B_discrete


def parallel_scan_operation(
    A_discrete: jnp.ndarray,
    B_discrete: jnp.ndarray,
    inputs: jnp.ndarray
) -> jnp.ndarray:
    """
    Verify O(log L) parallel scan claim.

    Args:
        A_discrete: Discretized state matrix (N x N)
        B_discrete: Discretized input matrix (N x M)
        inputs: Input sequence (L x M)

    Returns:
        states: Hidden states (L x N)
    """
    L = inputs.shape[0]
    N = A_discrete.shape[0]

    # Define associative operator for parallel scan
    def operator(elem1, elem2):
        A1, Bu1 = elem1
        A2, Bu2 = elem2
        # Composition: (A2*A1, A2*Bu1 + Bu2)
        return (A2 @ A1, A2 @ Bu1 + Bu2)

    # Prepare elements: (A^i, A^{i-1}*B*u_i + ... + B*u_1)
    # Start with (A, B*u) for each timestep
    Bu = jax.vmap(lambda u: B_discrete @ u)(inputs)

    # Initialize with identity for composition
    I = jnp.eye(N)
    elements = [(A_discrete, bu) for bu in Bu]

    # JAX parallel scan (implemented as tree reduction - O(log L))
    result = jax.lax.associative_scan(operator, elements)

    # Extract states
    states = jnp.array([bu for _, bu in result])

    return states


def sequential_ssm(
    A_discrete: jnp.ndarray,
    B_discrete: jnp.ndarray,
    inputs: jnp.ndarray
) -> jnp.ndarray:
    """
    Sequential O(L) implementation for comparison.
    """
    L = inputs.shape[0]
    N = A_discrete.shape[0]

    states = []
    h = jnp.zeros(N)

    for t in range(L):
        h = A_discrete @ h + B_discrete @ inputs[t]
        states.append(h)

    return jnp.array(states)


def verify_discretization():
    """Verification tests for ZOH discretization."""
    print("=" * 70)
    print("ZOH DISCRETIZATION VERIFICATION")
    print("=" * 70)

    # Test 1: Compare analytical vs series approximation
    print("\n[Test 1] Analytical vs Series Approximation")
    N, M = 64, 32
    key = jax.random.PRNGKey(42)

    # Diagonal A for efficiency (as paper states)
    eigenvalues = jax.random.uniform(key, (N,), minval=-0.5, maxval=0)
    A = jnp.diag(eigenvalues)
    B = jax.random.normal(key, (N, M)) * 0.1
    delta = 0.01

    A_disc_analytical, B_disc_analytical = zoh_discretization_analytical(A, B, delta)
    A_disc_series, B_disc_series = zoh_discretization_series(A, B, delta, n_terms=20)

    error_A = jnp.linalg.norm(A_disc_analytical - A_disc_series)
    error_B = jnp.linalg.norm(B_disc_analytical - B_disc_series)

    print(f"  A matrix error: {error_A:.2e} (should be < 1e-6)")
    print(f"  B matrix error: {error_B:.2e} (should be < 1e-6)")
    print(f"  ✓ PASS" if error_A < 1e-6 and error_B < 1e-6 else "  ✗ FAIL")

    # Test 2: Verify stability (eigenvalues of A_discrete should be < 1)
    print("\n[Test 2] Stability of Discretized System")
    eigenvalues_discrete = jnp.linalg.eigvals(A_disc_analytical)
    max_eigenvalue = jnp.max(jnp.abs(eigenvalues_discrete))

    print(f"  Max |eigenvalue| of A_discrete: {max_eigenvalue:.4f}")
    print(f"  Theoretical bound: exp(delta * max(Re(λ))) = {jnp.exp(delta * jnp.max(eigenvalues)):.4f}")
    print(f"  ✓ PASS - System is stable" if max_eigenvalue < 1 else "  ✗ FAIL - System unstable")

    # Test 3: Compare parallel vs sequential execution
    print("\n[Test 3] Parallel Scan vs Sequential (Correctness)")
    L = 128
    inputs = jax.random.normal(key, (L, M))

    states_parallel = parallel_scan_operation(A_disc_analytical, B_disc_analytical, inputs)
    states_sequential = sequential_ssm(A_disc_analytical, B_disc_analytical, inputs)

    scan_error = jnp.linalg.norm(states_parallel - states_sequential)
    print(f"  Reconstruction error: {scan_error:.2e}")
    print(f"  ✓ PASS" if scan_error < 1e-4 else "  ✗ FAIL")

    # Test 4: Complexity analysis (timing)
    print("\n[Test 4] Complexity Analysis (O(L) vs O(log L))")

    def time_operation(func, *args):
        # JIT compile
        jitted_func = jax.jit(func)
        # Warmup
        _ = jitted_func(*args)
        # Time
        start = jax.lib.xla_bridge.get_backend().synchronize()
        import time
        start = time.time()
        _ = jitted_func(*args).block_until_ready()
        return time.time() - start

    sequence_lengths = [64, 128, 256, 512, 1024]
    print(f"  {'L':>6} {'Sequential (ms)':>18} {'Parallel (ms)':>16} {'Speedup':>10}")
    print("  " + "-" * 60)

    for L in sequence_lengths:
        inputs_test = jax.random.normal(key, (L, M))

        time_seq = time_operation(sequential_ssm, A_disc_analytical, B_disc_analytical, inputs_test) * 1000
        time_par = time_operation(parallel_scan_operation, A_disc_analytical, B_disc_analytical, inputs_test) * 1000

        speedup = time_seq / time_par if time_par > 0 else 0
        print(f"  {L:6d} {time_seq:18.4f} {time_par:16.4f} {speedup:10.2f}x")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    verify_discretization()
