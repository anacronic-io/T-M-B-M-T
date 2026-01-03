"""
Complexity Analysis for T-M-B-M-T Architecture
Verifies theoretical complexity claims from the paper.
"""

import jax
import jax.numpy as jnp
import time
import numpy as np
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def transformer_attention_complexity(seq_len: int, d_model: int, batch_size: int) -> Dict[str, float]:
    """
    Measure Transformer self-attention complexity: O(L²d)

    Args:
        seq_len: Sequence length L
        d_model: Model dimension d
        batch_size: Batch size B

    Returns:
        Dictionary with time, memory, and theoretical complexity
    """
    key = jax.random.PRNGKey(0)

    # Simulate attention: Q @ K.T @ V
    Q = jax.random.normal(key, (batch_size, seq_len, d_model))
    K = jax.random.normal(key, (batch_size, seq_len, d_model))
    V = jax.random.normal(key, (batch_size, seq_len, d_model))

    @jax.jit
    def attention(q, k, v):
        # QK^T: [B, L, d] @ [B, d, L] → [B, L, L]
        scores = jnp.einsum('bld,bkd->blk', q, k) / jnp.sqrt(d_model)
        weights = jax.nn.softmax(scores, axis=-1)
        # Attention @ V: [B, L, L] @ [B, L, d] → [B, L, d]
        output = jnp.einsum('blk,bkd->bld', weights, v)
        return output

    # Warmup
    _ = attention(Q, K, V).block_until_ready()

    # Time measurement
    start = time.time()
    output = attention(Q, K, V).block_until_ready()
    elapsed = time.time() - start

    # Memory: Store attention matrix [B, L, L]
    memory_gb = (batch_size * seq_len * seq_len * 4) / 1e9  # float32

    # Theoretical FLOPs: 2 * B * L^2 * d
    flops = 2 * batch_size * (seq_len ** 2) * d_model

    return {
        'time_ms': elapsed * 1000,
        'memory_gb': memory_gb,
        'flops': flops,
        'complexity_class': 'O(L²d)'
    }


def ssm_linear_complexity(seq_len: int, d_model: int, batch_size: int) -> Dict[str, float]:
    """
    Measure SSM (Mamba) complexity: O(Ld)

    Args:
        seq_len: Sequence length L
        d_model: Model dimension d
        batch_size: Batch size B

    Returns:
        Dictionary with time, memory, and theoretical complexity
    """
    key = jax.random.PRNGKey(0)

    # SSM parameters
    X = jax.random.normal(key, (batch_size, seq_len, d_model))
    A = jax.random.normal(key, (d_model, d_model)) * 0.1
    B = jax.random.normal(key, (d_model, d_model)) * 0.1

    @jax.jit
    def ssm_scan(x, A, B):
        """Sequential scan: h_t = A @ h_{t-1} + B @ x_t"""
        def step(h, x_t):
            h_new = A @ h + B @ x_t
            return h_new, h_new

        h_init = jnp.zeros(d_model)

        # vmap over batch
        def batch_scan(x_batch):
            _, h_sequence = jax.lax.scan(step, h_init, x_batch)
            return h_sequence

        return jax.vmap(batch_scan)(x)

    # Warmup
    _ = ssm_scan(X, A, B).block_until_ready()

    # Time measurement
    start = time.time()
    output = ssm_scan(X, A, B).block_until_ready()
    elapsed = time.time() - start

    # Memory: Store hidden states [B, L, d]
    memory_gb = (batch_size * seq_len * d_model * 4) / 1e9  # float32

    # Theoretical FLOPs: 2 * B * L * d^2
    flops = 2 * batch_size * seq_len * (d_model ** 2)

    return {
        'time_ms': elapsed * 1000,
        'memory_gb': memory_gb,
        'flops': flops,
        'complexity_class': 'O(Ld²)'
    }


def parallel_scan_complexity(seq_len: int, d_model: int, batch_size: int) -> Dict[str, float]:
    """
    Measure parallel scan complexity: O(log L) parallel time

    Args:
        seq_len: Sequence length L
        d_model: Model dimension d
        batch_size: Batch size B

    Returns:
        Dictionary with time, memory, and theoretical complexity
    """
    key = jax.random.PRNGKey(0)

    X = jax.random.normal(key, (batch_size, seq_len, d_model))
    A = jax.random.normal(key, (d_model, d_model)) * 0.1
    B = jax.random.normal(key, (d_model, d_model)) * 0.1

    @jax.jit
    def parallel_scan_ssm(x, A, B):
        """Parallel associative scan"""
        def operator(elem1, elem2):
            A1, h1 = elem1
            A2, h2 = elem2
            # Composition
            return (A2 @ A1, A2 @ h1 + h2)

        def process_sequence(x_seq):
            # Prepare elements
            Bu = jax.vmap(lambda x_t: B @ x_t)(x_seq)
            elements = [(A, bu) for bu in Bu]

            # Parallel scan
            result = jax.lax.associative_scan(operator, elements)
            return jnp.array([h for _, h in result])

        return jax.vmap(process_sequence)(x)

    # Warmup
    _ = parallel_scan_ssm(X, A, B).block_until_ready()

    # Time measurement
    start = time.time()
    output = parallel_scan_ssm(X, A, B).block_until_ready()
    elapsed = time.time() - start

    # Memory: Same as sequential
    memory_gb = (batch_size * seq_len * d_model * 4) / 1e9

    # Theoretical parallel steps: log2(L)
    parallel_steps = int(np.ceil(np.log2(seq_len)))

    return {
        'time_ms': elapsed * 1000,
        'memory_gb': memory_gb,
        'parallel_steps': parallel_steps,
        'complexity_class': 'O(log L) parallel'
    }


def rbm_complexity(visible_dim: int, hidden_dim: int, batch_size: int, gibbs_steps: int) -> Dict[str, float]:
    """
    Measure RBM Gibbs sampling complexity: O(DK) per step

    Args:
        visible_dim: Visible dimension D
        hidden_dim: Hidden dimension K
        batch_size: Batch size B
        gibbs_steps: Number of Gibbs iterations

    Returns:
        Dictionary with time, memory, and theoretical complexity
    """
    key = jax.random.PRNGKey(0)

    V = jax.random.normal(key, (batch_size, visible_dim))
    W = jax.random.normal(key, (visible_dim, hidden_dim)) * 0.01
    b = jnp.zeros(visible_dim)
    c = jnp.zeros(hidden_dim)

    @jax.jit
    def gibbs_sampling(v, W, b, c, key, steps):
        def gibbs_step(carry, _):
            v_current, key_current = carry
            key_current, k1, k2 = jax.random.split(key_current, 3)

            # Sample h | v
            logits_h = c + W.T @ v_current
            h = jax.random.bernoulli(k1, jax.nn.sigmoid(logits_h)).astype(jnp.float32)

            # Sample v | h
            v_new = b + W @ h

            return (v_new, key_current), v_new

        (v_final, _), _ = jax.lax.scan(gibbs_step, (v, key), jnp.arange(steps))
        return v_final

    # Warmup
    _ = jax.vmap(lambda v, k: gibbs_sampling(v, W, b, c, k, gibbs_steps))(V, jax.random.split(key, batch_size)).block_until_ready()

    # Time measurement
    start = time.time()
    output = jax.vmap(lambda v, k: gibbs_sampling(v, W, b, c, k, gibbs_steps))(V, jax.random.split(key, batch_size)).block_until_ready()
    elapsed = time.time() - start

    # Memory: Store V, H, W
    memory_gb = (batch_size * visible_dim + batch_size * hidden_dim + visible_dim * hidden_dim) * 4 / 1e9

    # Theoretical FLOPs per Gibbs step: 2 * D * K
    flops_per_step = 2 * visible_dim * hidden_dim
    total_flops = batch_size * gibbs_steps * flops_per_step

    return {
        'time_ms': elapsed * 1000,
        'memory_gb': memory_gb,
        'flops': total_flops,
        'complexity_class': f'O(DK) × {gibbs_steps} steps'
    }


def run_scalability_analysis():
    """Run comprehensive scalability analysis."""
    print("=" * 70)
    print("COMPLEXITY AND SCALABILITY ANALYSIS")
    print("=" * 70)

    # Test 1: Transformer vs SSM scaling with sequence length
    print("\n[Test 1] Sequence Length Scaling: Transformer vs SSM")

    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    d_model = 256
    batch_size = 4

    results_transformer = []
    results_ssm_sequential = []
    results_ssm_parallel = []

    print(f"\n  {'L':>6} {'Trans (ms)':>12} {'SSM-Seq (ms)':>14} {'SSM-Par (ms)':>14} {'Speedup':>10}")
    print("  " + "-" * 70)

    for L in seq_lengths:
        # Transformer
        try:
            res_trans = transformer_attention_complexity(L, d_model, batch_size)
            results_transformer.append(res_trans)
        except Exception as e:
            print(f"  Transformer failed at L={L}: {e}")
            results_transformer.append({'time_ms': float('inf')})

        # SSM Sequential
        res_ssm_seq = ssm_linear_complexity(L, d_model, batch_size)
        results_ssm_sequential.append(res_ssm_seq)

        # SSM Parallel
        res_ssm_par = parallel_scan_complexity(L, d_model, batch_size)
        results_ssm_parallel.append(res_ssm_par)

        speedup = res_trans['time_ms'] / res_ssm_par['time_ms'] if res_ssm_par['time_ms'] > 0 else 0

        print(f"  {L:6d} {res_trans['time_ms']:12.2f} {res_ssm_seq['time_ms']:14.2f} {res_ssm_par['time_ms']:14.2f} {speedup:10.2f}x")

    # Test 2: Memory scaling
    print("\n[Test 2] Memory Footprint Scaling")

    print(f"\n  {'L':>6} {'Trans (GB)':>12} {'SSM (GB)':>12} {'Ratio':>10}")
    print("  " + "-" * 45)

    for i, L in enumerate(seq_lengths):
        mem_trans = results_transformer[i].get('memory_gb', float('inf'))
        mem_ssm = results_ssm_sequential[i]['memory_gb']
        ratio = mem_trans / mem_ssm if mem_ssm > 0 else 0

        print(f"  {L:6d} {mem_trans:12.6f} {mem_ssm:12.6f} {ratio:10.2f}x")

    # Test 3: Verify O(L²) vs O(L) growth
    print("\n[Test 3] Empirical Complexity Verification")

    # Fit polynomial to log-log plot
    log_L = np.log(seq_lengths[:len(results_transformer)])

    # Transformer times
    trans_times = [r['time_ms'] for r in results_transformer if r['time_ms'] != float('inf')]
    if len(trans_times) >= 3:
        log_times_trans = np.log(trans_times[:len(log_L)])
        poly_trans = np.polyfit(log_L[:len(log_times_trans)], log_times_trans, 1)
        exponent_trans = poly_trans[0]
        print(f"  Transformer: T ∝ L^{exponent_trans:.2f} (expected ~2.0)")

    # SSM times
    ssm_times = [r['time_ms'] for r in results_ssm_sequential]
    log_times_ssm = np.log(ssm_times)
    poly_ssm = np.polyfit(log_L, log_times_ssm, 1)
    exponent_ssm = poly_ssm[0]
    print(f"  SSM:         T ∝ L^{exponent_ssm:.2f} (expected ~1.0)")

    print(f"\n  ✓ PASS" if abs(exponent_ssm - 1.0) < 0.3 else "  ⚠ WARNING")

    # Test 4: RBM Gibbs sampling
    print("\n[Test 4] RBM Gibbs Sampling Complexity")

    visible_dims = [64, 128, 256, 512]
    hidden_dim = 128
    batch_size = 4
    gibbs_steps = 10

    print(f"\n  {'D':>6} {'Time (ms)':>12} {'Memory (GB)':>14} {'FLOPs':>14}")
    print("  " + "-" * 50)

    for D in visible_dims:
        res_rbm = rbm_complexity(D, hidden_dim, batch_size, gibbs_steps)
        print(f"  {D:6d} {res_rbm['time_ms']:12.2f} {res_rbm['memory_gb']:14.6f} {res_rbm['flops']:14.2e}")

    # Test 5: End-to-end pipeline complexity
    print("\n[Test 5] Full T-M-B-M-T Pipeline Complexity")

    L = 512
    D_trans = 512
    D_ssm = 256
    D_rbm = 128
    batch_size = 4
    gibbs_steps = 10

    # Component complexities
    time_trans = transformer_attention_complexity(L, D_trans, batch_size)['time_ms']
    time_ssm = parallel_scan_complexity(L, D_ssm, batch_size)['time_ms']
    time_rbm = rbm_complexity(D_rbm, D_rbm // 2, batch_size, gibbs_steps)['time_ms']

    total_time = time_trans * 2 + time_ssm * 2 + time_rbm  # Encoder + Decoder

    print(f"\n  Component Breakdown (L={L}):")
    print(f"    Transformer Encoder:  {time_trans:8.2f} ms")
    print(f"    Mamba Encoder:        {time_ssm:8.2f} ms")
    print(f"    RBM Core:             {time_rbm:8.2f} ms")
    print(f"    Mamba Decoder:        {time_ssm:8.2f} ms")
    print(f"    Transformer Decoder:  {time_trans:8.2f} ms")
    print(f"    " + "-" * 40)
    print(f"    Total Pipeline:       {total_time:8.2f} ms")

    # Compare to pure Transformer
    pure_transformer_time = transformer_attention_complexity(L, D_trans, batch_size)['time_ms'] * 4  # Enc + Dec layers
    speedup = pure_transformer_time / total_time

    print(f"\n  Pure Transformer (4 layers): {pure_transformer_time:8.2f} ms")
    print(f"  T-M-B-M-T Pipeline:          {total_time:8.2f} ms")
    print(f"  Theoretical Speedup:         {speedup:8.2f}x")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Generate plots
    generate_plots(seq_lengths, results_transformer, results_ssm_sequential, results_ssm_parallel)


def generate_plots(seq_lengths, results_trans, results_ssm_seq, results_ssm_par):
    """Generate visualization plots."""
    print("\n[Generating Plots]")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Time vs Sequence Length (log-log)
    ax1 = axes[0]

    trans_times = [r['time_ms'] for r in results_trans if r['time_ms'] != float('inf')]
    ssm_seq_times = [r['time_ms'] for r in results_ssm_seq]
    ssm_par_times = [r['time_ms'] for r in results_ssm_par]

    L_trans = seq_lengths[:len(trans_times)]
    L_ssm = seq_lengths[:len(ssm_seq_times)]

    ax1.loglog(L_trans, trans_times, 'o-', label='Transformer O(L²)', linewidth=2)
    ax1.loglog(L_ssm, ssm_seq_times, 's-', label='SSM Sequential O(L)', linewidth=2)
    ax1.loglog(L_ssm, ssm_par_times, '^-', label='SSM Parallel O(log L)', linewidth=2)

    ax1.set_xlabel('Sequence Length L', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Computational Complexity Scaling', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Memory vs Sequence Length
    ax2 = axes[1]

    trans_mem = [r.get('memory_gb', 0) for r in results_trans if r['time_ms'] != float('inf')]
    ssm_mem = [r['memory_gb'] for r in results_ssm_seq]

    ax2.semilogy(L_trans, trans_mem, 'o-', label='Transformer O(L²)', linewidth=2)
    ax2.semilogy(L_ssm, ssm_mem, 's-', label='SSM O(L)', linewidth=2)

    ax2.set_xlabel('Sequence Length L', fontsize=12)
    ax2.set_ylabel('Memory (GB)', fontsize=12)
    ax2.set_title('Memory Footprint Scaling', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/T-M-B-M-T/verification/analysis/complexity_plots.png', dpi=150)
    print(f"  Saved: verification/analysis/complexity_plots.png")


if __name__ == "__main__":
    run_scalability_analysis()
