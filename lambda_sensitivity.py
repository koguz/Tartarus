"""
Lambda Sensitivity Analysis for Behavior Segmentation.

Sweeps lambda from 0.0 to 1.0 and reports:
- Number of clusters (k) at each lambda via eigengap analysis
- Adjusted Rand Index (ARI) between consecutive lambda values
- Cluster sizes at each lambda

This validates that the behavioral clustering results are stable
across a range of lambda values.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from behavior_segmentation import (
    normalize_transition_weights,
    compute_hamming_similarity_matrix,
    create_adjusted_weight_matrix,
    compute_symmetric_similarity,
    compute_laplacian,
    find_optimal_k_eigengap,
    tactic_to_decimal,
)


def run_clustering_for_lambda(nodes, edges, lambda_val):
    """
    Run the full spectral clustering pipeline for a given lambda value.

    Returns:
        (optimal_k, cluster_labels, eigenvalues)
    """
    normalized_weights = normalize_transition_weights(nodes, edges)
    hamming_sim = compute_hamming_similarity_matrix(nodes)
    adjusted_weights = create_adjusted_weight_matrix(
        normalized_weights, hamming_sim, lambda_val
    )
    symmetric_sim = compute_symmetric_similarity(adjusted_weights)
    laplacian = compute_laplacian(symmetric_sim)

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    optimal_k = find_optimal_k_eigengap(eigenvalues)

    feature_matrix = eigenvectors[:, :optimal_k]
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    return optimal_k, cluster_labels, eigenvalues


def run_sensitivity_analysis(json_path, lambda_min=0.0, lambda_max=1.0, step=0.05):
    """Run the full lambda sensitivity sweep."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    edges = data['edges']

    lambda_values = np.arange(lambda_min, lambda_max + step / 2, step)
    lambda_values = np.round(lambda_values, 4)

    results = []
    for lam in lambda_values:
        k, labels, eigenvalues = run_clustering_for_lambda(nodes, edges, lam)
        cluster_sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        results.append({
            'lambda': lam,
            'k': k,
            'labels': labels,
            'cluster_sizes': cluster_sizes,
            'eigenvalues': eigenvalues,
        })
        print(f"  lambda={lam:.2f}  k={k:2d}  sizes={cluster_sizes}")

    # Compute ARI between consecutive lambda values
    ari_values = []
    for i in range(1, len(results)):
        ari = adjusted_rand_score(results[i - 1]['labels'], results[i]['labels'])
        ari_values.append(ari)

    return lambda_values, results, ari_values


def plot_results(lambda_values, results, ari_values, output_prefix='lambda_sensitivity'):
    """Generate the sensitivity analysis plots."""
    ks = [r['k'] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'hspace': 0.35})

    # Plot 1: k vs lambda
    ax1 = axes[0]
    ax1.plot(lambda_values, ks, 'bo-', markersize=6, linewidth=1.5)
    ax1.set_xlabel('λ (lambda)')
    ax1.set_ylabel('Number of Clusters (k)')
    ax1.set_title('Eigengap-Optimal k vs. Lambda')
    ax1.set_xticks(np.arange(0, 1.05, 0.1))
    ax1.set_yticks(range(min(ks), max(ks) + 1))
    ax1.grid(True, alpha=0.3)

    # Plot 2: ARI between consecutive lambda values
    ax2 = axes[1]
    ari_lambdas = lambda_values[1:]
    ax2.plot(ari_lambdas, ari_values, 'rs-', markersize=6, linewidth=1.5)
    ax2.set_xlabel('λ (lambda)')
    ax2.set_ylabel('ARI with Previous λ')
    ax2.set_title('Adjusted Rand Index Between Consecutive Lambda Values')
    ax2.set_xticks(np.arange(0, 1.05, 0.1))
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.savefig(f'{output_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to '{output_prefix}.png'")
    plt.close()


def print_summary(lambda_values, results, ari_values):
    """Print a summary table of the sensitivity analysis."""
    print("\n" + "=" * 70)
    print("LAMBDA SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"{'Lambda':>8}  {'k':>3}  {'ARI':>6}  {'Cluster Sizes'}")
    print("-" * 70)
    for i, r in enumerate(results):
        ari_str = f"{ari_values[i-1]:.3f}" if i > 0 else "  -  "
        sizes_str = ', '.join(str(s) for s in r['cluster_sizes'])
        print(f"  {r['lambda']:5.2f}   {r['k']:3d}   {ari_str}   [{sizes_str}]")

    # Find stable regions (consecutive lambdas with same k)
    print("\n" + "=" * 70)
    print("STABILITY REGIONS (consecutive lambda values with same k)")
    print("=" * 70)
    ks = [r['k'] for r in results]
    i = 0
    while i < len(ks):
        j = i
        while j < len(ks) and ks[j] == ks[i]:
            j += 1
        lam_start = lambda_values[i]
        lam_end = lambda_values[j - 1]
        region_aris = ari_values[i:j - 1] if j - 1 > i else []
        mean_ari = np.mean(region_aris) if region_aris else float('nan')
        print(f"  k={ks[i]:2d}  lambda=[{lam_start:.2f}, {lam_end:.2f}]  "
              f"span={lam_end - lam_start:.2f}  mean ARI={mean_ari:.3f}")
        i = j


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Lambda sensitivity analysis for behavior segmentation'
    )
    parser.add_argument(
        '--input', '-i', type=str,
        default='analysis_tactic_transitions.json',
        help='Input JSON file with tactic transitions'
    )
    parser.add_argument(
        '--min', type=float, default=0.0,
        help='Minimum lambda value (default: 0.0)'
    )
    parser.add_argument(
        '--max', type=float, default=1.0,
        help='Maximum lambda value (default: 1.0)'
    )
    parser.add_argument(
        '--step', type=float, default=0.05,
        help='Lambda step size (default: 0.05)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='lambda_sensitivity',
        help='Output file prefix for plots (default: lambda_sensitivity)'
    )

    args = parser.parse_args()

    print("Running lambda sensitivity analysis...")
    print(f"  Range: [{args.min}, {args.max}], step={args.step}")
    lambda_values, results, ari_values = run_sensitivity_analysis(
        args.input, args.min, args.max, args.step
    )

    print_summary(lambda_values, results, ari_values)
    plot_results(lambda_values, results, ari_values, args.output)
