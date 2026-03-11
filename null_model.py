"""
Null Model for State-Behavior Cluster Alignment.

Tests whether the observed ARI/NMI/AMI between tactic clusters and
Infomap state clusters is distinguishable from what a random partition
of states (with the same cluster sizes) would produce.

Procedure:
1. Load the real Infomap state clusters and tactic clusters.
2. Compute the observed ARI, NMI, AMI using the same methodology as state_clustering.py.
3. Generate N random partitions of 128 states into clusters matching
   the Infomap size distribution (96, 14, 6, 4, 5, 3).
4. For each random partition, recompute ARI/NMI/AMI.
5. Report the distribution and where the observed values fall (p-values).
"""

import argparse
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt


def load_state_clusters(filepath):
    """Load Infomap state clusters from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    clusters = data['infomap']['clusters']
    return {int(k): v for k, v in clusters.items()}


def load_tactic_clusters(filepath):
    """Load tactic clusters."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['by_tactic_string']


def load_tactic_state_mapping(filepath):
    """Load tactic -> states mapping."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_alignment(state_clusters, tactic_clusters, tactic_state_mapping):
    """
    Compute ARI, NMI, AMI between tactic clusters and state-derived labels.

    For each tactic, find its dominant state cluster (weighted by usage count),
    then compare tactic cluster labels vs state-derived labels.
    """
    tactics_with_data = [t for t in tactic_clusters.keys() if t in tactic_state_mapping]

    tactic_cluster_labels = []
    state_derived_labels = []

    for tactic in sorted(tactics_with_data):
        state_counts = tactic_state_mapping[tactic]['state_counts']

        # Aggregate counts by state cluster
        cluster_weights = {}
        for state_str, count in state_counts.items():
            state = int(state_str)
            if state in state_clusters:
                sc = state_clusters[state]
                cluster_weights[sc] = cluster_weights.get(sc, 0) + count

        if cluster_weights:
            dominant = max(cluster_weights, key=cluster_weights.get)
            tactic_cluster_labels.append(tactic_clusters[tactic])
            state_derived_labels.append(dominant)

    ari = adjusted_rand_score(tactic_cluster_labels, state_derived_labels)
    nmi = normalized_mutual_info_score(tactic_cluster_labels, state_derived_labels)
    ami = adjusted_mutual_info_score(tactic_cluster_labels, state_derived_labels)

    return ari, nmi, ami


def get_cluster_sizes(state_clusters):
    """Get the size of each cluster, ordered by cluster ID."""
    sizes = {}
    for state, cluster in state_clusters.items():
        sizes[cluster] = sizes.get(cluster, 0) + 1
    return sizes


def generate_random_partition(states, cluster_sizes, rng):
    """
    Generate a random partition of states matching the given cluster sizes.

    Returns a dict mapping state -> cluster_id.
    """
    shuffled = states.copy()
    rng.shuffle(shuffled)

    result = {}
    offset = 0
    for cluster_id, size in cluster_sizes.items():
        for state in shuffled[offset:offset + size]:
            result[state] = cluster_id
        offset += size

    return result


def run_null_model(state_clusters, tactic_clusters, tactic_state_mapping,
                   n_permutations=10000, seed=42):
    """Run the null model analysis."""
    # Compute observed values
    obs_ari, obs_nmi, obs_ami = compute_alignment(
        state_clusters, tactic_clusters, tactic_state_mapping
    )

    print(f"Observed values:")
    print(f"  ARI = {obs_ari:.4f}")
    print(f"  NMI = {obs_nmi:.4f}")
    print(f"  AMI = {obs_ami:.4f}")

    # Get cluster sizes and states
    cluster_sizes = get_cluster_sizes(state_clusters)
    states = sorted(state_clusters.keys())

    print(f"\nCluster sizes: {dict(sorted(cluster_sizes.items()))}")
    print(f"Total states: {len(states)}")
    print(f"\nRunning {n_permutations} random permutations...")

    rng = np.random.RandomState(seed)
    null_ari = np.zeros(n_permutations)
    null_nmi = np.zeros(n_permutations)
    null_ami = np.zeros(n_permutations)

    for i in range(n_permutations):
        random_clusters = generate_random_partition(states, cluster_sizes, rng)
        ari, nmi, ami = compute_alignment(
            random_clusters, tactic_clusters, tactic_state_mapping
        )
        null_ari[i] = ari
        null_nmi[i] = nmi
        null_ami[i] = ami

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_permutations}...")

    # Compute p-values (two-sided for ARI/AMI, one-sided for NMI)
    # For ARI/AMI: fraction of null values with |value| >= |observed|
    p_ari = np.mean(np.abs(null_ari) >= np.abs(obs_ari))
    p_ami = np.mean(np.abs(null_ami) >= np.abs(obs_ami))
    # For NMI: fraction of null values >= observed (NMI is always >= 0)
    p_nmi = np.mean(null_nmi >= obs_nmi)

    return {
        'observed': {'ari': obs_ari, 'nmi': obs_nmi, 'ami': obs_ami},
        'null_ari': null_ari, 'null_nmi': null_nmi, 'null_ami': null_ami,
        'p_values': {'ari': p_ari, 'nmi': p_nmi, 'ami': p_ami},
    }


def print_results(results):
    """Print formatted results."""
    obs = results['observed']
    pvals = results['p_values']

    print(f"\n{'='*60}")
    print("NULL MODEL RESULTS")
    print("="*60)

    for metric in ['ari', 'nmi', 'ami']:
        null_vals = results[f'null_{metric}']
        print(f"\n  {metric.upper()}:")
        print(f"    Observed:       {obs[metric]:+.4f}")
        print(f"    Null mean:      {np.mean(null_vals):+.4f}")
        print(f"    Null std:       {np.std(null_vals):.4f}")
        print(f"    Null [5%, 95%]: [{np.percentile(null_vals, 5):+.4f}, {np.percentile(null_vals, 95):+.4f}]")
        print(f"    p-value:        {pvals[metric]:.4f}")

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print("="*60)

    for metric in ['ari', 'nmi', 'ami']:
        p = pvals[metric]
        if p > 0.05:
            print(f"  {metric.upper()}: Observed value is NOT significantly different from "
                  f"random (p={p:.4f})")
        else:
            print(f"  {metric.upper()}: Observed value IS significantly different from "
                  f"random (p={p:.4f})")


def plot_results(results, output_prefix='null_model'):
    """Plot null distributions with observed values marked."""
    obs = results['observed']
    pvals = results['p_values']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, metric, title in zip(axes, ['ari', 'nmi', 'ami'],
                                  ['Adjusted Rand Index', 'Normalized Mutual Information',
                                   'Adjusted Mutual Information']):
        null_vals = results[f'null_{metric}']
        obs_val = obs[metric]
        p = pvals[metric]

        ax.hist(null_vals, bins=50, density=True, alpha=0.7, color='steelblue',
                edgecolor='white', linewidth=0.5)
        ax.axvline(obs_val, color='red', linestyle='--', linewidth=2,
                   label=f'Observed = {obs_val:.4f}\np = {p:.4f}')
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.set_title(f'{title}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to '{output_prefix}.png'")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Null model for state-behavior cluster alignment'
    )
    parser.add_argument(
        '--state-clusters', type=str, default='state_clusters.json',
        help='State clusters JSON file (default: state_clusters.json)'
    )
    parser.add_argument(
        '--tactic-clusters', type=str, default='behavior_clusters.json',
        help='Tactic clusters JSON file (default: behavior_clusters.json)'
    )
    parser.add_argument(
        '--tactic-state-mapping', type=str,
        default='analysis_tactic_state_mapping.json',
        help='Tactic-state mapping JSON file'
    )
    parser.add_argument(
        '--permutations', '-n', type=int, default=10000,
        help='Number of random permutations (default: 10000)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='null_model',
        help='Output file prefix for plots (default: null_model)'
    )

    args = parser.parse_args()

    print("Null Model: State-Behavior Cluster Alignment")
    print(f"  Permutations: {args.permutations}")
    print(f"  Seed: {args.seed}")

    state_clusters = load_state_clusters(args.state_clusters)
    tactic_clusters = load_tactic_clusters(args.tactic_clusters)
    tactic_state_mapping = load_tactic_state_mapping(args.tactic_state_mapping)

    results = run_null_model(
        state_clusters, tactic_clusters, tactic_state_mapping,
        n_permutations=args.permutations, seed=args.seed
    )

    print_results(results)
    plot_results(results, args.output)
