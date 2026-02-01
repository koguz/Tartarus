"""
State Clustering for Tartarus Agent.

Clusters the 128-state transition graph using:
1. Spectral Clustering (same method as tactic clustering for consistency)
2. Infomap (designed for directed graphs, captures flow patterns)

Then compares results with tactic clusters to find overlap.
"""

import argparse
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt

# Try to import infomap, but don't fail if not installed
try:
    import infomap
    INFOMAP_AVAILABLE = True
except ImportError:
    INFOMAP_AVAILABLE = False
    print("Note: infomap not installed. Run 'pip install infomap' to enable Infomap clustering.")


def load_transition_graph(json_path: str):
    """Load state transition graph from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['nodes'], data['edges']


def normalize_transition_weights(nodes: list, edges: list) -> np.ndarray:
    """Normalize transition weights row-wise (outgoing edges sum to 1 per node)."""
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    weight_matrix = np.zeros((n, n))
    for edge in edges:
        src_idx = node_to_idx[edge['source']]
        tgt_idx = node_to_idx[edge['target']]
        weight_matrix[src_idx, tgt_idx] = edge['weight']

    row_sums = weight_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized = weight_matrix / row_sums

    return normalized


def spectral_clustering(nodes: list, edges: list, plot_eigenvalues: bool = True,
                        k_override: int = None, min_k: int = 2, max_k: int = 15):
    """
    Perform spectral clustering on state transition graph.
    Returns cluster labels for each node.

    Args:
        k_override: Force a specific number of clusters (ignores eigengap)
        min_k: Minimum k to consider in eigengap analysis
        max_k: Maximum k to consider in eigengap analysis
    """
    n = len(nodes)
    print(f"\n{'='*60}")
    print("SPECTRAL CLUSTERING")
    print("="*60)

    # Step 1: Normalize weights
    print("Step 1: Normalizing transition weights...")
    normalized_weights = normalize_transition_weights(nodes, edges)

    # Step 2: Symmetrize (S = W + W^T)
    print("Step 2: Creating symmetric similarity matrix...")
    symmetric_sim = normalized_weights + normalized_weights.T

    # Step 3: Construct Laplacian (L = D - S)
    print("Step 3: Constructing Laplacian matrix...")
    D = np.diag(symmetric_sim.sum(axis=1))
    L = D - symmetric_sim

    # Step 4: Compute eigenvalues and eigenvectors
    print("Step 4: Computing eigenvalues and eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    print(f"   First 10 eigenvalues: {eigenvalues[:10]}")

    # Step 5: Eigengap analysis
    print("Step 5: Performing eigengap analysis...")
    gaps = np.diff(eigenvalues)

    # Search for max gap between min_k and max_k
    search_start = max(1, min_k - 1)  # Skip first gap, adjust for 0-indexing
    search_end = min(len(gaps), max_k)
    search_gaps = gaps[search_start:search_end]
    best_idx = np.argmax(search_gaps) + search_start
    eigengap_k = best_idx + 1

    print(f"   Eigengap suggests k={eigengap_k}")

    if k_override is not None:
        optimal_k = k_override
        print(f"   Using override k={optimal_k}")
    else:
        optimal_k = eigengap_k

    if plot_eigenvalues:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        ax1.plot(range(1, min(21, n+1)), eigenvalues[:20], 'bo-', markersize=8)
        ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Spectral Clustering: Sorted Eigenvalues (First 20)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.bar(range(2, min(21, n+1)), gaps[:19], color='steelblue', alpha=0.7)
        ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Eigengap')
        ax2.set_title('Eigengaps')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('state_spectral_eigengap.png', dpi=150, bbox_inches='tight')
        print("   Saved eigenvalue plot to 'state_spectral_eigengap.png'")
        plt.close()

    # Step 6: K-Means on eigenvectors
    print(f"Step 6: Running K-Means with {optimal_k} clusters...")
    feature_matrix = eigenvectors[:, :optimal_k]
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    # Create result
    result = {nodes[i]: int(cluster_labels[i]) for i in range(n)}

    # Print summary
    print(f"\nSpectral Clustering Results:")
    print(f"  Number of clusters: {optimal_k}")

    clusters = {}
    for node, label in result.items():
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)

    for label in sorted(clusters.keys()):
        members = sorted(clusters[label])
        print(f"\n  Cluster {label} ({len(members)} states):")
        print(f"    {'-'.join(str(s) for s in members)}")

    return result, optimal_k


def infomap_clustering(nodes: list, edges: list, markov_time: float = 1.0,
                       num_trials: int = 10, preferred_num_modules: int = None):
    """
    Perform Infomap clustering on directed state transition graph.
    Returns cluster labels for each node.

    Args:
        markov_time: Markov time scale (>1 = larger clusters, <1 = smaller clusters)
        num_trials: Number of optimization trials
        preferred_num_modules: Preferred number of modules (soft constraint)
    """
    if not INFOMAP_AVAILABLE:
        print("\nInfomap not available. Skipping.")
        return None, 0

    print(f"\n{'='*60}")
    print("INFOMAP CLUSTERING")
    print("="*60)

    # Build Infomap command
    cmd = f"--directed --silent --num-trials {num_trials} --markov-time {markov_time}"
    if preferred_num_modules is not None:
        cmd += f" --preferred-number-of-modules {preferred_num_modules}"
    print(f"  Parameters: {cmd}")

    # Create Infomap instance
    im = infomap.Infomap(cmd)

    # Add edges
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    for edge in edges:
        src_idx = node_to_idx[edge['source']]
        tgt_idx = node_to_idx[edge['target']]
        weight = edge['weight']
        im.add_link(src_idx, tgt_idx, weight)

    # Run Infomap
    print("Running Infomap algorithm...")
    im.run()

    # Extract cluster assignments
    result = {}
    num_clusters = im.num_top_modules
    print(f"  Found {num_clusters} clusters")

    for node in im.tree:
        if node.is_leaf:
            result[nodes[node.node_id]] = node.module_id - 1  # 0-indexed

    # Print summary
    clusters = {}
    for node, label in result.items():
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)

    for label in sorted(clusters.keys()):
        members = sorted(clusters[label])
        print(f"\n  Cluster {label} ({len(members)} states):")
        print(f"    {'-'.join(str(s) for s in members)}")

    return result, num_clusters


def load_tactic_state_mapping(json_path: str):
    """Load tactic -> states mapping."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_tactic_clusters(json_path: str):
    """Load tactic clusters."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['by_tactic_string']


def tactic_to_decimal(tactic_str: str) -> int:
    """Convert tactic string to decimal."""
    combo, action = tactic_str.split('_')
    binary = 0
    binary |= (1 if combo[0] == 'B' else 0) << 5
    binary |= (1 if combo[1] == 'F' else 0) << 4
    binary |= (1 if combo[2] == 'S' else 0) << 3
    binary |= (1 if combo[3] == 'K' else 0) << 2
    binary |= (1 if combo[4] == 'W' else 0) << 1
    binary |= (1 if action == 'T' else 0)
    return binary


def compute_overlap_metrics(state_clusters: dict, tactic_clusters: dict,
                            tactic_state_mapping: dict, method_name: str):
    """
    Compute overlap metrics between state clusters and tactic clusters.

    Since states and tactics are different entities, we:
    1. For each tactic, compute its "dominant" state cluster (weighted by usage count)
    2. Compare tactic cluster assignments with state-derived assignments using ARI/NMI
    3. Also compute a contingency-based analysis
    """
    print(f"\n{'='*60}")
    print(f"OVERLAP METRICS: Tactic Clusters vs {method_name} State Clusters")
    print("="*60)

    # Get all tactics that have state mapping data
    tactics_with_data = [t for t in tactic_clusters.keys() if t in tactic_state_mapping]

    if not tactics_with_data:
        print("No tactic-state mapping data available.")
        return {}

    # For each tactic, find its dominant state cluster (weighted by state usage)
    tactic_to_dominant_state_cluster = {}
    tactic_state_cluster_weights = {}  # For detailed analysis

    for tactic in tactics_with_data:
        state_counts = tactic_state_mapping[tactic]['state_counts']

        # Aggregate counts by state cluster
        cluster_weights = {}
        total_weight = 0
        for state_str, count in state_counts.items():
            state = int(state_str)
            if state in state_clusters:
                sc = state_clusters[state]
                cluster_weights[sc] = cluster_weights.get(sc, 0) + count
                total_weight += count

        if cluster_weights:
            # Find dominant cluster
            dominant = max(cluster_weights, key=cluster_weights.get)
            tactic_to_dominant_state_cluster[tactic] = dominant

            # Normalize weights
            tactic_state_cluster_weights[tactic] = {
                sc: w / total_weight for sc, w in cluster_weights.items()
            }

    # Create aligned arrays for metrics computation
    tactic_cluster_labels = []
    state_derived_labels = []
    tactic_names = []

    for tactic in sorted(tactics_with_data):
        if tactic in tactic_to_dominant_state_cluster:
            tactic_cluster_labels.append(tactic_clusters[tactic])
            state_derived_labels.append(tactic_to_dominant_state_cluster[tactic])
            tactic_names.append(tactic)

    if len(tactic_cluster_labels) < 2:
        print("Not enough data for metrics computation.")
        return {}

    # Compute clustering comparison metrics
    ari = adjusted_rand_score(tactic_cluster_labels, state_derived_labels)
    nmi = normalized_mutual_info_score(tactic_cluster_labels, state_derived_labels)
    ami = adjusted_mutual_info_score(tactic_cluster_labels, state_derived_labels)

    print(f"\nMetrics (comparing tactic clusters vs dominant state cluster per tactic):")
    print(f"  Adjusted Rand Index (ARI):           {ari:.4f}")
    print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"  Adjusted Mutual Information (AMI):   {ami:.4f}")
    print(f"\n  Interpretation:")
    print(f"    ARI:  -1 to 1, where 1 = perfect agreement, 0 = random, <0 = worse than random")
    print(f"    NMI:  0 to 1, where 1 = perfect agreement, 0 = no mutual information")
    print(f"    AMI:  adjusted for chance, 0 = random agreement, 1 = perfect")

    # Build contingency table
    n_tactic_clusters = max(tactic_clusters.values()) + 1
    n_state_clusters = max(state_clusters.values()) + 1

    # Weighted contingency: how much each (tactic_cluster, state_cluster) pair co-occurs
    contingency = np.zeros((n_tactic_clusters, n_state_clusters))

    for tactic in tactics_with_data:
        tc = tactic_clusters[tactic]
        if tactic in tactic_state_cluster_weights:
            for sc, weight in tactic_state_cluster_weights[tactic].items():
                contingency[tc, sc] += weight

    # Normalize rows (each tactic cluster sums to 1)
    row_sums = contingency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    contingency_norm = contingency / row_sums

    print(f"\nContingency Table (row-normalized, rows=tactic clusters, cols=state clusters):")
    print(f"  Shows what fraction of each tactic cluster's state usage falls into each state cluster.\n")

    # Header
    header = "Tactic\\State |" + "".join(f"  SC{i:2d} " for i in range(n_state_clusters))
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for tc in range(n_tactic_clusters):
        row = f"    TC{tc:2d}     |"
        for sc in range(n_state_clusters):
            val = contingency_norm[tc, sc]
            if val > 0.01:
                row += f"  {val:.2f} "
            else:
                row += "    -  "
        print(f"  {row}")

    # Compute entropy-based measures
    # How concentrated is each tactic cluster's state usage?
    print(f"\nConcentration Analysis:")
    print(f"  (Lower entropy = more concentrated in fewer state clusters)")

    for tc in range(n_tactic_clusters):
        row = contingency_norm[tc]
        # Compute entropy
        entropy = -np.sum(row[row > 0] * np.log2(row[row > 0]))
        max_entropy = np.log2(n_state_clusters)  # If uniform
        concentration = 1 - (entropy / max_entropy) if max_entropy > 0 else 1

        # Find dominant state clusters
        sorted_idx = np.argsort(-row)
        top_clusters = [(i, row[i]) for i in sorted_idx[:3] if row[i] > 0.05]
        top_str = ", ".join(f"SC{i}:{v:.0%}" for i, v in top_clusters)

        print(f"    Tactic Cluster {tc}: entropy={entropy:.2f}, concentration={concentration:.2f}")
        print(f"      Top state clusters: {top_str}")

    # Overall uniformity measure
    # If tactic clusters mapped perfectly to state clusters, each row would have one 1.0
    # Compute average max value per row as a simple alignment score
    alignment_score = np.mean([max(contingency_norm[tc]) for tc in range(n_tactic_clusters)])
    print(f"\n  Overall Alignment Score: {alignment_score:.2%}")
    print(f"    (Average max weight per tactic cluster; 100% = perfect 1-to-1 mapping)")

    return {
        'ari': ari,
        'nmi': nmi,
        'ami': ami,
        'alignment_score': alignment_score,
        'contingency': contingency_norm.tolist()
    }


def compare_clusterings(state_clusters: dict, tactic_clusters: dict,
                        tactic_state_mapping: dict, method_name: str):
    """
    Compare state clusters with tactic clusters.

    For each tactic cluster, find which states are used by tactics in that cluster,
    then see how those states are distributed across state clusters.
    """
    print(f"\n{'='*60}")
    print(f"COMPARISON: Tactic Clusters vs {method_name} State Clusters")
    print("="*60)

    # Group tactics by cluster
    tactics_by_cluster = {}
    for tactic, cluster_id in tactic_clusters.items():
        if cluster_id not in tactics_by_cluster:
            tactics_by_cluster[cluster_id] = []
        tactics_by_cluster[cluster_id].append(tactic)

    # For each tactic cluster, find which states are involved
    for tactic_cluster_id in sorted(tactics_by_cluster.keys()):
        tactics = tactics_by_cluster[tactic_cluster_id]
        tactic_decimals = sorted([tactic_to_decimal(t) for t in tactics])

        print(f"\nTactic Cluster {tactic_cluster_id}:")
        print(f"  Tactics: {'-'.join(str(d) for d in tactic_decimals)}")

        # Collect all states used by these tactics
        states_used = set()
        for tactic in tactics:
            if tactic in tactic_state_mapping:
                states_used.update(tactic_state_mapping[tactic]['states'])

        if not states_used:
            print(f"  No state data for these tactics")
            continue

        # See how these states are distributed across state clusters
        state_cluster_distribution = {}
        for state in states_used:
            if state in state_clusters:
                sc = state_clusters[state]
                if sc not in state_cluster_distribution:
                    state_cluster_distribution[sc] = []
                state_cluster_distribution[sc].append(state)

        print(f"  States used: {len(states_used)}")
        print(f"  State cluster distribution:")
        for sc in sorted(state_cluster_distribution.keys()):
            states_in_sc = sorted(state_cluster_distribution[sc])
            print(f"    State Cluster {sc}: {len(states_in_sc)} states")
            if len(states_in_sc) <= 20:
                print(f"      {'-'.join(str(s) for s in states_in_sc)}")


def save_results(spectral_result: dict, infomap_result: dict, output_path: str,
                 spectral_metrics: dict = None, infomap_metrics: dict = None):
    """Save clustering results to JSON."""
    output = {
        'spectral': {
            'clusters': spectral_result if spectral_result else {},
            'metrics': spectral_metrics if spectral_metrics else {}
        },
        'infomap': {
            'clusters': infomap_result if infomap_result else {},
            'metrics': infomap_metrics if infomap_metrics else {}
        }
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to '{output_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='State clustering for Tartarus agent using spectral and infomap methods'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='analysis_transitions.json',
        help='Input JSON file with state transitions (default: analysis_transitions.json)'
    )
    parser.add_argument(
        '--tactic-clusters', '-t',
        type=str,
        default='behavior_clusters.json',
        help='Tactic clusters JSON file (default: behavior_clusters.json)'
    )
    parser.add_argument(
        '--tactic-state-mapping', '-m',
        type=str,
        default='analysis_tactic_state_mapping.json',
        help='Tactic-state mapping JSON file (default: analysis_tactic_state_mapping.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='state_clusters.json',
        help='Output JSON file (default: state_clusters.json)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable eigenvalue analysis plot'
    )
    parser.add_argument(
        '--k', '-k',
        type=int,
        default=None,
        help='Force specific number of clusters for spectral clustering (overrides eigengap)'
    )
    parser.add_argument(
        '--min-k',
        type=int,
        default=2,
        help='Minimum k to consider in eigengap analysis (default: 2)'
    )
    parser.add_argument(
        '--max-k',
        type=int,
        default=15,
        help='Maximum k to consider in eigengap analysis (default: 15)'
    )
    parser.add_argument(
        '--markov-time',
        type=float,
        default=1.0,
        help='Infomap Markov time scale: >1 for larger clusters, <1 for smaller (default: 1.0)'
    )
    parser.add_argument(
        '--infomap-modules',
        type=int,
        default=None,
        help='Preferred number of modules for Infomap (soft constraint)'
    )

    args = parser.parse_args()

    # Load state transition graph
    print("Loading state transition graph...")
    nodes, edges = load_transition_graph(args.input)
    print(f"  {len(nodes)} nodes, {len(edges)} edges")

    # Run spectral clustering
    spectral_result, spectral_k = spectral_clustering(
        nodes, edges,
        plot_eigenvalues=not args.no_plot,
        k_override=args.k,
        min_k=args.min_k,
        max_k=args.max_k
    )

    # Run Infomap clustering
    infomap_result, infomap_k = infomap_clustering(
        nodes, edges,
        markov_time=args.markov_time,
        preferred_num_modules=args.infomap_modules
    )

    # Try to load tactic data for comparison
    spectral_metrics = {}
    infomap_metrics = {}

    try:
        print("\nLoading tactic data for comparison...")
        tactic_clusters = load_tactic_clusters(args.tactic_clusters)
        tactic_state_mapping = load_tactic_state_mapping(args.tactic_state_mapping)

        # Compute overlap metrics
        spectral_metrics = compute_overlap_metrics(
            spectral_result, tactic_clusters, tactic_state_mapping, "Spectral"
        )

        if infomap_result:
            infomap_metrics = compute_overlap_metrics(
                infomap_result, tactic_clusters, tactic_state_mapping, "Infomap"
            )

        # Detailed comparison (state distributions per tactic cluster)
        print("\n" + "="*60)
        print("DETAILED STATE DISTRIBUTIONS")
        print("="*60)
        compare_clusterings(spectral_result, tactic_clusters, tactic_state_mapping, "Spectral")

        if infomap_result:
            compare_clusterings(infomap_result, tactic_clusters, tactic_state_mapping, "Infomap")

    except FileNotFoundError as e:
        print(f"\nCould not load tactic data for comparison: {e}")
        print("Run analyze_agent.py first to generate tactic_state_mapping.json")

    # Save results (including metrics)
    save_results(spectral_result, infomap_result, args.output, spectral_metrics, infomap_metrics)
