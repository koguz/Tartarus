"""
Spectral Clustering-based Behavior Segmentation for Tartarus Agent Tactics.

This module performs behavior segmentation using spectral clustering on the
tactic transition graph. It combines transition probabilities with Hamming
similarity between tactic encodings to identify distinct behavioral clusters.

Method based on spectral clustering with Hamming distance regularization.
"""

import argparse
import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def tactic_to_binary(tactic_str: str) -> np.ndarray:
    """
    Convert tactic string encoding to 6-bit binary representation.

    Encoding format: "XFSKW_A" where:
    - X: B (box in front) = 1, C (clear) = 0
    - F: F (front row box) = 1, - = 0
    - S: S (side boxes) = 1, - = 0
    - K: K (back box) = 1, - = 0
    - W: W (walls) = 1, O (open) = 0
    - A: F (forward) = 0, T (turn) = 1

    Returns:
        6-element binary numpy array
    """
    combo, action = tactic_str.split('_')

    binary = np.zeros(6, dtype=int)

    # Position 0: B (box in front) vs C (clear)
    binary[0] = 1 if combo[0] == 'B' else 0

    # Position 1: F (front row box) vs -
    binary[1] = 1 if combo[1] == 'F' else 0

    # Position 2: S (side boxes) vs -
    binary[2] = 1 if combo[2] == 'S' else 0

    # Position 3: K (back box) vs -
    binary[3] = 1 if combo[3] == 'K' else 0

    # Position 4: W (walls) vs O (open)
    binary[4] = 1 if combo[4] == 'W' else 0

    # Position 5: F (forward) = 0, T (turn) = 1
    binary[5] = 1 if action == 'T' else 0

    return binary


def tactic_to_decimal(tactic_str: str) -> int:
    """
    Convert tactic string encoding to decimal representation of 6-bit binary.

    Returns:
        Integer (0-63) representing the binary encoding
    """
    binary = tactic_to_binary(tactic_str)
    # Convert binary array to decimal (MSB first)
    decimal = 0
    for bit in binary:
        decimal = (decimal << 1) | bit
    return decimal


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Hamming distance between two binary vectors."""
    return np.sum(a != b)


def compute_hamming_similarity_matrix(nodes: list) -> np.ndarray:
    """
    Compute 64x64 Hamming similarity matrix for all tactic nodes.

    S_{i,j} = 1 - HammingDistance(c_i, c_j) / 6
    """
    n = len(nodes)
    binary_codes = [tactic_to_binary(node) for node in nodes]

    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hd = hamming_distance(binary_codes[i], binary_codes[j])
            similarity[i, j] = 1.0 - hd / 6.0

    return similarity


def normalize_transition_weights(nodes: list, edges: list) -> np.ndarray:
    """
    Normalize transition weights row-wise (outgoing edges sum to 1 per node).

    Returns:
        (n, n) normalized weight matrix
    """
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Build raw weight matrix
    weight_matrix = np.zeros((n, n))
    for edge in edges:
        src_idx = node_to_idx[edge['source']]
        tgt_idx = node_to_idx[edge['target']]
        weight_matrix[src_idx, tgt_idx] = edge['weight']

    # Normalize row-wise
    row_sums = weight_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = weight_matrix / row_sums

    return normalized


def create_adjusted_weight_matrix(normalized_weights: np.ndarray,
                                   hamming_similarity: np.ndarray,
                                   lambda_param: float = 0.45) -> np.ndarray:
    """
    Create adjusted weight matrix combining transition probabilities and Hamming dissimilarity.

    w_{ij} = (1 - lambda) * NormalizedWeight + lambda * (1 - S_{ij})

    Note: (1 - S_{ij}) represents Hamming dissimilarity, penalizing similar nodes
    to encourage diverse behavior clustering.
    """
    hamming_dissimilarity = 1.0 - hamming_similarity
    adjusted = (1 - lambda_param) * normalized_weights + lambda_param * hamming_dissimilarity
    return adjusted


def compute_symmetric_similarity(weight_matrix: np.ndarray) -> np.ndarray:
    """
    Compute symmetric similarity matrix: S_{ij} = w_{ij} + w_{ji}
    """
    return weight_matrix + weight_matrix.T


def compute_laplacian(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Construct the graph Laplacian: L = D - S
    where D_{ii} = sum_j S_{ij}
    """
    D = np.diag(similarity_matrix.sum(axis=1))
    L = D - similarity_matrix
    return L


def find_optimal_k_eigengap(eigenvalues: np.ndarray, min_k: int = 2, max_k: int = 15) -> int:
    """
    Find optimal number of clusters using eigengap analysis.

    The optimal k is where the gap between consecutive eigenvalues is maximized.
    We ignore the first gap (index 0 to 1) since the first eigenvalue is ~0.
    """
    # Calculate gaps between consecutive eigenvalues
    gaps = np.diff(eigenvalues)

    # Search for max gap in range [min_k-1, max_k-1] (adjusting for 0-indexing)
    # Gap[i] = eigenvalue[i+1] - eigenvalue[i], so gap[k-1] tells us about cluster k
    search_start = max(1, min_k - 1)  # Start from index 1 to skip first gap
    search_end = min(len(gaps), max_k)

    if search_end <= search_start:
        return min_k

    search_gaps = gaps[search_start:search_end]
    best_idx = np.argmax(search_gaps) + search_start

    # The optimal k is the index where the gap starts (before the jump)
    # So if gap[i] is max, we use i+1 eigenvectors (k = i+1)
    optimal_k = best_idx + 1

    return optimal_k


def spectral_clustering_behavior_segmentation(json_path: str,
                                               lambda_param: float = 0.45,
                                               plot_eigenvalues: bool = True) -> dict:
    """
    Perform spectral clustering-based behavior segmentation.

    Args:
        json_path: Path to the tactic transitions JSON file
        lambda_param: Weight for Hamming dissimilarity (default: 0.45)
        plot_eigenvalues: Whether to plot eigenvalue analysis

    Returns:
        Dictionary mapping each tactic node ID to its cluster label (0 to k-1)
    """
    # Load tactic transition graph
    with open(json_path, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    edges = data['edges']
    n = len(nodes)

    print(f"Loaded {n} tactic nodes and {len(edges)} edges")

    # Step 1: Normalize transition weights
    print("Step 1: Normalizing transition weights...")
    normalized_weights = normalize_transition_weights(nodes, edges)

    # Step 2: Compute Hamming similarity matrix
    print("Step 2: Computing Hamming similarity matrix...")
    hamming_sim = compute_hamming_similarity_matrix(nodes)

    # Step 3: Create adjusted weight matrix
    print(f"Step 3: Creating adjusted weight matrix (lambda={lambda_param})...")
    adjusted_weights = create_adjusted_weight_matrix(normalized_weights, hamming_sim, lambda_param)

    # Step 4: Compute symmetric similarity matrix
    print("Step 4: Computing symmetric similarity matrix...")
    symmetric_sim = compute_symmetric_similarity(adjusted_weights)

    # Step 5: Construct Laplacian
    print("Step 5: Constructing Laplacian matrix...")
    laplacian = compute_laplacian(symmetric_sim)

    # Step 6: Compute eigenvalues and eigenvectors
    print("Step 6: Computing eigenvalues and eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    # Sort by eigenvalue (should already be sorted, but ensure)
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    print(f"   First 10 eigenvalues: {eigenvalues[:10]}")

    # Step 7: Eigengap analysis to find optimal k
    print("Step 7: Performing eigengap analysis...")
    optimal_k = find_optimal_k_eigengap(eigenvalues)
    print(f"   Optimal k from eigengap analysis: {optimal_k}")

    # Plot eigenvalue analysis if requested
    if plot_eigenvalues:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot eigenvalues
        ax1 = axes[0]
        ax1.plot(range(1, min(21, n+1)), eigenvalues[:20], 'bo-', markersize=8)
        ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Sorted Eigenvalues (First 20)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot eigengaps
        ax2 = axes[1]
        gaps = np.diff(eigenvalues[:20])
        ax2.bar(range(2, min(21, n+1)), gaps, color='steelblue', alpha=0.7)
        ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Eigengap')
        ax2.set_title('Eigengaps (Difference Between Consecutive Eigenvalues)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('eigengap_analysis.png', dpi=150, bbox_inches='tight')
        print("   Saved eigenvalue analysis plot to 'eigengap_analysis.png'")
        plt.close()

    # Step 8: Select first k eigenvectors
    print(f"Step 8: Selecting first {optimal_k} eigenvectors...")
    feature_matrix = eigenvectors[:, :optimal_k]
    print(f"   Feature matrix shape: {feature_matrix.shape}")

    # Step 9: K-Means clustering
    print(f"Step 9: Running K-Means with {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    # Create result dictionary (using string keys for JSON compatibility, but storing decimal)
    result = {nodes[i]: int(cluster_labels[i]) for i in range(n)}

    # Also create decimal-keyed result for output
    result_decimal = {tactic_to_decimal(nodes[i]): int(cluster_labels[i]) for i in range(n)}

    # Print cluster summary
    print("\n" + "="*60)
    print("BEHAVIOR SEGMENTATION RESULTS")
    print("="*60)
    print(f"Number of clusters: {optimal_k}")
    print(f"Lambda parameter: {lambda_param}")

    # Group nodes by cluster
    clusters = {}
    for node, label in result.items():
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node)

    for label in sorted(clusters.keys()):
        members = clusters[label]
        # Convert to decimal representation and sort
        decimals = sorted([tactic_to_decimal(m) for m in members])
        decimal_str = '-'.join(str(d) for d in decimals)
        print(f"\nCluster {label} ({len(members)} tactics):")
        print(f"  {decimal_str}")

    return result, result_decimal


def save_results(result: dict, result_decimal: dict, output_path: str = 'behavior_clusters.json'):
    """Save clustering results to JSON file."""
    output = {
        'by_tactic_string': result,
        'by_decimal': {str(k): v for k, v in sorted(result_decimal.items())}
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to '{output_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Spectral clustering-based behavior segmentation for Tartarus tactics'
    )
    parser.add_argument(
        '--lambda', '-l',
        type=float,
        default=0.45,
        dest='lambda_param',
        help='Lambda parameter for Hamming dissimilarity weight (default: 0.45)'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='analysis_tactic_transitions.json',
        help='Input JSON file with tactic transitions (default: analysis_tactic_transitions.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='behavior_clusters.json',
        help='Output JSON file for cluster results (default: behavior_clusters.json)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable eigenvalue analysis plot'
    )

    args = parser.parse_args()

    # Run behavior segmentation
    result, result_decimal = spectral_clustering_behavior_segmentation(
        args.input,
        lambda_param=args.lambda_param,
        plot_eigenvalues=not args.no_plot
    )

    # Save results
    save_results(result, result_decimal, args.output)
