"""
Extract behavioral patterns (transition sequences) that characterize each cluster.
Reads from behavior_clusters.json and analysis_tactic_transitions.json
"""

import json
from collections import defaultdict


def decimal_to_tactic(decimal: int) -> str:
    """Convert decimal (0-63) back to tactic string representation."""
    bit5 = (decimal >> 5) & 1
    bit4 = (decimal >> 4) & 1
    bit3 = (decimal >> 3) & 1
    bit2 = (decimal >> 2) & 1
    bit1 = (decimal >> 1) & 1
    bit0 = decimal & 1

    combo = ''
    combo += 'B' if bit5 else 'C'
    combo += 'F' if bit4 else '-'
    combo += 'S' if bit3 else '-'
    combo += 'K' if bit2 else '-'
    combo += 'W' if bit1 else 'O'
    action = 'T' if bit0 else 'F'

    return f"{combo}_{action}"


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


# Load cluster assignments
with open('behavior_clusters.json', 'r') as f:
    clusters_data = json.load(f)

# Load transition graph
with open('analysis_tactic_transitions.json', 'r') as f:
    transitions_data = json.load(f)

# Build tactic -> cluster mapping
tactic_to_cluster = {}
for tactic_str, cluster_id in clusters_data['by_tactic_string'].items():
    tactic_to_cluster[tactic_str] = cluster_id

# Build adjacency list with weights (for finding sequences)
adjacency = defaultdict(list)  # source -> [(target, weight), ...]
for edge in transitions_data['edges']:
    src = edge['source']
    tgt = edge['target']
    weight = edge['weight']
    adjacency[src].append((tgt, weight))

# Sort adjacency lists by weight (descending)
for src in adjacency:
    adjacency[src].sort(key=lambda x: -x[1])


def find_sequences(cluster_id, tactic_to_cluster, adjacency, max_len=6, top_n=10):
    """
    Find characteristic sequences within a cluster by following high-weight transitions.
    Returns sequences that stay within the cluster or show common exit/entry patterns.
    """
    # Get tactics in this cluster
    cluster_tactics = set(t for t, c in tactic_to_cluster.items() if c == cluster_id)

    sequences = []

    # Start from each tactic in the cluster
    for start_tactic in cluster_tactics:
        # Follow the highest weight transitions
        seq = [start_tactic]
        current = start_tactic
        visited = {start_tactic}

        for _ in range(max_len - 1):
            if current not in adjacency:
                break

            # Find best next tactic (preferring within-cluster, but showing exits too)
            best_next = None
            best_weight = 0

            for next_tactic, weight in adjacency[current]:
                if next_tactic not in visited and weight > best_weight:
                    best_next = next_tactic
                    best_weight = weight

            if best_next is None:
                break

            seq.append(best_next)
            visited.add(best_next)
            current = best_next

            # If we've left the cluster, stop
            if tactic_to_cluster.get(best_next) != cluster_id:
                break

        if len(seq) >= 2:
            # Calculate total weight of sequence
            total_weight = 0
            for i in range(len(seq) - 1):
                for tgt, w in adjacency[seq[i]]:
                    if tgt == seq[i + 1]:
                        total_weight += w
                        break
            sequences.append((seq, total_weight))

    # Sort by weight and return top sequences
    sequences.sort(key=lambda x: -x[1])

    # Remove duplicates (sequences that are subsets of others)
    unique_sequences = []
    for seq, weight in sequences:
        is_subset = False
        for other_seq, _ in unique_sequences:
            if len(seq) <= len(other_seq):
                # Check if seq is a subsequence
                seq_str = '->'.join(seq)
                other_str = '->'.join(other_seq)
                if seq_str in other_str:
                    is_subset = True
                    break
        if not is_subset:
            unique_sequences.append((seq, weight))

    return unique_sequences[:top_n]


# Group tactics by cluster
clusters = defaultdict(list)
for tactic_str, cluster_id in tactic_to_cluster.items():
    clusters[cluster_id].append(tactic_str)

# Write to file
with open('cluster_tactics.txt', 'w') as f:
    for cluster_id in sorted(clusters.keys()):
        tactics = clusters[cluster_id]
        decimals = sorted([tactic_to_decimal(t) for t in tactics])

        f.write(f"Cluster {cluster_id}\n")
        f.write(f"Tactics: {'-'.join(str(d) for d in decimals)}\n")
        f.write(f"\nBehavioral patterns:\n")

        sequences = find_sequences(cluster_id, tactic_to_cluster, adjacency)

        for seq, weight in sequences:
            # Convert to decimals
            decimal_seq = [tactic_to_decimal(t) for t in seq]
            # Mark cluster boundaries
            cluster_seq = [tactic_to_cluster.get(t, -1) for t in seq]

            # Format: show decimal and mark if it exits cluster
            parts = []
            for i, (d, c) in enumerate(zip(decimal_seq, cluster_seq)):
                parts.append(str(d))
                #if c != cluster_id:
                #    parts.append(f"[{d}]")  # Brackets indicate outside cluster
                #else:
                #    parts.append(str(d))

            f.write(f"  {'-'.join(parts)}  (weight: {weight})\n")

        f.write("\n")

print("Written to cluster_tactics.txt")
