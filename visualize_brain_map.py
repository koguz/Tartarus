"""
Visualize Behavioral Flowchart of the Tartarus Agent.

Generates a graph where:
- Nodes = Tactic Clusters
- Edges = Aggregated transitions between tactic clusters
- Colors = Based on "Specialist" State Cluster (most used non-core state cluster)

Inputs:
- behavior_clusters.json (tactic cluster definitions)
- state_clusters.json (state cluster definitions, uses 'infomap' key)
- analysis_tactic_transitions.json (tactic transition edges)
- analysis_tactic_state_mapping.json (tactic -> states mapping)
"""

import argparse
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def load_data(behavior_clusters_path, state_clusters_path,
              tactic_transitions_path, tactic_state_mapping_path):
    """Load all required JSON files."""
    with open(behavior_clusters_path, 'r') as f:
        behavior_clusters = json.load(f)

    with open(state_clusters_path, 'r') as f:
        state_clusters_data = json.load(f)

    with open(tactic_transitions_path, 'r') as f:
        tactic_transitions = json.load(f)

    with open(tactic_state_mapping_path, 'r') as f:
        tactic_state_mapping = json.load(f)

    return behavior_clusters, state_clusters_data, tactic_transitions, tactic_state_mapping


def identify_core_state_cluster(tactic_clusters, tactic_state_mapping, state_clusters):
    """
    Calculate total usage of each State Cluster across all tactics.
    The one with highest global usage is the "Core Brain".
    """
    state_cluster_usage = defaultdict(int)

    for tactic, tc_id in tactic_clusters.items():
        if tactic in tactic_state_mapping:
            state_counts = tactic_state_mapping[tactic]['state_counts']
            for state_str, count in state_counts.items():
                state = int(state_str)
                if state in state_clusters:
                    sc_id = state_clusters[state]
                    state_cluster_usage[sc_id] += count

    # Find the core (most used)
    core_sc = max(state_cluster_usage, key=state_cluster_usage.get)
    print(f"Core State Cluster: SC{core_sc} (total usage: {state_cluster_usage[core_sc]:,})")

    return core_sc, state_cluster_usage


def compute_specialist_ids(tactic_clusters, tactic_state_mapping, state_clusters, core_sc):
    """
    For each Tactic Cluster, find its "Specialist" State Cluster.
    This is the most-used state cluster AFTER ignoring the core.
    """
    # Group tactics by tactic cluster
    tactics_by_tc = defaultdict(list)
    for tactic, tc_id in tactic_clusters.items():
        tactics_by_tc[tc_id].append(tactic)

    specialist_ids = {}

    for tc_id in sorted(tactics_by_tc.keys()):
        tactics = tactics_by_tc[tc_id]

        # Aggregate state cluster usage for this tactic cluster
        sc_usage = defaultdict(int)
        for tactic in tactics:
            if tactic in tactic_state_mapping:
                state_counts = tactic_state_mapping[tactic]['state_counts']
                for state_str, count in state_counts.items():
                    state = int(state_str)
                    if state in state_clusters:
                        sc_id = state_clusters[state]
                        sc_usage[sc_id] += count

        # Remove core from consideration
        sc_usage_no_core = {sc: usage for sc, usage in sc_usage.items() if sc != core_sc}

        if sc_usage_no_core:
            specialist = max(sc_usage_no_core, key=sc_usage_no_core.get)
            specialist_usage = sc_usage_no_core[specialist]
            total_non_core = sum(sc_usage_no_core.values())
            pct = 100 * specialist_usage / total_non_core if total_non_core > 0 else 0
            print(f"  TC{tc_id}: Specialist = SC{specialist} ({pct:.1f}% of non-core usage)")
        else:
            specialist = -1  # No specialist (all core)
            print(f"  TC{tc_id}: No specialist (all core)")

        specialist_ids[tc_id] = specialist

    return specialist_ids, tactics_by_tc


def aggregate_tactic_cluster_transitions(tactic_transitions, tactic_clusters):
    """
    Aggregate transitions from individual tactics to tactic cluster level.
    Returns a dict: (from_tc, to_tc) -> weight
    """
    tc_transitions = defaultdict(int)
    total_weight = 0

    for edge in tactic_transitions['edges']:
        src_tactic = edge['source']
        tgt_tactic = edge['target']
        weight = edge['weight']

        if src_tactic in tactic_clusters and tgt_tactic in tactic_clusters:
            src_tc = tactic_clusters[src_tactic]
            tgt_tc = tactic_clusters[tgt_tactic]
            tc_transitions[(src_tc, tgt_tc)] += weight
            total_weight += weight

    return tc_transitions, total_weight


def build_graph(tc_transitions, total_weight, tactics_by_tc, specialist_ids,
                min_weight_pct=0.05):
    """
    Build a NetworkX directed graph of tactic clusters.
    Filter edges below min_weight_pct of total transitions.
    """
    G = nx.DiGraph()

    # Add nodes
    for tc_id in sorted(tactics_by_tc.keys()):
        num_tactics = len(tactics_by_tc[tc_id])
        specialist = specialist_ids.get(tc_id, -1)
        G.add_node(tc_id, size=num_tactics, specialist=specialist)

    # Add edges (filtered)
    min_weight = min_weight_pct * total_weight
    for (src_tc, tgt_tc), weight in tc_transitions.items():
        if weight >= min_weight:
            G.add_edge(src_tc, tgt_tc, weight=weight)

    return G


def visualize_brain_map(G, specialist_ids, output_path='brain_map_flowchart.png'):
    """
    Visualize the behavioral flowchart.
    """
    if len(G.nodes()) == 0:
        print("No nodes to visualize.")
        return

    # Get unique specialist IDs for coloring
    unique_specialists = sorted(set(specialist_ids.values()))
    if -1 in unique_specialists:
        unique_specialists.remove(-1)
        unique_specialists.append(-1)  # Put -1 at end

    # Create color map
    cmap = plt.cm.get_cmap('tab10')
    specialist_to_color = {}
    for i, spec in enumerate(unique_specialists):
        if spec == -1:
            specialist_to_color[spec] = 'gray'
        else:
            specialist_to_color[spec] = cmap(i % 10)

    # Node colors
    node_colors = [specialist_to_color[G.nodes[n]['specialist']] for n in G.nodes()]

    # Node sizes (proportional to number of tactics)
    sizes = [G.nodes[n]['size'] for n in G.nodes()]
    min_size, max_size = min(sizes), max(sizes)
    if max_size > min_size:
        node_sizes = [300 + 700 * (s - min_size) / (max_size - min_size) for s in sizes]
    else:
        node_sizes = [500] * len(sizes)

    # Edge widths (proportional to weight)
    if G.edges():
        weights = [G.edges[e]['weight'] for e in G.edges()]
        max_weight = max(weights)
        edge_widths = [1 + 4 * (w / max_weight) for w in weights]
    else:
        edge_widths = []

    # Node labels
    labels = {}
    for n in G.nodes():
        spec = G.nodes[n]['specialist']
        if spec >= 0:
            labels[n] = f"TC{n}\n(Spec: SC{spec})"
        else:
            labels[n] = f"TC{n}\n(Core only)"

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Layout
    pos = nx.kamada_kawai_layout(G)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1'
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='black',
        linewidths=2
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=labels,
        font_size=9,
        font_weight='bold'
    )

    # Edge labels (weights)
    edge_labels = {(u, v): f"{G.edges[u, v]['weight']:,}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=7,
        alpha=0.8
    )

    # Legend for specialist colors
    legend_elements = []
    for spec in unique_specialists:
        if spec >= 0:
            color = specialist_to_color[spec]
            legend_elements.append(mpatches.Patch(color=color, label=f'Specialist: SC{spec}'))
        else:
            legend_elements.append(mpatches.Patch(color='gray', label='Core only'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    ax.set_title('Behavioral Flowchart: Tactic Cluster Transitions\n'
                 '(Node color = Specialist State Cluster, Edge width = Transition frequency)',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved brain map to '{output_path}'")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate Behavioral Flowchart of the Tartarus Agent'
    )
    parser.add_argument(
        '--behavior-clusters', '-b',
        type=str,
        default='behavior_clusters.json',
        help='Tactic cluster definitions (default: behavior_clusters.json)'
    )
    parser.add_argument(
        '--state-clusters', '-s',
        type=str,
        default='state_clusters.json',
        help='State cluster definitions (default: state_clusters.json)'
    )
    parser.add_argument(
        '--tactic-transitions', '-t',
        type=str,
        default='analysis_tactic_transitions.json',
        help='Tactic transition graph (default: analysis_tactic_transitions.json)'
    )
    parser.add_argument(
        '--tactic-state-mapping', '-m',
        type=str,
        default='analysis_tactic_state_mapping.json',
        help='Tactic-state mapping (default: analysis_tactic_state_mapping.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='brain_map_flowchart.png',
        help='Output image path (default: brain_map_flowchart.png)'
    )
    parser.add_argument(
        '--min-edge-pct',
        type=float,
        default=0.01,
        help='Minimum edge weight as fraction of total (default: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--use-spectral',
        action='store_true',
        help='Use spectral clustering instead of infomap for state clusters'
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    behavior_clusters, state_clusters_data, tactic_transitions, tactic_state_mapping = load_data(
        args.behavior_clusters,
        args.state_clusters,
        args.tactic_transitions,
        args.tactic_state_mapping
    )

    # Extract tactic clusters
    tactic_clusters = behavior_clusters['by_tactic_string']
    print(f"  Loaded {len(tactic_clusters)} tactics in {max(tactic_clusters.values()) + 1} clusters")

    # Extract state clusters (infomap by default)
    cluster_key = 'spectral' if args.use_spectral else 'infomap'
    if cluster_key not in state_clusters_data or 'clusters' not in state_clusters_data[cluster_key]:
        print(f"Warning: '{cluster_key}' clusters not found, trying alternative...")
        cluster_key = 'spectral' if cluster_key == 'infomap' else 'infomap'

    state_clusters = {int(k): v for k, v in state_clusters_data[cluster_key]['clusters'].items()}
    n_state_clusters = max(state_clusters.values()) + 1
    print(f"  Loaded {len(state_clusters)} states in {n_state_clusters} clusters ({cluster_key})")

    # Identify core state cluster
    print("\nIdentifying Core State Cluster...")
    core_sc, sc_usage = identify_core_state_cluster(tactic_clusters, tactic_state_mapping, state_clusters)

    # Compute specialist IDs
    print("\nComputing Specialist IDs for each Tactic Cluster...")
    specialist_ids, tactics_by_tc = compute_specialist_ids(
        tactic_clusters, tactic_state_mapping, state_clusters, core_sc
    )

    # Aggregate transitions
    print("\nAggregating tactic cluster transitions...")
    tc_transitions, total_weight = aggregate_tactic_cluster_transitions(
        tactic_transitions, tactic_clusters
    )
    print(f"  Total transitions: {total_weight:,}")
    print(f"  Unique TC->TC edges: {len(tc_transitions)}")

    # Build graph
    print(f"\nBuilding graph (min edge weight: {args.min_edge_pct:.1%} of total)...")
    G = build_graph(tc_transitions, total_weight, tactics_by_tc, specialist_ids,
                    min_weight_pct=args.min_edge_pct)
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Visualize
    print("\nGenerating visualization...")
    visualize_brain_map(G, specialist_ids, args.output)


if __name__ == '__main__':
    main()
