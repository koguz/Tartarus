#!/usr/bin/env python3
"""
Community detection on the state transition graph.

Finds clusters of states that form behavioral modes:
- "searching", "wall-following", "stuck loop", "corridor traversal", etc.

Uses Infomap algorithm which respects edge direction (flow-based detection).

For each community:
- Top states and actions
- Average dwell time before exit
- Exit edges to other communities

Usage:
    python analyze_communities.py [analysis_prefix] [--markov-time T]

Options:
    --markov-time T    Resolution parameter (default: 1.0)
                       Higher = more smaller communities
                       Lower = fewer larger communities

Install infomap:
    pip install infomap
"""

import json
import pickle
import sys
import networkx as nx
from collections import defaultdict, Counter


def load_data(prefix='analysis'):
    """Load transitions, state stats, and sequences."""
    with open(f'{prefix}_transitions.json', 'r') as f:
        transitions = json.load(f)
    with open(f'{prefix}_state_stats.json', 'r') as f:
        state_stats = json.load(f)
    with open(f'{prefix}_sequences.pkl', 'rb') as f:
        sequences = pickle.load(f)
    return transitions, state_stats, sequences


def build_graph(transitions):
    """Build a networkx DiGraph from transitions."""
    G = nx.DiGraph()

    for node in transitions.get('nodes', []):
        G.add_node(node)

    for edge in transitions.get('edges', []):
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

    return G


def get_state_action_profile(state_id, state_stats):
    """Get action distribution for a state."""
    stats = state_stats.get(str(state_id), {})
    action_counts = stats.get('action_counts', {})
    return action_counts


def get_dominant_action(state_id, state_stats):
    """Get dominant action for a state (P/F/T)."""
    action_counts = get_state_action_profile(state_id, state_stats)
    if not action_counts:
        return '?'

    push = action_counts.get('push', 0)
    forward = action_counts.get('forward', 0)
    turn = action_counts.get('turn_left', 0) + action_counts.get('turn_right', 0)

    if push >= forward and push >= turn:
        return 'P'
    elif forward >= turn:
        return 'F'
    else:
        return 'T'


def detect_communities_infomap(G, markov_time=1.0):
    """
    Detect communities using Infomap algorithm on DIRECTED graph.

    Infomap uses random walk dynamics to find communities where flow tends to stay.
    This is ideal for state transition graphs.

    Args:
        G: NetworkX DiGraph with edge weights
        markov_time: Resolution parameter (default 1.0)
                     Higher = more smaller communities
                     Lower = fewer larger communities
    """
    try:
        import infomap
        print(f"Using Infomap (directed, flow-based) with markov_time={markov_time}...")

        # Create Infomap object
        im = infomap.Infomap(f"--directed --markov-time {markov_time} --seed 42")

        # Create node ID mapping (Infomap needs integer IDs starting from 0)
        nodes = list(G.nodes())
        node_to_id = {node: i for i, node in enumerate(nodes)}
        id_to_node = {i: node for node, i in node_to_id.items()}

        # Add edges to Infomap
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            im.add_link(node_to_id[u], node_to_id[v], weight)

        # Run Infomap
        im.run()

        # Extract communities
        comm_dict = defaultdict(set)
        for node_id in im.tree:
            if node_id.is_leaf:
                original_node = id_to_node[node_id.node_id]
                module_id = node_id.module_id
                comm_dict[module_id].add(original_node)

        communities = list(comm_dict.values())
        print(f"Infomap found {len(communities)} communities")
        print(f"Codelength: {im.codelength:.4f} bits")

        return communities

    except ImportError:
        print("Infomap not installed. Install with: pip install infomap")
        print("Falling back to Louvain (undirected)...")
        return detect_communities_louvain_fallback(G)


def detect_communities_louvain_fallback(G):
    """Fallback: Louvain on undirected graph."""
    # Convert to undirected for community detection (symmetrize weights)
    G_undirected = nx.Graph()
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        if G_undirected.has_edge(u, v):
            G_undirected[u][v]['weight'] += weight
        else:
            G_undirected.add_edge(u, v, weight=weight)

    # Try to use community detection
    try:
        print("Using Louvain community detection (undirected fallback)...")
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G_undirected, weight='weight', seed=42)
        return [set(c) for c in communities]
    except ImportError:
        pass

    try:
        import community as community_louvain
        print("Using python-louvain community detection...")
        partition = community_louvain.best_partition(G_undirected, weight='weight', random_state=42)
        comm_dict = defaultdict(set)
        for node, comm_id in partition.items():
            comm_dict[comm_id].add(node)
        return list(comm_dict.values())
    except ImportError:
        pass

    # Fallback: use label propagation
    try:
        print("Using label propagation community detection...")
        from networkx.algorithms.community import label_propagation_communities
        communities = list(label_propagation_communities(G_undirected))
        return [set(c) for c in communities]
    except:
        pass

    # Last resort: connected components
    print("Warning: No community detection algorithm available, using connected components")
    return [set(c) for c in nx.connected_components(G_undirected)]


def compute_dwell_times(sequences, state_to_community):
    """
    Compute how long the agent stays in each community before exiting.
    Returns dict: community_id -> list of dwell times (in steps)
    """
    dwell_times = defaultdict(list)

    for seq in sequences:
        if len(seq) < 2:
            continue

        # Track current community and entry time
        current_comm = state_to_community.get(seq[0])
        entry_time = 0

        for t, state in enumerate(seq[1:], 1):
            comm = state_to_community.get(state)
            if comm != current_comm:
                # Exited the community
                if current_comm is not None:
                    dwell_time = t - entry_time
                    dwell_times[current_comm].append(dwell_time)
                current_comm = comm
                entry_time = t

        # Handle final segment (may not have exited)
        # Don't count incomplete dwells at the end

    return dwell_times


def compute_inter_community_transitions(G, state_to_community, communities):
    """
    Compute transitions between communities.
    Returns dict: (from_comm, to_comm) -> total weight
    """
    inter_transitions = Counter()
    intra_transitions = Counter()

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        comm_u = state_to_community.get(u)
        comm_v = state_to_community.get(v)

        if comm_u is not None and comm_v is not None:
            if comm_u == comm_v:
                intra_transitions[comm_u] += weight
            else:
                inter_transitions[(comm_u, comm_v)] += weight

    return inter_transitions, intra_transitions


def analyze_community(comm_id, members, G, state_stats, dwell_times, inter_trans):
    """Analyze a single community."""
    analysis = {
        'id': comm_id,
        'size': len(members),
        'members': sorted(members),
    }

    # Total visits (sum of visits to all member states)
    total_visits = sum(
        state_stats.get(str(s), {}).get('visit_count', 0)
        for s in members
    )
    analysis['total_visits'] = total_visits

    # Top states by visit count
    state_visits = [
        (s, state_stats.get(str(s), {}).get('visit_count', 0))
        for s in members
    ]
    state_visits.sort(key=lambda x: -x[1])
    analysis['top_states'] = state_visits[:5]

    # Action profile for the community
    total_actions = {'push': 0, 'forward': 0, 'turn_left': 0, 'turn_right': 0}
    for state in members:
        action_counts = get_state_action_profile(state, state_stats)
        for action, count in action_counts.items():
            if action in total_actions:
                total_actions[action] += count

    analysis['action_counts'] = total_actions
    total = sum(total_actions.values())
    if total > 0:
        analysis['action_percentages'] = {
            'push': 100 * total_actions['push'] / total,
            'forward': 100 * total_actions['forward'] / total,
            'turn': 100 * (total_actions['turn_left'] + total_actions['turn_right']) / total
        }

        # Determine dominant behavior
        push_pct = analysis['action_percentages']['push']
        fwd_pct = analysis['action_percentages']['forward']
        turn_pct = analysis['action_percentages']['turn']

        if push_pct > 40:
            analysis['behavior_label'] = 'pushing'
        elif turn_pct > 50:
            analysis['behavior_label'] = 'searching/turning'
        elif fwd_pct > 40:
            analysis['behavior_label'] = 'traversing'
        elif push_pct > 25 and turn_pct > 30:
            analysis['behavior_label'] = 'wall-following'
        else:
            analysis['behavior_label'] = 'mixed'

    # Dwell time statistics
    if comm_id in dwell_times and dwell_times[comm_id]:
        times = dwell_times[comm_id]
        analysis['dwell_stats'] = {
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'count': len(times)
        }

    # Exit edges (transitions to other communities)
    exits = [(to_comm, weight) for (from_comm, to_comm), weight in inter_trans.items()
             if from_comm == comm_id]
    exits.sort(key=lambda x: -x[1])
    analysis['exits'] = exits[:5]

    # Entry edges (transitions from other communities)
    entries = [(from_comm, weight) for (from_comm, to_comm), weight in inter_trans.items()
               if to_comm == comm_id]
    entries.sort(key=lambda x: -x[1])
    analysis['entries'] = entries[:5]

    return analysis


def print_community_analysis(analysis, all_communities):
    """Print analysis for a community."""
    print(f"\n{'='*60}")
    print(f"COMMUNITY {analysis['id']}: {analysis.get('behavior_label', 'unknown').upper()}")
    print(f"{'='*60}")

    print(f"\nSize: {analysis['size']} states")
    print(f"Total visits: {analysis['total_visits']:,}")

    print(f"\nMember states: {analysis['members']}")

    print(f"\nTop states by visits:")
    for state, visits in analysis['top_states']:
        print(f"  State {state}: {visits:,} visits")

    if 'action_percentages' in analysis:
        pcts = analysis['action_percentages']
        print(f"\nAction distribution:")
        print(f"  Push:    {pcts['push']:5.1f}% {'█' * int(pcts['push']/2)}")
        print(f"  Forward: {pcts['forward']:5.1f}% {'█' * int(pcts['forward']/2)}")
        print(f"  Turn:    {pcts['turn']:5.1f}% {'█' * int(pcts['turn']/2)}")

    if 'dwell_stats' in analysis:
        ds = analysis['dwell_stats']
        print(f"\nDwell time (steps before exiting):")
        print(f"  Mean: {ds['mean']:.1f} steps")
        print(f"  Range: {ds['min']} - {ds['max']} steps")
        print(f"  Observations: {ds['count']:,}")

    if analysis['exits']:
        print(f"\nTop exits to other communities:")
        for to_comm, weight in analysis['exits']:
            to_label = all_communities[to_comm].get('behavior_label', '?')
            print(f"  → Community {to_comm} ({to_label}): {weight:,} transitions")

    if analysis['entries']:
        print(f"\nTop entries from other communities:")
        for from_comm, weight in analysis['entries']:
            from_label = all_communities[from_comm].get('behavior_label', '?')
            print(f"  ← Community {from_comm} ({from_label}): {weight:,} transitions")


def create_community_graph_summary(communities_analysis, inter_trans):
    """Create a high-level summary of inter-community transitions."""
    print("\n" + "=" * 70)
    print("HIGH-LEVEL BEHAVIOR GRAPH (Community Transitions)")
    print("=" * 70)

    # Sort inter-community transitions by weight
    sorted_trans = sorted(inter_trans.items(), key=lambda x: -x[1])

    print(f"\nTop inter-community transitions:")
    print(f"{'From':<25} {'To':<25} {'Weight':>12}")
    print("-" * 65)

    for (from_comm, to_comm), weight in sorted_trans[:20]:
        from_label = communities_analysis[from_comm].get('behavior_label', '?')
        to_label = communities_analysis[to_comm].get('behavior_label', '?')
        from_str = f"C{from_comm} ({from_label})"
        to_str = f"C{to_comm} ({to_label})"
        print(f"{from_str:<25} {to_str:<25} {weight:>12,}")

    # Create ASCII visualization of main flows
    print("\n" + "-" * 70)
    print("Community Flow Summary:")
    print("-" * 70)

    # For each community, show its main flow
    for comm_id, analysis in sorted(communities_analysis.items(), key=lambda x: -x[1]['total_visits']):
        label = analysis.get('behavior_label', '?')
        visits = analysis['total_visits']

        # Find main exit
        main_exit = analysis['exits'][0] if analysis['exits'] else None
        main_entry = analysis['entries'][0] if analysis['entries'] else None

        entry_str = ""
        if main_entry:
            entry_label = communities_analysis[main_entry[0]].get('behavior_label', '?')
            entry_str = f"← C{main_entry[0]}({entry_label})"

        exit_str = ""
        if main_exit:
            exit_label = communities_analysis[main_exit[0]].get('behavior_label', '?')
            exit_str = f"→ C{main_exit[0]}({exit_label})"

        print(f"  {entry_str:20} C{comm_id}({label}) {exit_str:20} [{visits:,} visits]")


def analyze_communities(prefix='analysis', markov_time=1.0):
    """Main analysis function.

    Args:
        prefix: Analysis file prefix
        markov_time: Infomap resolution parameter (default 1.0)
                     Higher = more smaller communities
                     Lower = fewer larger communities
    """
    print(f"Loading data from {prefix}...")
    transitions, state_stats, sequences = load_data(prefix)

    print("Building graph...")
    G = build_graph(transitions)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Remove isolated nodes (states never visited)
    active_nodes = [n for n in G.nodes() if state_stats.get(str(n), {}).get('visit_count', 0) > 0]
    G_active = G.subgraph(active_nodes).copy()
    print(f"Active subgraph: {G_active.number_of_nodes()} nodes")

    print("\nDetecting communities (using directed flow-based algorithm)...")
    communities = detect_communities_infomap(G_active, markov_time=markov_time)
    print(f"Found {len(communities)} communities")

    # Create state -> community mapping
    state_to_community = {}
    for comm_id, members in enumerate(communities):
        for state in members:
            state_to_community[state] = comm_id

    # Compute dwell times
    print("Computing dwell times...")
    dwell_times = compute_dwell_times(sequences, state_to_community)

    # Compute inter-community transitions
    print("Computing inter-community transitions...")
    inter_trans, intra_trans = compute_inter_community_transitions(G_active, state_to_community, communities)

    # Analyze each community
    print("Analyzing communities...")
    communities_analysis = {}
    for comm_id, members in enumerate(communities):
        analysis = analyze_community(
            comm_id, members, G_active, state_stats, dwell_times, inter_trans
        )
        communities_analysis[comm_id] = analysis

    # Sort communities by total visits
    sorted_communities = sorted(
        communities_analysis.values(),
        key=lambda x: -x['total_visits']
    )

    # Print results
    print("\n" + "=" * 70)
    print("COMMUNITY DETECTION RESULTS")
    print("=" * 70)

    print(f"\nFound {len(communities)} behavioral communities")
    print(f"\nSummary:")
    print(f"{'ID':<4} {'Label':<20} {'Size':>6} {'Visits':>12} {'Avg Dwell':>10}")
    print("-" * 55)

    for analysis in sorted_communities:
        comm_id = analysis['id']
        label = analysis.get('behavior_label', '?')
        size = analysis['size']
        visits = analysis['total_visits']
        dwell = analysis.get('dwell_stats', {}).get('mean', 0)
        print(f"{comm_id:<4} {label:<20} {size:>6} {visits:>12,} {dwell:>10.1f}")

    # Detailed analysis for each community
    for analysis in sorted_communities:
        print_community_analysis(analysis, communities_analysis)

    # High-level behavior graph
    create_community_graph_summary(communities_analysis, inter_trans)

    # Save results
    results = {
        'algorithm': 'infomap',
        'markov_time': markov_time,
        'num_communities': len(communities),
        'communities': [
            {
                'id': a['id'],
                'behavior_label': a.get('behavior_label', 'unknown'),
                'size': a['size'],
                'members': a['members'],
                'total_visits': a['total_visits'],
                'top_states': a['top_states'],
                'action_percentages': a.get('action_percentages', {}),
                'dwell_stats': a.get('dwell_stats', {}),
                'exits': a['exits'],
                'entries': a['entries']
            }
            for a in sorted_communities
        ],
        'inter_community_transitions': [
            {'from': f, 'to': t, 'weight': w}
            for (f, t), w in sorted(inter_trans.items(), key=lambda x: -x[1])[:50]
        ]
    }

    output_file = f'{prefix}_communities.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")

    return results


if __name__ == '__main__':
    prefix = 'analysis'
    markov_time = 1.0

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--markov-time' and i + 1 < len(sys.argv):
            markov_time = float(sys.argv[i + 1])
            i += 2
        elif arg == '--help':
            print("Usage: python analyze_communities.py [prefix] [options]")
            print("\nOptions:")
            print("  --markov-time T    Resolution parameter (default: 1.0)")
            print("                     Higher = more smaller communities")
            print("                     Lower = fewer larger communities")
            print("\nExamples:")
            print("  python analyze_communities.py analysis")
            print("  python analyze_communities.py analysis --markov-time 0.5  # fewer, larger communities")
            print("  python analyze_communities.py analysis --markov-time 2.0  # more, smaller communities")
            sys.exit(0)
        elif not arg.startswith('-'):
            prefix = arg
            i += 1
        else:
            i += 1

    analyze_communities(prefix, markov_time=markov_time)
