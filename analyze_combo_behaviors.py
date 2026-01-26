#!/usr/bin/env python3
"""
Analyze combination-based behavior patterns from the combination graph.

This script extracts behavioral patterns using what the agent actually perceives
(combinations of boxes/walls around it) rather than abstract FSM states.

Features:
1. Frequent combination paths (behavioral sequences)
2. Behavioral loops (cycles in perception-action space)
3. Action patterns associated with perceptions
4. Entry/exit combinations for behaviors
5. Combination motifs and transitions

Usage:
    python analyze_combo_behaviors.py [prefix] [options]

Options:
    --min-weight N       Minimum edge weight for pattern mining (default: 100)
    --max-path-length N  Maximum path length to explore (default: 10)
    --combo COMBO_ID     Query details about a specific combination

Examples:
    python analyze_combo_behaviors.py analysis
    python analyze_combo_behaviors.py analysis --min-weight 200
    python analyze_combo_behaviors.py analysis --combo 45
"""

import json
import sys
from collections import defaultdict, Counter, deque

# Combination decoding (from analyze_agent.py)
IIDX = [0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546]


def decode_combination(combo_raw):
    """Decode a raw combination value to 8 cell values (0=empty, 1=box, 2=wall)."""
    cells = []
    for i in range(8):
        cells.append(combo_raw % 3)
        combo_raw //= 3
    return cells


def describe_combination(combo_idx):
    """Get human-readable description of what the agent sees."""
    if combo_idx >= len(IIDX):
        return "invalid"
    combo_raw = IIDX[combo_idx]
    cells = decode_combination(combo_raw)
    # Physical direction names
    phys_names = ['L', 'BL', 'B', 'BR', 'R', 'FR', 'F', 'FL']
    val_names = ['_', 'B', 'W']  # empty, box, wall
    desc = ''.join(f"{phys_names[i]}:{val_names[cells[i]]} " for i in range(8) if cells[i] != 0)
    return desc.strip() if desc.strip() else "all_empty"


def get_combo_features(combo_idx):
    """Extract features from a combination for classification."""
    if combo_idx >= len(IIDX):
        return {}

    combo_raw = IIDX[combo_idx]
    cells = decode_combination(combo_raw)

    # Physical positions: L, BL, B, BR, R, FR, F, FL
    # Front is position 6, Left is 0, Right is 4, Back is 2
    return {
        'box_front': cells[6] == 1,
        'box_left': cells[0] == 1,
        'box_right': cells[4] == 1,
        'box_back': cells[2] == 1,
        'wall_front': cells[6] == 2,
        'wall_left': cells[0] == 2,
        'wall_right': cells[4] == 2,
        'wall_back': cells[2] == 2,
        'num_boxes': sum(1 for c in cells if c == 1),
        'num_walls': sum(1 for c in cells if c == 2),
        'total_obstacles': sum(1 for c in cells if c != 0)
    }


def classify_combo_action_pattern(combo_stats):
    """Classify the behavioral pattern based on actions taken in this combo."""
    actions = combo_stats['action_counts']
    total = sum(actions.values())
    if total == 0:
        return 'unvisited'

    push = actions.get('push', 0)
    forward = actions.get('forward', 0)
    turn_left = actions.get('turn_left', 0)
    turn_right = actions.get('turn_right', 0)
    turn = turn_left + turn_right

    # Calculate percentages
    push_pct = 100 * push / total
    forward_pct = 100 * forward / total
    turn_pct = 100 * turn / total

    # Classify based on dominant action
    if push_pct > 60:
        return 'push-dominant'
    elif forward_pct > 70:
        return 'forward-dominant'
    elif turn_pct > 60:
        if turn_left > turn_right * 1.5:
            return 'turn-left-dominant'
        elif turn_right > turn_left * 1.5:
            return 'turn-right-dominant'
        else:
            return 'turn-mixed'
    elif push_pct > 30 and forward_pct > 30:
        return 'push-forward-mix'
    elif push_pct > 20 and turn_pct > 30:
        return 'push-turn-mix'
    else:
        return 'balanced-actions'


def load_data(prefix='analysis'):
    """Load combination graph and statistics."""
    with open(f'{prefix}_combo_transitions.json', 'r') as f:
        graph_data = json.load(f)

    with open(f'{prefix}_combo_stats.json', 'r') as f:
        combo_stats = json.load(f)

    return graph_data, combo_stats


def build_adjacency_lists(graph_data):
    """Build forward and reverse adjacency lists from edge data."""
    forward = defaultdict(list)  # combo -> [(target, weight), ...]
    reverse = defaultdict(list)  # combo -> [(source, weight), ...]

    for edge in graph_data['edges']:
        src = edge['source']
        tgt = edge['target']
        weight = edge['weight']
        forward[src].append((tgt, weight))
        reverse[tgt].append((src, weight))

    return forward, reverse


def find_frequent_paths(forward_adj, min_weight, max_length):
    """
    Find frequent paths through the combination graph using DFS.
    Returns paths as lists of combo IDs with their total weight.
    """
    paths = []

    def dfs(path, min_edge_weight):
        """DFS to explore paths maintaining minimum edge weight."""
        if len(path) >= max_length:
            return

        current = path[-1]
        if current not in forward_adj:
            return

        for next_combo, weight in forward_adj[current]:
            if weight >= min_weight and next_combo not in path:  # Avoid cycles in path
                new_path = path + [next_combo]
                new_min_weight = min(min_edge_weight, weight)
                paths.append((new_path, new_min_weight))
                dfs(new_path, new_min_weight)

    # Start DFS from all nodes with outgoing edges
    print(f"  Exploring paths with min edge weight {min_weight}...")
    for start_combo in forward_adj.keys():
        dfs([start_combo], float('inf'))

    return paths


def find_cycles(forward_adj, min_weight, max_length=10, allow_repeats=True):
    """
    Find cycles (loops) in the combination graph.

    If allow_repeats=True, combinations can appear multiple times in a path,
    which captures behaviors like repeated pushing in the same perception.
    """
    cycles = []
    seen_cycles = set()  # To avoid duplicate cycles

    def dfs_cycle(path, weights):
        """DFS to find cycles, allowing repeated nodes."""
        if len(path) > max_length:
            return

        current = path[-1]
        if current not in forward_adj:
            return

        for next_combo, weight in forward_adj[current]:
            if weight < min_weight:
                continue

            # Found a cycle back to start
            if next_combo == path[0] and len(path) >= 2:
                cycle_tuple = tuple(path)
                if cycle_tuple not in seen_cycles:
                    cycles.append((path, min(weights + [weight])))
                    seen_cycles.add(cycle_tuple)
            # Continue exploring
            elif allow_repeats:
                # Allow repeats but limit to prevent infinite loops
                # Only revisit if we haven't been there too many times
                if path.count(next_combo) < 5:  # Max 5 repetitions of same combo
                    dfs_cycle(path + [next_combo], weights + [weight])
            else:
                # Original behavior: no repeats except to close cycle
                if next_combo not in path[1:]:
                    dfs_cycle(path + [next_combo], weights + [weight])

    print(f"  Finding cycles with min edge weight {min_weight}...")
    if allow_repeats:
        print(f"  Allowing repeated combinations in paths (up to 5 repetitions)")

    # Start from all nodes
    for start_combo in forward_adj.keys():
        dfs_cycle([start_combo], [])

    return cycles


def find_behavioral_motifs(forward_adj, combo_stats, min_weight):
    """
    Find common behavioral motifs: perception -> action -> perception patterns.
    """
    motifs = []  # (combo1, combo2, action_pattern, weight)

    for src in forward_adj.keys():
        src_stats = combo_stats.get(str(src), {})
        src_pattern = classify_combo_action_pattern(src_stats)

        for tgt, weight in forward_adj[src]:
            if weight >= min_weight:
                tgt_stats = combo_stats.get(str(tgt), {})
                tgt_pattern = classify_combo_action_pattern(tgt_stats)

                motifs.append({
                    'from_combo': src,
                    'to_combo': tgt,
                    'from_pattern': src_pattern,
                    'to_pattern': tgt_pattern,
                    'weight': weight,
                    'from_desc': describe_combination(src),
                    'to_desc': describe_combination(tgt)
                })

    return motifs


def analyze_combo_statistics(combo_stats):
    """Analyze overall statistics about combinations."""
    # Action patterns distribution
    pattern_counts = Counter()

    # Perception features distribution
    feature_counts = {
        'box_front': 0,
        'wall_front': 0,
        'box_left': 0,
        'box_right': 0,
        'surrounded': 0  # many obstacles
    }

    visited_combos = []

    for combo_id, stats in combo_stats.items():
        combo_idx = int(combo_id)
        if stats['visit_count'] > 0:
            visited_combos.append((combo_idx, stats))
            pattern = classify_combo_action_pattern(stats)
            pattern_counts[pattern] += 1

            features = get_combo_features(combo_idx)
            if features['box_front']:
                feature_counts['box_front'] += 1
            if features['wall_front']:
                feature_counts['wall_front'] += 1
            if features['box_left']:
                feature_counts['box_left'] += 1
            if features['box_right']:
                feature_counts['box_right'] += 1
            if features['total_obstacles'] >= 5:
                feature_counts['surrounded'] += 1

    return pattern_counts, feature_counts, visited_combos


def analyze_combo_behaviors(prefix='analysis', min_weight=100, max_path_length=10):
    """Main analysis function."""
    print(f"Loading combination graph data from {prefix}...")
    graph_data, combo_stats = load_data(prefix)

    print(f"Total combinations: {len(combo_stats)}")
    print(f"Graph edges: {len(graph_data['edges'])}")

    # Build adjacency lists
    forward_adj, reverse_adj = build_adjacency_lists(graph_data)
    print(f"Combinations with outgoing edges: {len(forward_adj)}")
    print(f"Combinations with incoming edges: {len(reverse_adj)}")

    # Overall statistics
    print("\n" + "=" * 70)
    print("COMBINATION STATISTICS")
    print("=" * 70)

    pattern_counts, feature_counts, visited_combos = analyze_combo_statistics(combo_stats)

    print("\nAction pattern distribution:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern:<25} {count:>6} combinations")

    print("\nPerception feature distribution:")
    print(f"  Box in front:           {feature_counts['box_front']} combinations")
    print(f"  Wall in front:          {feature_counts['wall_front']} combinations")
    print(f"  Box on left:            {feature_counts['box_left']} combinations")
    print(f"  Box on right:           {feature_counts['box_right']} combinations")
    print(f"  Heavily surrounded:     {feature_counts['surrounded']} combinations")

    # Most visited combinations
    print("\n" + "=" * 70)
    print("MOST VISITED COMBINATIONS")
    print("=" * 70)

    visited_combos.sort(key=lambda x: x[1]['visit_count'], reverse=True)

    print(f"\n{'Combo':<6} {'Visits':>10} {'Pattern':<20} {'Description':<50}")
    print("-" * 90)
    for combo_idx, stats in visited_combos[:20]:
        pattern = classify_combo_action_pattern(stats)
        desc = describe_combination(combo_idx)
        if len(desc) > 48:
            desc = desc[:45] + "..."
        print(f"{combo_idx:<6} {stats['visit_count']:>10,} {pattern:<20} {desc:<50}")

    # Strongest transitions
    print("\n" + "=" * 70)
    print("STRONGEST TRANSITIONS (behavioral sequences)")
    print("=" * 70)

    edges_sorted = sorted(graph_data['edges'], key=lambda x: x['weight'], reverse=True)

    print(f"\n{'From':<6} {'To':<6} {'Weight':>10} {'From Desc':<35} {'To Desc':<35}")
    print("-" * 100)
    for edge in edges_sorted[:30]:
        src = edge['source']
        tgt = edge['target']
        weight = edge['weight']
        src_desc = describe_combination(src)
        tgt_desc = describe_combination(tgt)
        if len(src_desc) > 33:
            src_desc = src_desc[:30] + "..."
        if len(tgt_desc) > 33:
            tgt_desc = tgt_desc[:30] + "..."
        print(f"{src:<6} {tgt:<6} {weight:>10,} {src_desc:<35} {tgt_desc:<35}")

    # Find behavioral loops (cycles)
    print("\n" + "=" * 70)
    print("BEHAVIORAL LOOPS (cycles in perception space)")
    print("=" * 70)

    cycles = find_cycles(forward_adj, min_weight=min_weight, max_length=8)
    cycles.sort(key=lambda x: x[1], reverse=True)  # Sort by minimum edge weight

    print(f"\nFound {len(cycles)} cycles with min edge weight >= {min_weight}")
    print(f"\nShowing top 20 cycles:\n")
    print(f"{'Cycle':<50} {'Length':<8} {'Min Weight':<12}")
    print("-" * 75)

    for cycle, min_edge_weight in cycles[:20]:
        cycle_str = '-'.join(map(str, cycle)) + f"-{cycle[0]}"
        if len(cycle_str) > 48:
            cycle_str = cycle_str[:45] + "..."
        print(f"{cycle_str:<50} {len(cycle):<8} {min_edge_weight:<12,}")

    # Show details of top cycles
    if cycles:
        print("\n" + "=" * 70)
        print("TOP CYCLE DETAILS")
        print("=" * 70)

        for i, (cycle, min_edge_weight) in enumerate(cycles[:5]):
            print(f"\nCycle {i+1}: {'-'.join(map(str, cycle))}-{cycle[0]}")
            print(f"Length: {len(cycle)}, Min edge weight: {min_edge_weight:,}")
            print("\nPerceptions in cycle:")
            for combo in cycle:
                stats = combo_stats.get(str(combo), {})
                pattern = classify_combo_action_pattern(stats)
                desc = describe_combination(combo)
                print(f"  {combo}: {pattern:<20} {desc}")

    # Find common behavioral motifs
    print("\n" + "=" * 70)
    print("BEHAVIORAL MOTIFS (perception-action patterns)")
    print("=" * 70)

    motifs = find_behavioral_motifs(forward_adj, combo_stats, min_weight)
    motifs.sort(key=lambda x: x['weight'], reverse=True)

    # Group by pattern pairs
    pattern_pairs = Counter()
    for motif in motifs:
        key = (motif['from_pattern'], motif['to_pattern'])
        pattern_pairs[key] += motif['weight']

    print("\nMost common pattern transitions:")
    print(f"{'From Pattern':<25} {'To Pattern':<25} {'Total Weight':<15}")
    print("-" * 70)
    for (from_pat, to_pat), total_weight in pattern_pairs.most_common(20):
        print(f"{from_pat:<25} {to_pat:<25} {total_weight:<15,}")

    print("\n\nTop individual motifs:")
    print(f"{'From':<6} {'To':<6} {'Weight':>10} {'From Pattern':<20} {'To Pattern':<20}")
    print("-" * 70)
    for motif in motifs[:30]:
        print(f"{motif['from_combo']:<6} {motif['to_combo']:<6} {motif['weight']:>10,} " +
              f"{motif['from_pattern']:<20} {motif['to_pattern']:<20}")

    # Entry and exit combinations
    print("\n" + "=" * 70)
    print("ENTRY AND EXIT COMBINATIONS")
    print("=" * 70)

    # Entry: combinations with many incoming edges
    entry_scores = {}
    for combo in reverse_adj.keys():
        total_incoming = sum(w for _, w in reverse_adj[combo])
        entry_scores[combo] = total_incoming

    # Exit: combinations with many outgoing edges
    exit_scores = {}
    for combo in forward_adj.keys():
        total_outgoing = sum(w for _, w in forward_adj[combo])
        exit_scores[combo] = total_outgoing

    print("\nTop entry combinations (many incoming transitions):")
    print(f"{'Combo':<6} {'In-Weight':>12} {'Pattern':<20} {'Description':<40}")
    print("-" * 85)
    for combo, weight in sorted(entry_scores.items(), key=lambda x: -x[1])[:15]:
        stats = combo_stats.get(str(combo), {})
        pattern = classify_combo_action_pattern(stats)
        desc = describe_combination(combo)
        if len(desc) > 38:
            desc = desc[:35] + "..."
        print(f"{combo:<6} {weight:>12,} {pattern:<20} {desc:<40}")

    print("\nTop exit combinations (many outgoing transitions):")
    print(f"{'Combo':<6} {'Out-Weight':>12} {'Pattern':<20} {'Description':<40}")
    print("-" * 85)
    for combo, weight in sorted(exit_scores.items(), key=lambda x: -x[1])[:15]:
        stats = combo_stats.get(str(combo), {})
        pattern = classify_combo_action_pattern(stats)
        desc = describe_combination(combo)
        if len(desc) > 38:
            desc = desc[:35] + "..."
        print(f"{combo:<6} {weight:>12,} {pattern:<20} {desc:<40}")

    # Save results
    results = {
        'combination_patterns': {
            str(combo_idx): {
                'visit_count': stats['visit_count'],
                'action_pattern': classify_combo_action_pattern(stats),
                'description': describe_combination(combo_idx),
                'features': get_combo_features(combo_idx),
                'action_counts': stats['action_counts']
            }
            for combo_idx, stats in visited_combos[:100]
        },
        'behavioral_loops': [
            {
                'cycle': list(cycle),
                'length': len(cycle),
                'min_edge_weight': min_edge_weight,
                'perceptions': [describe_combination(c) for c in cycle]
            }
            for cycle, min_edge_weight in cycles[:50]
        ],
        'top_transitions': [
            {
                'from': edge['source'],
                'to': edge['target'],
                'weight': edge['weight'],
                'from_desc': describe_combination(edge['source']),
                'to_desc': describe_combination(edge['target'])
            }
            for edge in edges_sorted[:100]
        ],
        'behavioral_motifs': [
            motif for motif in motifs[:100]
        ],
        'entry_combinations': [
            {
                'combo': combo,
                'incoming_weight': weight,
                'description': describe_combination(combo)
            }
            for combo, weight in sorted(entry_scores.items(), key=lambda x: -x[1])[:30]
        ],
        'exit_combinations': [
            {
                'combo': combo,
                'outgoing_weight': weight,
                'description': describe_combination(combo)
            }
            for combo, weight in sorted(exit_scores.items(), key=lambda x: -x[1])[:30]
        ]
    }

    output_file = f'{prefix}_combo_behaviors.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved results to {output_file}")

    return results


def query_combination(combo_id, prefix='analysis'):
    """Query details about a specific combination."""
    graph_data, combo_stats = load_data(prefix)

    combo_id = int(combo_id)

    print(f"\n{'='*70}")
    print(f"COMBINATION ANALYSIS: {combo_id}")
    print(f"{'='*70}")

    # Basic info
    stats = combo_stats.get(str(combo_id), {})
    if not stats:
        print(f"Combination {combo_id} not found in stats")
        return

    print(f"\nDescription: {describe_combination(combo_id)}")
    print(f"Visit count: {stats['visit_count']:,}")
    print(f"Action pattern: {classify_combo_action_pattern(stats)}")

    # Features
    features = get_combo_features(combo_id)
    print(f"\nPerception features:")
    print(f"  Box in front: {features['box_front']}")
    print(f"  Wall in front: {features['wall_front']}")
    print(f"  Box on left: {features['box_left']}")
    print(f"  Box on right: {features['box_right']}")
    print(f"  Total obstacles: {features['total_obstacles']}")

    # Action breakdown
    actions = stats['action_counts']
    total_actions = sum(actions.values())
    if total_actions > 0:
        print(f"\nAction distribution:")
        for action, count in actions.items():
            pct = 100 * count / total_actions
            print(f"  {action:<12} {count:>8,} ({pct:>5.1f}%)")

    # Build adjacency lists
    forward_adj, reverse_adj = build_adjacency_lists(graph_data)

    # Incoming transitions
    if combo_id in reverse_adj:
        print(f"\nIncoming transitions (from other combinations):")
        incoming = sorted(reverse_adj[combo_id], key=lambda x: -x[1])
        print(f"  {'From':<6} {'Weight':>10} {'Description':<50}")
        print(f"  {'-'*70}")
        for src, weight in incoming[:10]:
            desc = describe_combination(src)
            if len(desc) > 48:
                desc = desc[:45] + "..."
            print(f"  {src:<6} {weight:>10,} {desc:<50}")

    # Outgoing transitions
    if combo_id in forward_adj:
        print(f"\nOutgoing transitions (to other combinations):")
        outgoing = sorted(forward_adj[combo_id], key=lambda x: -x[1])
        print(f"  {'To':<6} {'Weight':>10} {'Description':<50}")
        print(f"  {'-'*70}")
        for tgt, weight in outgoing[:10]:
            desc = describe_combination(tgt)
            if len(desc) > 48:
                desc = desc[:45] + "..."
            print(f"  {tgt:<6} {weight:>10,} {desc:<50}")

    # Check if part of cycles
    print(f"\nCycles containing this combination:")
    cycles = find_cycles(forward_adj, min_weight=50, max_length=8)
    relevant_cycles = [c for c, w in cycles if combo_id in c]
    if relevant_cycles:
        for i, cycle in enumerate(relevant_cycles[:5]):
            cycle_str = '-'.join(map(str, cycle)) + f"-{cycle[0]}"
            print(f"  {i+1}. {cycle_str}")
    else:
        print("  None found")


if __name__ == '__main__':
    prefix = 'analysis'
    combo_query = None
    min_weight = 100
    max_path_length = 10

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--combo' and i + 1 < len(sys.argv):
            combo_query = sys.argv[i + 1]
            i += 2
        elif arg == '--min-weight' and i + 1 < len(sys.argv):
            min_weight = int(sys.argv[i + 1])
            i += 2
        elif arg == '--max-path-length' and i + 1 < len(sys.argv):
            max_path_length = int(sys.argv[i + 1])
            i += 2
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('-'):
            prefix = arg
            i += 1
        else:
            i += 1

    if combo_query:
        query_combination(combo_query, prefix)
    else:
        analyze_combo_behaviors(prefix, min_weight, max_path_length)
