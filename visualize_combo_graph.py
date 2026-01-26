#!/usr/bin/env python3
"""
Visualize the combination transition graph from agent analysis.

Usage:
    python visualize_combo_graph.py [analysis_prefix] [--top-n N]
    python visualize_combo_graph.py [analysis_prefix] --pattern COMBO1-COMBO2-...

Options:
    analysis_prefix: Prefix used when running analyze_agent.py (default: 'analysis')
    --top-n N: Number of top combinations to visualize (default: 15)
    --pattern: Visualize a specific sequence of combinations (e.g., --pattern 45-67-89)

Examples:
    python visualize_combo_graph.py                           # Top 15 graph
    python visualize_combo_graph.py --top-n 20                # Top 20 graph
    python visualize_combo_graph.py --pattern 45-67-12-89     # Combo sequence
"""

import json
import sys
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Valid combination indices (maps combo_idx back to raw 8-cell encoding)
IIDX = [0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546]

# 3x3 grid mapping: position index -> (row, col) in 3x3 grid
# Scan order: 0=F, 1=FR, 2=R, 3=BR, 4=B, 5=BL, 6=L, 7=FL
# Grid layout (agent facing up):
#   [FL][F ][FR]     [7][0][1]
#   [L ][A ][R ]  =  [6][X][2]
#   [BL][B ][BR]     [5][4][3]
GRID_POS = {
    0: (0, 1),  # Front -> top middle
    1: (0, 2),  # Front-Right -> top right
    2: (1, 2),  # Right -> middle right
    3: (2, 2),  # Back-Right -> bottom right
    4: (2, 1),  # Back -> bottom middle
    5: (2, 0),  # Back-Left -> bottom left
    6: (1, 0),  # Left -> middle left
    7: (0, 0),  # Front-Left -> top left
}


def decode_combination(combo_idx):
    """Decode a combination index to 8 cell values (0=empty, 1=box, 2=wall)."""
    if combo_idx >= len(IIDX):
        return [0] * 8
    combo_raw = IIDX[combo_idx]
    cells = []
    for _ in range(8):
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


def classify_combo_action_pattern(combo_stats):
    """Classify the behavioral pattern based on actions taken in this combo."""
    actions = combo_stats.get('action_counts', {})
    total = sum(actions.values())
    if total == 0:
        return 'none'

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
        return 'push'
    elif forward_pct > 70:
        return 'forward'
    elif turn_pct > 60:
        return 'turn'
    elif push_pct >= forward_pct and push_pct >= turn_pct:
        return 'push'
    elif forward_pct >= turn_pct:
        return 'forward'
    else:
        return 'turn'


def plot_combo_sequence(combo_ids, combo_stats, output_file='combo_sequence.png', title=None):
    """
    Visualize a specific sequence of combinations (a behavioral pattern).
    Shows each combination in order as a 3x3 grid with the agent's perception.

    combo_ids: list of combination IDs in the pattern, e.g., [45, 67, 12, 89]
    """
    n_combos = len(combo_ids)

    # Create figure - horizontal layout for sequence
    fig, axes = plt.subplots(1, n_combos, figsize=(2.8 * n_combos, 4))
    if n_combos == 1:
        axes = [axes]

    # Colors for cells
    COLOR_EMPTY = np.array([1.0, 1.0, 1.0])      # White
    COLOR_BOX = np.array([0.9, 0.4, 0.1])        # Orange
    COLOR_WALL = np.array([0.2, 0.2, 0.2])       # Dark gray/black

    for idx, combo_id in enumerate(combo_ids):
        ax = axes[idx]

        # Decode combination to get exact cell values
        cells = decode_combination(combo_id)

        # Create 3x3 image
        img = np.zeros((3, 3, 3))

        # Fill in the 8 surrounding cells
        for pos in range(8):
            row, col = GRID_POS[pos]
            cell_type = cells[pos]

            if cell_type == 0:
                img[row, col] = COLOR_EMPTY
            elif cell_type == 1:
                img[row, col] = COLOR_BOX
            elif cell_type == 2:
                img[row, col] = COLOR_WALL

        # Center cell is agent (white background)
        img[1, 1] = COLOR_EMPTY

        # Plot
        ax.imshow(img, interpolation='nearest')

        # Add grid lines
        for i in range(4):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        # Arrow in center cell pointing up (agent facing direction)
        ax.annotate('', xy=(1, 0.7), xytext=(1, 1.3),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

        # Get stats for this combination
        stats = combo_stats.get(str(combo_id), {})
        visit_count = stats.get('visit_count', 0)
        action_pattern = classify_combo_action_pattern(stats)

        # Title colors based on action
        action_colors = {
            'push': '#2ecc71',     # Green
            'forward': '#3498db',  # Blue
            'turn': '#f1c40f',     # Yellow
            'none': '#95a5a6'      # Gray
        }
        title_color = action_colors.get(action_pattern, '#95a5a6')

        # Title with step number, combo ID, and action
        ax.set_title(f'Step {idx + 1}: Combo {combo_id}\n{action_pattern.upper()}',
                    fontsize=10, fontweight='bold', color=title_color)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_EMPTY, edgecolor='black', label='Empty'),
        mpatches.Patch(facecolor=COLOR_BOX, edgecolor='black', label='Box'),
        mpatches.Patch(facecolor=COLOR_WALL, edgecolor='black', label='Wall'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    # Title
    combo_str = '-'.join(map(str, combo_ids))
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'Combination Sequence: {combo_str}\n(Arrow = agent facing up)',
                    fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combination sequence visualization to {output_file}")


def load_combo_graph_data(prefix='analysis'):
    """Load combination transition graph and statistics."""
    with open(f'{prefix}_combo_transitions.json', 'r') as f:
        graph_data = json.load(f)

    with open(f'{prefix}_combo_stats.json', 'r') as f:
        combo_stats = json.load(f)

    return graph_data, combo_stats


def create_networkx_graph(graph_data, combo_stats):
    """Create a NetworkX directed graph from the combination data."""
    G = nx.DiGraph()

    # Add nodes with attributes
    for combo_id, stats in combo_stats.items():
        combo_idx = int(combo_id)
        visit_count = stats.get('visit_count', 0)

        if visit_count == 0:
            continue  # Skip unvisited combinations

        action_counts = stats.get('action_counts', {})
        push_count = action_counts.get('push', 0)
        forward_count = action_counts.get('forward', 0)
        turn_left_count = action_counts.get('turn_left', 0)
        turn_right_count = action_counts.get('turn_right', 0)
        turn_count = turn_left_count + turn_right_count

        dominant_action = classify_combo_action_pattern(stats)

        G.add_node(combo_idx,
                   visit_count=visit_count,
                   push_count=push_count,
                   forward_count=forward_count,
                   turn_count=turn_count,
                   dominant_action=dominant_action,
                   description=describe_combination(combo_idx))

    # Add edges with weights
    for edge in graph_data['edges']:
        src = edge['source']
        tgt = edge['target']
        weight = edge['weight']
        # Only add edge if both nodes exist
        if src in G.nodes() and tgt in G.nodes():
            G.add_edge(src, tgt, weight=weight)

    return G


def separate_overlapping_nodes(pos, min_distance=0.8):
    """Iteratively push apart nodes that are too close together."""
    nodes = list(pos.keys())
    max_iterations = 100

    for iteration in range(max_iterations):
        moved = False
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                x1, y1 = pos[n1]
                x2, y2 = pos[n2]

                dx = x2 - x1
                dy = y2 - y1
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < min_distance and dist > 0:
                    # Push apart
                    overlap = min_distance - dist
                    push = overlap / 2 + 0.1

                    # Normalize direction
                    dx /= dist
                    dy /= dist

                    pos[n1] = (x1 - dx * push, y1 - dy * push)
                    pos[n2] = (x2 + dx * push, y2 + dy * push)
                    moved = True
                elif dist == 0:
                    # Nodes at exact same position, push in random direction
                    angle = hash(str(n1) + str(n2)) % 360 * math.pi / 180
                    pos[n1] = (x1 - math.cos(angle) * min_distance/2, y1 - math.sin(angle) * min_distance/2)
                    pos[n2] = (x2 + math.cos(angle) * min_distance/2, y2 + math.sin(angle) * min_distance/2)
                    moved = True

        if not moved:
            break

    return pos


def radial_layout(G, center_metric='visits', min_node_distance=1.0):
    """Create a radial layout with high-centrality nodes in the center."""
    if center_metric == 'visits':
        centrality = {n: G.nodes[n].get('visit_count', 0) for n in G.nodes()}
    else:
        centrality = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}

    # Sort nodes by centrality (highest first)
    sorted_nodes = sorted(G.nodes(), key=lambda n: centrality[n], reverse=True)

    n_nodes = len(sorted_nodes)
    pos = {}

    # Calculate ring assignments with proper spacing
    rings = []
    nodes_placed = 0
    ring_idx = 0
    base_radius = 2.0

    while nodes_placed < n_nodes:
        if ring_idx == 0:
            # Center: just 1 node
            ring_size = 1
            radius = 0
        else:
            radius = base_radius + (ring_idx - 1) * min_node_distance * 1.5
            # How many nodes can fit on this ring?
            circumference = 2 * math.pi * radius
            ring_size = max(1, int(circumference / min_node_distance))
            ring_size = min(ring_size, n_nodes - nodes_placed)

        rings.append((sorted_nodes[nodes_placed:nodes_placed + ring_size], radius))
        nodes_placed += ring_size
        ring_idx += 1

    # Place nodes in concentric circles
    for ring_nodes, radius in rings:
        n_in_ring = len(ring_nodes)
        for i, node in enumerate(ring_nodes):
            if radius == 0:
                pos[node] = (0, 0)
            else:
                angle = 2 * math.pi * i / n_in_ring - math.pi / 2
                pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

    # Final overlap check and separation
    pos = separate_overlapping_nodes(pos, min_node_distance)

    return pos, centrality


def plot_top_n_combos(G, output_file='combo_graph_top15.png', top_n=15):
    """
    Create a clean graph showing only the top N most visited combinations
    with all transitions between them.
    """
    fig, ax = plt.subplots(figsize=(16, 16))

    # Get top N combinations by visit count
    all_nodes = [(n, G.nodes[n].get('visit_count', 0)) for n in G.nodes()]
    all_nodes.sort(key=lambda x: -x[1])
    top_nodes = [n for n, v in all_nodes[:top_n]]

    # Create subgraph with only these nodes
    H = G.subgraph(top_nodes).copy()

    print(f"Top {top_n} graph: {len(H.nodes())} combinations and {len(H.edges())} transitions")

    # Use radial layout - most visited in center
    pos, centrality = radial_layout(H, center_metric='visits', min_node_distance=2.0)

    # Node sizes based on visit count
    visit_counts = [H.nodes[n].get('visit_count', 1) for n in H.nodes()]
    max_visits = max(visit_counts) if visit_counts else 1
    node_sizes = [500 + 4000 * (v / max_visits) for v in visit_counts]

    # Create a dict mapping node to its size for edge shrinking
    node_size_dict = {n: sz for n, sz in zip(H.nodes(), node_sizes)}

    # Node colors based on dominant action
    action_colors = {
        'push': '#2ecc71',     # Green
        'forward': '#3498db',  # Blue
        'turn': '#f1c40f',     # Yellow
        'none': '#95a5a6'      # Gray
    }
    node_colors = [action_colors.get(H.nodes[n].get('dominant_action', 'none'), '#95a5a6')
                   for n in H.nodes()]

    # Edge widths based on weight
    edge_weights = [H.edges[e].get('weight', 1) for e in H.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        edge_widths = [0.5 + 4 * (w / max_weight) for w in edge_weights]
        edge_alphas = [0.4 + 0.4 * (w / max_weight) for w in edge_weights]
    else:
        edge_widths = []
        edge_alphas = []

    # Draw edges - black arrows that stop at node borders
    for (u, v), width, alpha in zip(H.edges(), edge_widths, edge_alphas):
        # Calculate shrink values: node_size is area in points², radius = sqrt(size/pi)
        radius_a = math.sqrt(node_size_dict[u] / math.pi)
        radius_b = math.sqrt(node_size_dict[v] / math.pi)

        ax.annotate("",
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="-|>",
                                    color='black',
                                    alpha=alpha,
                                    connectionstyle="arc3,rad=0.15",
                                    lw=width,
                                    shrinkA=radius_a,
                                    shrinkB=radius_b,
                                    mutation_scale=15))

    # Draw nodes
    nx.draw_networkx_nodes(H, pos, ax=ax,
                           node_size=node_sizes,
                           node_color=node_colors,
                           alpha=0.9,
                           edgecolors='black',
                           linewidths=2)

    # Draw labels
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=11, font_weight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Push'),
        mpatches.Patch(color='#3498db', label='Forward'),
        mpatches.Patch(color='#f1c40f', label='Turn'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)

    ax.set_title(f'Top {top_n} Most Visited Combinations\n'
                f'Node color = dominant action, Node size = visit count',
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')

    # Set axis limits
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    margin = 3
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved top {top_n} combination graph to {output_file}")


def print_graph_stats(G):
    """Print useful statistics about the combination graph."""
    print("\n" + "="*60)
    print("COMBINATION GRAPH STATISTICS")
    print("="*60)

    print(f"\nCombinations (nodes): {G.number_of_nodes()}")
    print(f"Transitions (edges): {G.number_of_edges()}")

    visit_counts = [(n, G.nodes[n].get('visit_count', 0)) for n in G.nodes()]
    visit_counts.sort(key=lambda x: -x[1])

    print(f"\nTop 15 most visited combinations:")
    print(f"{'Combo':<6} {'Visits':>12} {'Action':<10} {'Description':<50}")
    print("-" * 85)
    for node, visits in visit_counts[:15]:
        action = G.nodes[node].get('dominant_action', 'none')
        desc = G.nodes[node].get('description', '')
        if len(desc) > 48:
            desc = desc[:45] + "..."
        print(f"{node:<6} {visits:>12,} {action:<10} {desc:<50}")

    # Action distribution
    print(f"\nCombinations by dominant action:")
    for action in ['push', 'forward', 'turn', 'none']:
        nodes = [n for n in G.nodes() if G.nodes[n].get('dominant_action') == action]
        if nodes:
            total_visits = sum(G.nodes[n].get('visit_count', 0) for n in nodes)
            print(f"  {action.capitalize():8s}: {len(nodes):3d} combinations, {total_visits:>10,} total visits")

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    print(f"\nTop 10 by in-degree (many incoming transitions):")
    top_in = sorted(in_degrees.items(), key=lambda x: -x[1])[:10]
    for node, deg in top_in:
        desc = G.nodes[node].get('description', '')[:40]
        print(f"  Combo {node:3d}: {deg} incoming - {desc}")

    print(f"\nTop 10 by out-degree (many outgoing transitions):")
    top_out = sorted(out_degrees.items(), key=lambda x: -x[1])[:10]
    for node, deg in top_out:
        desc = G.nodes[node].get('description', '')[:40]
        print(f"  Combo {node:3d}: {deg} outgoing - {desc}")

    print(f"\nTop 10 most frequent transitions:")
    edges_by_weight = sorted(G.edges(data=True), key=lambda x: -x[2].get('weight', 0))[:10]
    for u, v, data in edges_by_weight:
        print(f"  {u:3d} -> {v:3d}: {data.get('weight', 0):>10,}")


if __name__ == '__main__':
    prefix = 'analysis'
    top_n = 15
    pattern = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--top-n' and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])
            i += 2
        elif arg == '--pattern' and i + 1 < len(sys.argv):
            # Parse pattern like "45-67-12-89"
            pattern = [int(x) for x in sys.argv[i + 1].split('-')]
            i += 2
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        elif not arg.startswith('-'):
            prefix = arg
            i += 1
        else:
            i += 1

    print(f"Loading combination data from {prefix}_*.json files...")
    graph_data, combo_stats = load_combo_graph_data(prefix)

    # If pattern specified, just visualize that pattern
    if pattern:
        pattern_str = '-'.join(map(str, pattern))
        output_file = f'{prefix}_combo_pattern_{pattern_str}.png'
        print(f"Visualizing combination sequence: {pattern_str}")
        plot_combo_sequence(pattern, combo_stats, output_file)
        print("Done!")
        sys.exit(0)

    # Otherwise, create the graph visualization
    print("Building graph...")
    G = create_networkx_graph(graph_data, combo_stats)

    print_graph_stats(G)

    print(f"\nGenerating top {top_n} combination visualization...")
    output_file = f'{prefix}_combo_top{top_n}.png'
    plot_top_n_combos(G, output_file, top_n=top_n)

    print("\nDone!")
    print(f"\nTip: Use --pattern COMBO1-COMBO2-... to visualize a specific combination sequence")
