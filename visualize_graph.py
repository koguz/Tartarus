#!/usr/bin/env python3
"""
Visualize the state transition graph from agent analysis.

Usage:
    python visualize_graph.py [analysis_prefix] [--interactive]

Options:
    analysis_prefix: Prefix used when running analyze_agent.py (default: 'analysis')
    --interactive: Generate interactive HTML visualization using PyVis
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


def plot_state_perception_heatmaps(state_stats, output_file='state_perceptions.png', top_n=10):
    """
    Create 3x3 heatmaps showing what each top state typically "sees".

    Colors (blended based on proportion):
    - Empty (0): White
    - Box (1): Orange/Red
    - Wall (2): Dark gray/Black
    """
    # Get top N states by visit count
    states_by_visits = sorted(
        [(int(s), stats['visit_count'], stats) for s, stats in state_stats.items()],
        key=lambda x: -x[1]
    )[:top_n]

    # Create figure with subplots
    cols = 5
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes = axes.flatten() if top_n > 1 else [axes]

    # Colors for each cell type
    COLOR_EMPTY = np.array([1.0, 1.0, 1.0])      # White
    COLOR_BOX = np.array([0.9, 0.4, 0.1])        # Orange
    COLOR_WALL = np.array([0.2, 0.2, 0.2])       # Dark gray

    for idx, (state_id, visit_count, stats) in enumerate(states_by_visits):
        ax = axes[idx]

        # Aggregate cell type counts for each position
        # position_counts[pos][type] = count
        position_counts = {i: {0: 0, 1: 0, 2: 0} for i in range(8)}

        top_combos = stats.get('top_combinations', [])
        total_weight = 0

        for combo_info in top_combos:
            combo_idx = combo_info['combo_idx']
            count = combo_info['count']
            total_weight += count

            cells = decode_combination(combo_idx)
            for pos, cell_type in enumerate(cells):
                position_counts[pos][cell_type] += count

        # Create 3x3 image
        img = np.zeros((3, 3, 3))

        for pos in range(8):
            row, col = GRID_POS[pos]
            counts = position_counts[pos]
            total = sum(counts.values())

            if total > 0:
                # Calculate proportions
                p_empty = counts[0] / total
                p_box = counts[1] / total
                p_wall = counts[2] / total

                # Blend colors based on proportions
                color = (p_empty * COLOR_EMPTY +
                        p_box * COLOR_BOX +
                        p_wall * COLOR_WALL)
                img[row, col] = color
            else:
                img[row, col] = COLOR_EMPTY

        # Center cell is white (agent position, no special color)
        img[1, 1] = COLOR_EMPTY

        # Plot
        ax.imshow(img, interpolation='nearest')

        # Add grid lines
        for i in range(4):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        # Add percentage labels for box presence
        for pos in range(8):
            row, col = GRID_POS[pos]
            counts = position_counts[pos]
            total = sum(counts.values())
            if total > 0:
                p_box = counts[1] / total * 100
                if p_box > 5:  # Only show if significant
                    text_color = 'white' if p_box > 50 else 'black'
                    ax.text(col, row, f'{p_box:.0f}%', ha='center', va='center',
                           fontsize=9, fontweight='bold', color=text_color)

        # Arrow in center cell pointing up (indicates agent facing direction)
        ax.annotate('', xy=(1, 0.7), xytext=(1, 1.3),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

        # Get dominant action
        action_counts = stats.get('action_counts', {})
        push = action_counts.get('push', 0)
        forward = action_counts.get('forward', 0)
        turn = action_counts.get('turn_left', 0) + action_counts.get('turn_right', 0)

        if push >= forward and push >= turn:
            dominant = 'PUSH'
            title_color = '#2ecc71'
        elif forward >= turn:
            dominant = 'FWD'
            title_color = '#3498db'
        else:
            dominant = 'TURN'
            title_color = '#f1c40f'

        ax.set_title(f'State {state_id}\n{visit_count:,} visits [{dominant}]',
                    fontsize=10, fontweight='bold', color=title_color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5)

    # Hide unused subplots
    for idx in range(len(states_by_visits), len(axes)):
        axes[idx].axis('off')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_EMPTY, edgecolor='black', label='Empty'),
        mpatches.Patch(facecolor=COLOR_BOX, edgecolor='black', label='Box'),
        mpatches.Patch(facecolor=COLOR_WALL, edgecolor='black', label='Wall'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle('Top 10 States: What the Agent Typically Sees\n'
                '(% = box probability, arrow = front direction)',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved state perception heatmaps to {output_file}")


def plot_pattern_sequence(state_ids, state_stats, output_file='pattern_sequence.png', title=None):
    """
    Visualize the perceptions for a specific sequence of states (a behavioral pattern).
    Shows each state in order with its dominant action, creating a "storyboard" of the behavior.

    state_ids: list of state IDs in the pattern, e.g., [75, 42, 118, 2, 22, 94, 96, 80]
    """
    n_states = len(state_ids)

    # Create figure - horizontal layout for sequence
    fig, axes = plt.subplots(1, n_states, figsize=(2.8 * n_states, 4))
    if n_states == 1:
        axes = [axes]

    # Colors
    COLOR_EMPTY = np.array([1.0, 1.0, 1.0])
    COLOR_BOX = np.array([0.9, 0.4, 0.1])
    COLOR_WALL = np.array([0.2, 0.2, 0.2])

    action_colors = {
        'P': '#2ecc71',  # Green for Push
        'F': '#3498db',  # Blue for Forward
        'T': '#f1c40f',  # Yellow for Turn
        '?': '#95a5a6'   # Gray for unknown
    }

    action_names = {
        'P': 'PUSH',
        'F': 'FORWARD',
        'T': 'TURN',
        '?': '?'
    }

    for idx, state_id in enumerate(state_ids):
        ax = axes[idx]
        stats = state_stats.get(str(state_id), {})
        visit_count = stats.get('visit_count', 0)

        # Aggregate cell type counts
        position_counts = {i: {0: 0, 1: 0, 2: 0} for i in range(8)}

        top_combos = stats.get('top_combinations', [])
        for combo_info in top_combos:
            combo_idx = combo_info['combo_idx']
            count = combo_info['count']
            cells = decode_combination(combo_idx)
            for pos, cell_type in enumerate(cells):
                position_counts[pos][cell_type] += count

        # Create 3x3 image
        img = np.zeros((3, 3, 3))

        for pos in range(8):
            row, col = GRID_POS[pos]
            counts = position_counts[pos]
            total = sum(counts.values())

            if total > 0:
                p_empty = counts[0] / total
                p_box = counts[1] / total
                p_wall = counts[2] / total
                color = (p_empty * COLOR_EMPTY + p_box * COLOR_BOX + p_wall * COLOR_WALL)
                img[row, col] = color
            else:
                img[row, col] = COLOR_EMPTY

        img[1, 1] = COLOR_EMPTY  # Center

        ax.imshow(img, interpolation='nearest')

        # Grid lines
        for i in range(4):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)

        # Box percentages
        for pos in range(8):
            row, col = GRID_POS[pos]
            counts = position_counts[pos]
            total = sum(counts.values())
            if total > 0:
                p_box = counts[1] / total * 100
                if p_box > 5:
                    text_color = 'white' if p_box > 50 else 'black'
                    ax.text(col, row, f'{p_box:.0f}%', ha='center', va='center',
                           fontsize=8, fontweight='bold', color=text_color)

        # Arrow in center cell pointing up (agent facing direction)
        ax.annotate('', xy=(1, 0.7), xytext=(1, 1.3),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

        # Determine dominant action
        action_counts = stats.get('action_counts', {})
        push = action_counts.get('push', 0)
        forward = action_counts.get('forward', 0)
        turn = action_counts.get('turn_left', 0) + action_counts.get('turn_right', 0)

        if visit_count > 0:
            if push >= forward and push >= turn:
                dominant = 'P'
            elif forward >= turn:
                dominant = 'F'
            else:
                dominant = 'T'
        else:
            dominant = '?'

        # Title with step number, state ID, and action
        ax.set_title(f'Step {idx + 1}: State {state_id}\n{action_names[dominant]}',
                    fontsize=10, fontweight='bold', color=action_colors[dominant])

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
    pattern_str = '-'.join(map(str, state_ids))
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'Behavioral Pattern: {pattern_str}\n(% = box probability)',
                    fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved pattern visualization to {output_file}")


def load_graph_data(prefix='analysis'):
    """Load transition graph and state stats."""
    with open(f'{prefix}_transitions.json', 'r') as f:
        transitions = json.load(f)

    with open(f'{prefix}_state_stats.json', 'r') as f:
        state_stats = json.load(f)

    return transitions, state_stats


def create_networkx_graph(transitions, state_stats):
    """Create a NetworkX directed graph from the data."""
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in transitions['nodes']:
        stats = state_stats.get(str(node), {})
        visit_count = stats.get('visit_count', 0)
        action_counts = stats.get('action_counts', {})

        push_count = action_counts.get('push', 0)
        forward_count = action_counts.get('forward', 0)
        turn_left_count = action_counts.get('turn_left', 0)
        turn_right_count = action_counts.get('turn_right', 0)
        turn_count = turn_left_count + turn_right_count

        # Determine dominant action
        action_totals = {
            'push': push_count,
            'forward': forward_count,
            'turn': turn_count
        }
        dominant_action = max(action_totals, key=action_totals.get) if visit_count > 0 else 'none'

        G.add_node(node,
                   visit_count=visit_count,
                   push_count=push_count,
                   forward_count=forward_count,
                   turn_count=turn_count,
                   dominant_action=dominant_action)

    # Add edges with weights
    for edge in transitions['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

    return G


def separate_overlapping_nodes(pos, min_distance=0.8):
    """
    Iteratively push apart nodes that are too close together.
    """
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
    """
    Create a radial layout with high-centrality nodes in the center.
    Ensures no nodes overlap.
    """
    if center_metric == 'visits':
        centrality = {n: G.nodes[n].get('visit_count', 0) for n in G.nodes()}
    else:
        centrality = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}

    # Sort nodes by centrality (highest first)
    sorted_nodes = sorted(G.nodes(), key=lambda n: centrality[n], reverse=True)

    n_nodes = len(sorted_nodes)
    pos = {}

    # Calculate ring assignments with proper spacing
    # Each ring can hold: circumference / min_distance nodes
    rings = []
    nodes_placed = 0
    ring_idx = 0
    base_radius = 2.0  # Starting radius for first ring with nodes

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


def radial_layout_by_action(G, min_node_distance=1.2):
    """
    Create a radial layout clustered by dominant action.
    Push nodes in one sector, Forward in another, Turn in another.
    """
    # Group nodes by dominant action
    action_groups = {'push': [], 'forward': [], 'turn': [], 'none': []}

    for node in G.nodes():
        action = G.nodes[node].get('dominant_action', 'none')
        visit_count = G.nodes[node].get('visit_count', 0)
        action_groups[action].append((node, visit_count))

    # Sort each group by visit count (highest first)
    for action in action_groups:
        action_groups[action].sort(key=lambda x: -x[1])

    pos = {}

    # Define sector angles for each action type
    # Push: top (green), Forward: bottom-left (blue), Turn: bottom-right (yellow)
    sector_centers = {
        'push': -math.pi / 2,      # Top (90 degrees up)
        'forward': 5 * math.pi / 6,  # Bottom-left (150 degrees)
        'turn': -math.pi / 6,       # Bottom-right (330 degrees / -30 degrees)
        'none': math.pi             # Left (hidden)
    }
    sector_width = math.pi / 2  # Each sector spans 90 degrees

    for action, nodes_list in action_groups.items():
        if not nodes_list:
            continue

        center_angle = sector_centers[action]
        n_nodes = len(nodes_list)

        # Place nodes in expanding arcs within the sector
        nodes_placed = 0
        ring_idx = 0
        base_radius = 2.0

        while nodes_placed < n_nodes:
            if ring_idx == 0 and n_nodes > 0:
                # First node of group closer to center
                radius = base_radius
                ring_size = 1
            else:
                radius = base_radius + ring_idx * min_node_distance * 1.8
                # Arc length at this radius within sector
                arc_length = radius * sector_width
                ring_size = max(1, int(arc_length / min_node_distance))
                ring_size = min(ring_size, n_nodes - nodes_placed)

            ring_nodes = nodes_list[nodes_placed:nodes_placed + ring_size]

            for i, (node, visits) in enumerate(ring_nodes):
                if ring_size == 1:
                    angle = center_angle
                else:
                    # Spread across the sector
                    spread = sector_width * 0.8  # Use 80% of sector width
                    angle = center_angle - spread/2 + spread * i / (ring_size - 1)

                pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

            nodes_placed += ring_size
            ring_idx += 1

    # Separate overlapping nodes
    pos = separate_overlapping_nodes(pos, min_node_distance)

    return pos


def plot_static_graph(G, output_file='graph_static.png', state_stats=None):
    """Create a static matplotlib visualization with radial layout."""
    fig, ax = plt.subplots(figsize=(20, 20))

    # Filter to show only active nodes and significant edges
    min_visits = 100
    min_edge_weight = 500

    active_nodes = [n for n in G.nodes() if G.nodes[n].get('visit_count', 0) >= min_visits]

    if len(active_nodes) < 10:
        print("Few nodes meet threshold, showing all nodes with visits > 0")
        active_nodes = [n for n in G.nodes() if G.nodes[n].get('visit_count', 0) > 0]

    H = G.subgraph(active_nodes).copy()

    edges_to_remove = [(u, v) for u, v, d in H.edges(data=True) if d.get('weight', 0) < min_edge_weight]
    H.remove_edges_from(edges_to_remove)

    isolated = list(nx.isolates(H))
    H.remove_nodes_from(isolated)

    if len(H.nodes()) == 0:
        print("Graph empty after filtering, reducing thresholds...")
        H = G.subgraph([n for n in G.nodes() if G.nodes[n].get('visit_count', 0) > 0]).copy()
        min_edge_weight = 100
        edges_to_remove = [(u, v) for u, v, d in H.edges(data=True) if d.get('weight', 0) < min_edge_weight]
        H.remove_edges_from(edges_to_remove)

    print(f"Plotting {len(H.nodes())} nodes and {len(H.edges())} edges")

    # Use radial layout with overlap prevention
    pos, centrality = radial_layout(H, center_metric='visits', min_node_distance=1.2)

    # Node sizes based on visit count
    visit_counts = [H.nodes[n].get('visit_count', 1) for n in H.nodes()]
    max_visits = max(visit_counts) if visit_counts else 1
    node_sizes = [200 + 2500 * (v / max_visits) for v in visit_counts]

    # Node colors based on centrality
    centrality_values = [centrality.get(n, 0) for n in H.nodes()]
    max_cent = max(centrality_values) if centrality_values else 1

    # Edge widths based on weight
    edge_weights = [H.edges[e].get('weight', 1) for e in H.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.3 + 4 * (w / max_weight) for w in edge_weights]
    edge_alphas = [0.15 + 0.5 * (w / max_weight) for w in edge_weights]

    # Draw edges
    for (u, v), width, alpha in zip(H.edges(), edge_widths, edge_alphas):
        ax.annotate("",
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    color='#666666',
                                    alpha=alpha,
                                    connectionstyle="arc3,rad=0.1",
                                    lw=width))

    # Draw nodes
    nx.draw_networkx_nodes(H, pos, ax=ax,
                           node_size=node_sizes,
                           node_color=centrality_values,
                           cmap=plt.cm.YlOrRd,
                           vmin=0, vmax=max_cent,
                           alpha=0.9,
                           edgecolors='black',
                           linewidths=1)

    nx.draw_networkx_labels(H, pos, ax=ax, font_size=9, font_weight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                norm=plt.Normalize(vmin=0, vmax=max_cent))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Visit Count (center = most visited)', fontsize=11)

    ax.set_title(f'State Transition Graph (Radial Layout)\n'
                f'{len(H.nodes())} states, {len(H.edges())} transitions\n'
                f'Center = most visited states',
                fontsize=14)
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
    print(f"Saved static graph to {output_file}")


def plot_top_n_graph(G, output_file='graph_top15.png', state_stats=None, top_n=15):
    """
    Create a clean graph showing only the top N most visited states
    with all transitions between them.
    """
    fig, ax = plt.subplots(figsize=(16, 16))

    # Get top N states by visit count
    all_nodes = [(n, G.nodes[n].get('visit_count', 0)) for n in G.nodes()]
    all_nodes.sort(key=lambda x: -x[1])
    top_nodes = [n for n, v in all_nodes[:top_n]]

    # Create subgraph with only these nodes
    H = G.subgraph(top_nodes).copy()

    print(f"Top {top_n} graph: {len(H.nodes())} nodes and {len(H.edges())} edges")

    # Use radial layout - most visited in center
    pos, centrality = radial_layout(H, center_metric='visits', min_node_distance=2.0)

    # Node sizes based on visit count
    visit_counts = [H.nodes[n].get('visit_count', 1) for n in H.nodes()]
    max_visits = max(visit_counts) if visit_counts else 1
    node_sizes = [500 + 4000 * (v / max_visits) for v in visit_counts]

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
        edge_widths = [0.5 + 6 * (w / max_weight) for w in edge_weights]
        edge_alphas = [0.3 + 0.5 * (w / max_weight) for w in edge_weights]
    else:
        edge_widths = []
        edge_alphas = []

    # Draw edges with colors based on source node action
    for (u, v), width, alpha in zip(H.edges(), edge_widths, edge_alphas):
        source_action = H.nodes[u].get('dominant_action', 'none')
        edge_color = action_colors.get(source_action, '#666666')
        ax.annotate("",
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="-|>",
                                    color=edge_color,
                                    alpha=alpha,
                                    connectionstyle="arc3,rad=0.15",
                                    lw=width,
                                    mutation_scale=20))

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

    # Add visit counts as annotations
    for node in H.nodes():
        x, y = pos[node]
        visits = H.nodes[node].get('visit_count', 0)
        ax.text(x, y - 0.4, f'{visits:,}', ha='center', va='top',
               fontsize=8, color='#555555')

    ax.set_title(f'Top {top_n} Most Visited States\n'
                f'Node color = dominant action, Edge color = source action\n'
                f'Node size & label below = visit count',
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
    print(f"Saved top {top_n} graph to {output_file}")


def plot_action_clustered_graph(G, output_file='graph_actions.png', state_stats=None):
    """
    Create a graph clustered by dominant action type.
    Push = Green, Forward = Blue, Turn = Yellow
    Node size = visit count
    """
    fig, ax = plt.subplots(figsize=(22, 20))

    # Filter nodes
    min_visits = 100
    min_edge_weight = 500

    active_nodes = [n for n in G.nodes() if G.nodes[n].get('visit_count', 0) >= min_visits]

    if len(active_nodes) < 10:
        active_nodes = [n for n in G.nodes() if G.nodes[n].get('visit_count', 0) > 0]

    H = G.subgraph(active_nodes).copy()

    edges_to_remove = [(u, v) for u, v, d in H.edges(data=True) if d.get('weight', 0) < min_edge_weight]
    H.remove_edges_from(edges_to_remove)

    isolated = list(nx.isolates(H))
    H.remove_nodes_from(isolated)

    print(f"Action graph: {len(H.nodes())} nodes and {len(H.edges())} edges")

    # Use action-based clustered layout
    pos = radial_layout_by_action(H, min_node_distance=1.3)

    # Node sizes based on visit count
    visit_counts = [H.nodes[n].get('visit_count', 1) for n in H.nodes()]
    max_visits = max(visit_counts) if visit_counts else 1
    # Use sqrt scaling for better size distribution
    node_sizes = [100 + 3000 * math.sqrt(v / max_visits) for v in visit_counts]

    # Node colors based on dominant action
    action_colors = {
        'push': '#2ecc71',     # Green
        'forward': '#3498db',  # Blue
        'turn': '#f1c40f',     # Yellow
        'none': '#95a5a6'      # Gray
    }
    node_colors = [action_colors.get(H.nodes[n].get('dominant_action', 'none'), '#95a5a6')
                   for n in H.nodes()]

    # Edge widths
    edge_weights = [H.edges[e].get('weight', 1) for e in H.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.2 + 3 * (w / max_weight) for w in edge_weights]
    edge_alphas = [0.1 + 0.4 * (w / max_weight) for w in edge_weights]

    # Color edges based on source node's action
    edge_colors = []
    for u, v in H.edges():
        action = H.nodes[u].get('dominant_action', 'none')
        color = action_colors.get(action, '#666666')
        edge_colors.append(color)

    # Draw edges
    for (u, v), width, alpha, color in zip(H.edges(), edge_widths, edge_alphas, edge_colors):
        ax.annotate("",
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    color=color,
                                    alpha=alpha,
                                    connectionstyle="arc3,rad=0.15",
                                    lw=width))

    # Draw nodes
    nx.draw_networkx_nodes(H, pos, ax=ax,
                           node_size=node_sizes,
                           node_color=node_colors,
                           alpha=0.9,
                           edgecolors='black',
                           linewidths=1.5)

    nx.draw_networkx_labels(H, pos, ax=ax, font_size=9, font_weight='bold')

    # Legend
    legend_patches = [
        mpatches.Patch(color='#2ecc71', label='Push (dominant)'),
        mpatches.Patch(color='#3498db', label='Forward (dominant)'),
        mpatches.Patch(color='#f1c40f', label='Turn (dominant)'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=12, framealpha=0.9)

    # Add cluster labels
    ax.text(0, max([p[1] for p in pos.values()]) + 2, 'PUSH',
            ha='center', va='bottom', fontsize=16, fontweight='bold', color='#2ecc71')
    ax.text(-max([abs(p[0]) for p in pos.values()]) - 1, -2, 'FORWARD',
            ha='center', va='center', fontsize=16, fontweight='bold', color='#3498db', rotation=60)
    ax.text(max([abs(p[0]) for p in pos.values()]) + 1, -2, 'TURN',
            ha='center', va='center', fontsize=16, fontweight='bold', color='#f1c40f', rotation=-60)

    ax.set_title(f'State Transition Graph (Clustered by Action)\n'
                f'{len(H.nodes())} states · Node size = visit count\n'
                f'Green = Push, Blue = Forward, Yellow = Turn',
                fontsize=14)
    ax.set_aspect('equal')
    ax.axis('off')

    # Set axis limits
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    margin = 4
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved action-clustered graph to {output_file}")

    # Print action statistics
    print("\nAction distribution:")
    for action in ['push', 'forward', 'turn']:
        nodes = [n for n in H.nodes() if H.nodes[n].get('dominant_action') == action]
        total_visits = sum(H.nodes[n].get('visit_count', 0) for n in nodes)
        print(f"  {action.capitalize():8s}: {len(nodes):3d} states, {total_visits:>10,} total visits")


def plot_interactive_graph(G, output_file='graph_interactive.html', state_stats=None):
    """Create an interactive PyVis visualization."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("PyVis not installed. Install with: pip install pyvis")
        return

    net = Network(height='900px', width='100%', directed=True,
                  bgcolor='#1a1a2e', font_color='white')

    net.set_options('''
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.5,
                "springLength": 150,
                "springConstant": 0.04
            }
        },
        "nodes": {
            "font": {"size": 14, "face": "arial"}
        },
        "edges": {
            "smooth": {"type": "curvedCW", "roundness": 0.2}
        }
    }
    ''')

    visit_counts = {n: G.nodes[n].get('visit_count', 0) for n in G.nodes()}
    max_visits = max(visit_counts.values()) if visit_counts.values() else 1

    edge_weights = [G.edges[e].get('weight', 1) for e in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1

    action_colors = {
        'push': '#2ecc71',
        'forward': '#3498db',
        'turn': '#f1c40f',
        'none': '#95a5a6'
    }

    for node in G.nodes():
        stats = state_stats.get(str(node), {}) if state_stats else {}
        visit_count = visit_counts[node]
        dominant_action = G.nodes[node].get('dominant_action', 'none')

        size = 10 + 50 * (math.log10(visit_count + 1) / math.log10(max_visits + 1)) if visit_count > 0 else 5
        color = action_colors.get(dominant_action, '#95a5a6')

        action_counts = stats.get('action_counts', {})
        title = (f"<b>State {node}</b><br>"
                f"Dominant: {dominant_action}<br>"
                f"Visits: {visit_count:,}<br>"
                f"Forward: {action_counts.get('forward', 0):,}<br>"
                f"Push: {action_counts.get('push', 0):,}<br>"
                f"Turn L: {action_counts.get('turn_left', 0):,}<br>"
                f"Turn R: {action_counts.get('turn_right', 0):,}")

        mass = 1 + 5 * (visit_count / max_visits)
        net.add_node(node, label=str(node), size=size, color=color, title=title, mass=mass)

    min_weight = max_weight * 0.005
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        if weight >= min_weight:
            width = 0.5 + 5 * (weight / max_weight)
            intensity = int(100 + 155 * (weight / max_weight))
            edge_color = f'#{intensity:02x}{intensity:02x}{intensity:02x}'
            net.add_edge(u, v, width=width, color=edge_color, title=f"Transitions: {weight:,}")

    net.save_graph(output_file)
    print(f"Saved interactive graph to {output_file}")


def plot_adjacency_heatmap(G, output_file='graph_heatmap.png'):
    """Create a heatmap of the transition matrix, sorted by visit count."""
    fig, ax = plt.subplots(figsize=(14, 12))

    nodes = sorted(G.nodes(), key=lambda n: G.nodes[n].get('visit_count', 0), reverse=True)
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    matrix = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        matrix[node_idx[u], node_idx[v]] = data.get('weight', 0)

    matrix_log = np.log10(matrix + 1)

    im = ax.imshow(matrix_log, cmap='hot', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(transitions + 1)', fontsize=11)

    tick_step = max(1, n // 20)
    ax.set_xticks(range(0, n, tick_step))
    ax.set_xticklabels([nodes[i] for i in range(0, n, tick_step)], rotation=45, ha='right')
    ax.set_yticks(range(0, n, tick_step))
    ax.set_yticklabels([nodes[i] for i in range(0, n, tick_step)])

    ax.set_xlabel('To State', fontsize=12)
    ax.set_ylabel('From State', fontsize=12)
    ax.set_title('State Transition Matrix\n(sorted by visit count)', fontsize=13)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_file}")


def print_graph_stats(G):
    """Print useful statistics about the graph."""
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)

    print(f"\nNodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    visit_counts = [(n, G.nodes[n].get('visit_count', 0)) for n in G.nodes()]
    visit_counts.sort(key=lambda x: -x[1])

    print(f"\nTop 15 most visited states:")
    for node, visits in visit_counts[:15]:
        action = G.nodes[node].get('dominant_action', 'none')
        print(f"  State {node:3d}: {visits:>10,} visits  [{action}]")

    # Action distribution
    print(f"\nStates by dominant action:")
    for action in ['push', 'forward', 'turn', 'none']:
        nodes = [n for n in G.nodes() if G.nodes[n].get('dominant_action') == action]
        if nodes:
            total_visits = sum(G.nodes[n].get('visit_count', 0) for n in nodes)
            print(f"  {action.capitalize():8s}: {len(nodes):3d} states, {total_visits:>10,} total visits")

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    print(f"\nTop 10 by in-degree:")
    top_in = sorted(in_degrees.items(), key=lambda x: -x[1])[:10]
    for node, deg in top_in:
        print(f"  State {node:3d}: {deg} incoming")

    print(f"\nTop 10 by out-degree:")
    top_out = sorted(out_degrees.items(), key=lambda x: -x[1])[:10]
    for node, deg in top_out:
        print(f"  State {node:3d}: {deg} outgoing")

    sccs = list(nx.strongly_connected_components(G))
    print(f"\nStrongly connected components: {len(sccs)}")
    largest_scc = max(sccs, key=len)
    print(f"Largest SCC: {len(largest_scc)} states")

    print(f"\nTop 10 most frequent transitions:")
    edges_by_weight = sorted(G.edges(data=True), key=lambda x: -x[2].get('weight', 0))[:10]
    for u, v, data in edges_by_weight:
        print(f"  {u:3d} -> {v:3d}: {data.get('weight', 0):>10,}")


if __name__ == '__main__':
    prefix = 'analysis'
    interactive = False
    pattern = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--interactive':
            interactive = True
            i += 1
        elif arg == '--pattern':
            # Parse pattern like "75-42-118-2-22-94-96-80"
            if i + 1 < len(sys.argv):
                pattern = [int(x) for x in sys.argv[i + 1].split('-')]
                i += 2
            else:
                i += 1
        elif not arg.startswith('-'):
            prefix = arg
            i += 1
        else:
            i += 1

    print(f"Loading data from {prefix}_*.json files...")
    transitions, state_stats = load_graph_data(prefix)

    # If pattern specified, just visualize that pattern
    if pattern:
        pattern_str = '-'.join(map(str, pattern))
        output_file = f'{prefix}_pattern_{pattern_str}.png'
        plot_pattern_sequence(pattern, state_stats, output_file)
        print("Done!")
        sys.exit(0)

    print("Building graph...")
    G = create_networkx_graph(transitions, state_stats)

    print_graph_stats(G)

    print("\nGenerating visualizations...")
    plot_adjacency_heatmap(G, f'{prefix}_heatmap.png')
    plot_static_graph(G, f'{prefix}_graph.png', state_stats)
    plot_top_n_graph(G, f'{prefix}_top15.png', state_stats, top_n=15)
    plot_action_clustered_graph(G, f'{prefix}_actions.png', state_stats)
    plot_state_perception_heatmaps(state_stats, f'{prefix}_perceptions.png', top_n=10)

    if interactive:
        plot_interactive_graph(G, f'{prefix}_graph.html', state_stats)
    else:
        print("\nTip: Run with --interactive for an interactive HTML visualization")
        print("     Run with --pattern 75-42-118 to visualize a specific state sequence")
