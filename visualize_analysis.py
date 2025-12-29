"""
Tartarus FSM Brain Analysis Visualizer
Generates heatmaps, network graphs, and behavioral analysis from kernel_analyze.cu output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import networkx as nx
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_analysis_files(base_name):
    """Load all analysis CSV files for a given solution"""
    files = {}

    # Try different path patterns
    patterns = [
        f"analysis_heatmap_{base_name}.csv",
        f"analysis_heatmap_txt_{base_name}.csv",
    ]

    # Find the actual file names
    for f in Path('.').glob('analysis_*.csv'):
        name = f.stem
        if 'heatmap' in name:
            files['heatmap'] = pd.read_csv(f)
        elif 'transitions' in name:
            files['transitions'] = pd.read_csv(f)
        elif 'pushes' in name:
            files['pushes'] = pd.read_csv(f)
        elif 'state_actions' in name:
            files['state_actions'] = pd.read_csv(f)
        elif 'combo_triggers' in name:
            files['combo_triggers'] = pd.read_csv(f)
        elif 'genes' in name:
            files['genes'] = pd.read_csv(f)

    return files

def plot_state_combo_heatmap(heatmap_df, S, output_prefix):
    """Create heatmap of state-combination usage"""
    # Create full matrix
    matrix = np.zeros((S, 383))
    for _, row in heatmap_df.iterrows():
        matrix[int(row['state']), int(row['combo'])] = row['count']

    # Normalize by row (each state)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Raw counts (log scale)
    im1 = axes[0].imshow(np.log1p(matrix), aspect='auto', cmap='viridis')
    axes[0].set_xlabel('Combination Index (0-382)')
    axes[0].set_ylabel('State')
    axes[0].set_title('State-Combination Usage (log scale)')
    axes[0].set_yticks(range(S))
    plt.colorbar(im1, ax=axes[0], label='log(count + 1)')

    # Normalized (probability per state)
    im2 = axes[1].imshow(matrix_norm, aspect='auto', cmap='hot')
    axes[1].set_xlabel('Combination Index (0-382)')
    axes[1].set_ylabel('State')
    axes[1].set_title('Combination Distribution per State (normalized)')
    axes[1].set_yticks(range(S))
    plt.colorbar(im2, ax=axes[1], label='Probability')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_heatmap.png', dpi=150)
    plt.close()
    print(f"Saved: {output_prefix}_heatmap.png")

    return matrix

def plot_transition_graph(transitions_df, S, output_prefix, threshold_pct=0.5):
    """Create state transition network graph"""
    # Build adjacency matrix
    adj = np.zeros((S, S))
    for _, row in transitions_df.iterrows():
        adj[int(row['from_state']), int(row['to_state'])] = row['count']

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with sizes based on total incoming transitions
    node_sizes = adj.sum(axis=0) + adj.sum(axis=1)
    for i in range(S):
        G.add_node(i, size=node_sizes[i])

    # Add edges (filter by threshold)
    total = adj.sum()
    threshold = total * threshold_pct / 100

    for i in range(S):
        for j in range(S):
            if adj[i, j] > threshold:
                G.add_edge(i, j, weight=adj[i, j])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: Network graph
    ax1 = axes[0]
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes
    node_size = [G.nodes[n]['size'] / max(node_sizes) * 2000 + 200 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=node_size,
                           node_color=list(range(S)), cmap='tab20', alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10, font_weight='bold')

    # Draw edges with varying width
    edges = G.edges(data=True)
    if edges:
        weights = [e[2]['weight'] for e in edges]
        max_w = max(weights)
        edge_widths = [w / max_w * 5 + 0.5 for w in weights]

        # Separate self-loops and regular edges
        self_loops = [(u, v) for u, v, _ in edges if u == v]
        other_edges = [(u, v) for u, v, _ in edges if u != v]
        other_widths = [edge_widths[i] for i, (u, v, _) in enumerate(edges) if u != v]

        nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=other_edges,
                               width=other_widths, alpha=0.6,
                               edge_color='gray', arrows=True,
                               arrowsize=15, connectionstyle='arc3,rad=0.1')

    ax1.set_title(f'State Transition Graph (edges > {threshold_pct}% of total)')
    ax1.axis('off')

    # Right: Transition matrix heatmap
    ax2 = axes[1]
    im = ax2.imshow(np.log1p(adj), cmap='Blues')
    ax2.set_xlabel('To State')
    ax2.set_ylabel('From State')
    ax2.set_title('State Transition Matrix (log scale)')
    ax2.set_xticks(range(S))
    ax2.set_yticks(range(S))
    plt.colorbar(im, ax=ax2, label='log(transitions + 1)')

    # Highlight diagonal (self-loops)
    for i in range(S):
        ax2.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                     edgecolor='red', linewidth=2))

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_transitions.png', dpi=150)
    plt.close()
    print(f"Saved: {output_prefix}_transitions.png")

    return adj

def plot_state_profiles(state_actions_df, output_prefix):
    """Create state behavior profiles (action distribution per state)"""
    S = len(state_actions_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Stacked bar chart
    ax1 = axes[0]
    states = state_actions_df['state'].values
    fwd = state_actions_df['forward_pct'].values
    left = state_actions_df['left_pct'].values
    right = state_actions_df['right_pct'].values

    x = np.arange(S)
    width = 0.8

    ax1.bar(x, fwd, width, label='Forward', color='green', alpha=0.8)
    ax1.bar(x, left, width, bottom=fwd, label='Turn Left', color='blue', alpha=0.8)
    ax1.bar(x, right, width, bottom=fwd+left, label='Turn Right', color='orange', alpha=0.8)

    ax1.set_xlabel('State')
    ax1.set_ylabel('Percentage')
    ax1.set_title('Action Distribution per State')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.axhline(y=33.33, color='red', linestyle='--', alpha=0.5, label='Uniform')

    # Right: Scatter plot - Forward% vs Turn%
    ax2 = axes[1]
    turn_pct = left + right

    scatter = ax2.scatter(fwd, turn_pct, c=states, cmap='tab20', s=200, alpha=0.8)
    for i, state in enumerate(states):
        ax2.annotate(str(state), (fwd[i], turn_pct[i]), ha='center', va='center',
                    fontweight='bold', fontsize=9)

    ax2.set_xlabel('Forward %')
    ax2.set_ylabel('Turn % (Left + Right)')
    ax2.set_title('State Behavior: Movement vs Rotation')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5)

    # Quadrant labels
    ax2.text(75, 75, 'Balanced', ha='center', alpha=0.5)
    ax2.text(25, 75, 'Rotation\nFocused', ha='center', alpha=0.5)
    ax2.text(75, 25, 'Movement\nFocused', ha='center', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_state_profiles.png', dpi=150)
    plt.close()
    print(f"Saved: {output_prefix}_state_profiles.png")

    # Classify states
    print("\nState Classification:")
    for i, row in state_actions_df.iterrows():
        fwd_pct = row['forward_pct']
        if fwd_pct > 50:
            category = "MOVER (forward-heavy)"
        elif fwd_pct < 30:
            category = "TURNER (rotation-heavy)"
        else:
            category = "BALANCED"
        print(f"  State {int(row['state']):2d}: {category} (Fwd:{fwd_pct:.1f}%)")

def plot_push_analysis(pushes_df, state_actions_df, S, output_prefix):
    """Analyze pushing behavior per state"""
    # Aggregate pushes per state
    state_pushes = pushes_df.groupby('state')['push_count'].sum().reindex(range(S), fill_value=0)
    state_totals = pushes_df.groupby('state')['total_count'].sum().reindex(range(S), fill_value=1)
    push_rates = state_pushes / state_totals * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Push counts per state
    ax1 = axes[0]
    colors = plt.cm.Greens(push_rates / push_rates.max())
    ax1.bar(range(S), state_pushes.values, color=colors)
    ax1.set_xlabel('State')
    ax1.set_ylabel('Total Successful Pushes')
    ax1.set_title('Push Count per State')
    ax1.set_xticks(range(S))

    # Right: Push rate vs Forward rate
    ax2 = axes[1]
    fwd_rates = state_actions_df.set_index('state')['forward_pct'].reindex(range(S), fill_value=0)

    ax2.scatter(fwd_rates, push_rates, c=range(S), cmap='tab20', s=200, alpha=0.8)
    for i in range(S):
        ax2.annotate(str(i), (fwd_rates[i], push_rates[i]), ha='center', va='center',
                    fontweight='bold', fontsize=9)

    ax2.set_xlabel('Forward Action %')
    ax2.set_ylabel('Successful Push Rate %')
    ax2.set_title('Push Efficiency: Forward Actions vs Successful Pushes')

    # Add trend line
    z = np.polyfit(fwd_rates, push_rates, 1)
    p = np.poly1d(z)
    ax2.plot(fwd_rates.sort_values(), p(fwd_rates.sort_values()),
             'r--', alpha=0.5, label=f'Trend')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_push_analysis.png', dpi=150)
    plt.close()
    print(f"Saved: {output_prefix}_push_analysis.png")

    # Print top pushing states
    print("\nTop Pushing States:")
    for state in state_pushes.nlargest(5).index:
        print(f"  State {state}: {state_pushes[state]} pushes ({push_rates[state]:.2f}% rate)")

def analyze_memory_usage(genes_df, heatmap_df, S, output_prefix):
    """Detect memory usage: same combo, different state, different action"""

    # Find combos that appear in multiple states with different actions
    memory_combos = []

    for combo in range(383):
        combo_genes = genes_df[genes_df['combo'] == combo]
        actions_per_state = combo_genes.set_index('state')['action']

        # Check if different states have different actions for this combo
        unique_actions = actions_per_state.unique()
        if len(unique_actions) > 1:
            # This combo has different actions in different states = MEMORY!
            memory_combos.append({
                'combo': combo,
                'num_actions': len(unique_actions),
                'actions': list(unique_actions),
                'states_per_action': {
                    a: list(actions_per_state[actions_per_state == a].index)
                    for a in unique_actions
                }
            })

    print(f"\n=== MEMORY ANALYSIS ===")
    print(f"Combos with state-dependent actions: {len(memory_combos)} / 383 ({100*len(memory_combos)/383:.1f}%)")

    # Categorize by action diversity
    two_action = [m for m in memory_combos if m['num_actions'] == 2]
    three_action = [m for m in memory_combos if m['num_actions'] == 3]

    print(f"  - 2 different actions: {len(two_action)} combos")
    print(f"  - 3 different actions: {len(three_action)} combos")

    # Save detailed memory analysis
    with open(f'{output_prefix}_memory_analysis.txt', 'w') as f:
        f.write("MEMORY ANALYSIS: State-Dependent Action Selection\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total combos with memory-dependent behavior: {len(memory_combos)}/383\n\n")

        for m in sorted(memory_combos, key=lambda x: -x['num_actions'])[:50]:
            f.write(f"Combo {m['combo']}:\n")
            for action, states in m['states_per_action'].items():
                action_name = ['FORWARD', 'LEFT', 'RIGHT'][action]
                f.write(f"  {action_name}: States {states}\n")
            f.write("\n")

    print(f"Saved: {output_prefix}_memory_analysis.txt")

    # Visualize memory usage
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create matrix: rows=combos with memory, cols=states, color=action
    memory_matrix = np.full((len(memory_combos), S), -1)
    for i, m in enumerate(memory_combos):
        for action, states in m['states_per_action'].items():
            for state in states:
                memory_matrix[i, state] = action

    # Custom colormap
    cmap = mcolors.ListedColormap(['lightgray', 'green', 'blue', 'orange'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(memory_matrix, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xlabel('State')
    ax.set_ylabel('Memory-Active Combo Index')
    ax.set_title(f'Memory Usage: Same Combo, Different Actions ({len(memory_combos)} combos)')
    ax.set_xticks(range(S))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2])
    cbar.ax.set_yticklabels(['Unused', 'Forward', 'Left', 'Right'])

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_memory_usage.png', dpi=150)
    plt.close()
    print(f"Saved: {output_prefix}_memory_usage.png")

    return memory_combos

def plot_combo_triggers(combo_triggers_df, output_prefix):
    """Visualize which combos trigger state changes"""

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Top: Bar chart of state changes per combo
    ax1 = axes[0]
    changes = combo_triggers_df['state_changes'].values
    ax1.bar(range(383), changes, width=1.0, alpha=0.7)
    ax1.set_xlabel('Combination Index')
    ax1.set_ylabel('State Change Count')
    ax1.set_title('State Change Triggers by Combination')
    ax1.set_xlim(0, 383)

    # Highlight top triggers
    top_n = 10
    top_indices = np.argsort(changes)[-top_n:]
    for idx in top_indices:
        ax1.axvline(x=idx, color='red', alpha=0.3, linewidth=1)

    # Bottom: Distribution
    ax2 = axes[1]
    ax2.hist(changes, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('State Changes per Combo')
    ax2.set_ylabel('Number of Combos')
    ax2.set_title('Distribution of State Change Frequency')
    ax2.axvline(x=np.mean(changes), color='red', linestyle='--',
                label=f'Mean: {np.mean(changes):.0f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_combo_triggers.png', dpi=150)
    plt.close()
    print(f"Saved: {output_prefix}_combo_triggers.png")

    # Print top trigger combos
    print("\nTop State-Change Trigger Combos:")
    top_df = combo_triggers_df.nlargest(10, 'state_changes')
    for _, row in top_df.iterrows():
        print(f"  Combo {int(row['combo'])}: {int(row['state_changes'])} state changes")

def find_state_communities(transitions_df, S, output_prefix):
    """Find communities/modules in the state transition graph"""

    # Build graph
    G = nx.DiGraph()
    for i in range(S):
        G.add_node(i)

    for _, row in transitions_df.iterrows():
        G.add_edge(int(row['from_state']), int(row['to_state']),
                   weight=row['count'])

    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    # Find communities using greedy modularity
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G_undirected))

        print(f"\n=== STATE COMMUNITIES ===")
        print(f"Found {len(communities)} communities:")
        for i, comm in enumerate(communities):
            print(f"  Community {i+1}: States {sorted(comm)}")

        # Visualize communities
        fig, ax = plt.subplots(figsize=(12, 10))

        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        # Color nodes by community
        colors = []
        for node in G.nodes():
            for i, comm in enumerate(communities):
                if node in comm:
                    colors.append(i)
                    break

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                               cmap='Set1', node_size=800, alpha=0.8)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')

        # Draw edges
        edges = G.edges(data=True)
        weights = [e[2]['weight'] for e in edges]
        max_w = max(weights) if weights else 1

        nx.draw_networkx_edges(G, pos, ax=ax,
                               width=[w/max_w * 3 + 0.2 for w in weights],
                               alpha=0.4, arrows=True, arrowsize=15)

        ax.set_title(f'State Transition Graph with {len(communities)} Communities')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_prefix}_communities.png', dpi=150)
        plt.close()
        print(f"Saved: {output_prefix}_communities.png")

        return communities

    except Exception as e:
        print(f"Community detection failed: {e}")
        return []

def create_summary_report(files, S, output_prefix):
    """Generate a comprehensive summary report"""

    report = []
    report.append("=" * 70)
    report.append("TARTARUS FSM BRAIN ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"\nNumber of States: {S}")
    report.append(f"Number of Combinations: 383")
    report.append(f"Total Genes: {S * 383}")

    # State usage stats
    if 'state_actions' in files:
        df = files['state_actions']
        report.append(f"\n--- STATE USAGE ---")
        total = df['total'].sum()
        report.append(f"Total actions recorded: {total:,}")

        most_used = df.loc[df['total'].idxmax()]
        least_used = df.loc[df['total'].idxmin()]
        report.append(f"Most active state: {int(most_used['state'])} ({most_used['total']:,} actions)")
        report.append(f"Least active state: {int(least_used['state'])} ({least_used['total']:,} actions)")

    # Push stats
    if 'pushes' in files:
        df = files['pushes']
        total_pushes = df['push_count'].sum()
        report.append(f"\n--- PUSH STATISTICS ---")
        report.append(f"Total successful pushes: {total_pushes:,}")
        report.append(f"Pushes per run: {total_pushes / 74760:.2f}")

    # Transition stats
    if 'transitions' in files:
        df = files['transitions']
        self_loops = df[df['from_state'] == df['to_state']]['count'].sum()
        total_trans = df['count'].sum()
        report.append(f"\n--- TRANSITION STATISTICS ---")
        report.append(f"Total transitions: {total_trans:,}")
        report.append(f"Self-loops: {self_loops:,} ({100*self_loops/total_trans:.1f}%)")
        report.append(f"State changes: {total_trans - self_loops:,} ({100*(total_trans-self_loops)/total_trans:.1f}%)")

    # Memory usage
    if 'genes' in files:
        df = files['genes']
        memory_combos = 0
        for combo in range(383):
            actions = df[df['combo'] == combo]['action'].unique()
            if len(actions) > 1:
                memory_combos += 1
        report.append(f"\n--- MEMORY USAGE ---")
        report.append(f"Combos with state-dependent actions: {memory_combos}/383 ({100*memory_combos/383:.1f}%)")

    report.append("\n" + "=" * 70)

    # Print and save
    report_text = "\n".join(report)
    print(report_text)

    with open(f'{output_prefix}_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\nSaved: {output_prefix}_report.txt")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_analysis.py <S> [output_prefix]")
        print("  S: number of states")
        print("  output_prefix: prefix for output files (default: 'brain')")
        print("\nMake sure analysis CSV files are in the current directory.")
        sys.exit(1)

    S = int(sys.argv[1])
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'brain'

    print(f"Loading analysis files for S={S}...")
    files = load_analysis_files('')

    if not files:
        print("Error: No analysis CSV files found in current directory!")
        print("Run kernel_analyze.exe first to generate the data.")
        sys.exit(1)

    print(f"Found {len(files)} analysis files: {list(files.keys())}")

    # Generate all visualizations
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)

    if 'heatmap' in files:
        print("\n[1/7] State-Combination Heatmap...")
        plot_state_combo_heatmap(files['heatmap'], S, output_prefix)

    if 'transitions' in files:
        print("\n[2/7] State Transition Graph...")
        plot_transition_graph(files['transitions'], S, output_prefix)

    if 'state_actions' in files:
        print("\n[3/7] State Behavior Profiles...")
        plot_state_profiles(files['state_actions'], output_prefix)

    if 'pushes' in files and 'state_actions' in files:
        print("\n[4/7] Push Analysis...")
        plot_push_analysis(files['pushes'], files['state_actions'], S, output_prefix)

    if 'genes' in files and 'heatmap' in files:
        print("\n[5/7] Memory Usage Analysis...")
        analyze_memory_usage(files['genes'], files['heatmap'], S, output_prefix)

    if 'combo_triggers' in files:
        print("\n[6/7] Combo Trigger Analysis...")
        plot_combo_triggers(files['combo_triggers'], output_prefix)

    if 'transitions' in files:
        print("\n[7/7] State Communities...")
        find_state_communities(files['transitions'], S, output_prefix)

    # Summary report
    print("\n" + "=" * 50)
    print("SUMMARY REPORT")
    print("=" * 50)
    create_summary_report(files, S, output_prefix)

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE!")
    print("=" * 50)

if __name__ == '__main__':
    main()
