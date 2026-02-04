"""
Ablation Study: Test impact of removing specific states.

When the agent would transition to an ablated state, it instead
transitions to the initial state. This tests whether certain
state clusters are critical for edge cases.
"""

import argparse
import json
from collections import defaultdict

# Constants (matching analyze_agent.py / CUDA code)
C = 383  # Number of valid combinations

# Direction vectors: N=0, S=1, E=2, W=3
DIR_X = [0, 0, 1, -1]
DIR_Y = [-1, 1, 0, 0]

# Scan patterns for each direction (8 cells around agent)
SCAN_X = [
    [0, 1, 1, 1, 0, -1, -1, -1],     # N
    [0, -1, -1, -1, 0, 1, 1, 1],     # S
    [1, 1, 0, -1, -1, -1, 0, 1],     # E
    [-1, -1, 0, 1, 1, 1, 0, -1]      # W
]
SCAN_Y = [
    [-1, -1, 0, 1, 1, 1, 0, -1],     # N
    [1, 1, 0, -1, -1, -1, 0, 1],     # S
    [0, 1, 1, 1, 0, -1, -1, -1],     # E
    [0, -1, -1, -1, 0, 1, 1, 1]      # W
]

# Turn mappings
TURN_LEFT = [3, 2, 0, 1]   # N->W, S->E, E->N, W->S
TURN_RIGHT = [2, 3, 1, 0]  # N->E, S->W, E->S, W->N

# Inner 4x4 grid coordinates in 6x6 board
COORD = [7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28]

# Powers of 3 for base-3 encoding
POW3 = [1, 3, 9, 27, 81, 243, 729, 2187]

# Valid combination indices (from analyze_agent.py)
IIDX = [0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546]

# Build inverted index lookup
IX = [0] * 6561
for k, v in enumerate(IIDX):
    IX[v] = k

IIDX_SET = set(IIDX)


def load_agent(filepath, num_states=128):
    """Load agent from file."""
    with open(filepath, 'r') as f:
        data = list(map(int, f.read().split()))

    G = num_states * C + 1
    if len(data) != 2 * G:
        raise ValueError(f"Expected {2*G} values for {num_states} states with C={C}, got {len(data)}")

    actions = []
    next_states = []
    for i in range(G):
        actions.append(data[2*i])
        next_states.append(data[2*i + 1])

    initial_state = next_states[-1]
    return actions, next_states, initial_state


def load_boards(filepath):
    """Load board configurations."""
    boards = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) >= 16:
                board = [0] * 36
                for j in range(16):
                    coord_idx = 7 + (j // 4) * 6 + (j % 4)
                    board[coord_idx] = 1 if line[j] == '1' else 0
                boards.append(board)
    return boards


def find_starting_positions(board):
    """Find empty positions in inner 4x4 grid."""
    positions = []
    for i, coord in enumerate(COORD):
        if board[coord] == 0:
            y = coord // 6
            x = coord % 6
            positions.append((x, y))
    return positions


def compute_sensor_input(board, cpx, cpy, direction):
    """Compute sensor input for agent."""
    cc = 0
    for m in range(8):
        sx = cpx + SCAN_X[direction][m]
        sy = cpy + SCAN_Y[direction][m]
        if sx < 0 or sy < 0 or sx >= 6 or sy >= 6:
            val = 2
        else:
            val = board[sy * 6 + sx]
        cc += POW3[m] * val
    return IX[cc]


def run_agent_with_ablation(actions, next_states, initial_state, board_template,
                            start_x, start_y, direction, ablated_states):
    """
    Run agent on a board with certain states ablated.
    When transitioning to an ablated state, use initial_state instead.
    """
    board = board_template.copy()
    cpx, cpy = start_x, start_y
    dir_ = direction
    cs = initial_state

    # If initial state is ablated, we have a problem - but run anyway
    if cs in ablated_states:
        cs = initial_state  # No change, but log this case

    ablation_count = 0

    for move in range(80):
        combo_idx = compute_sensor_input(board, cpx, cpy, dir_)
        gene_idx = cs * C + combo_idx
        action = actions[gene_idx]
        next_state = next_states[gene_idx]

        # ABLATION: redirect ablated states to initial state
        if next_state in ablated_states:
            next_state = initial_state
            ablation_count += 1

        # Execute action
        if action == 0:  # Forward
            cx = cpx + DIR_X[dir_]
            cy = cpy + DIR_Y[dir_]
            if 0 <= cx < 6 and 0 <= cy < 6:
                if board[cy * 6 + cx] == 0:
                    cpx, cpy = cx, cy
                elif board[cy * 6 + cx] == 1:
                    dx = cx + DIR_X[dir_]
                    dy = cy + DIR_Y[dir_]
                    if 0 <= dx < 6 and 0 <= dy < 6 and board[dy * 6 + dx] == 0:
                        board[cy * 6 + cx] = 0
                        board[dy * 6 + dx] = 1
                        cpx, cpy = cx, cy
        elif action == 1:
            dir_ = TURN_LEFT[dir_]
        else:
            dir_ = TURN_RIGHT[dir_]

        cs = next_state

    # Calculate fitness
    fitness = 0
    for i in range(36):
        if board[i] == 1:
            x, y = i % 6, i // 6
            # REPLACE Lines 119-122 with this:
            if x == 0 or x == 5:
                fitness += 1
            if y == 0 or y == 5:
                fitness += 1

    return fitness, ablation_count


def run_ablation_study(agent_path, boards_path, ablated_states, num_states=128):
    """Run full ablation study."""
    print(f"Loading agent from '{agent_path}'...")
    actions, next_states, initial_state = load_agent(agent_path, num_states)

    print(f"Loading boards from '{boards_path}'...")
    boards = load_boards(boards_path)
    print(f"  Loaded {len(boards)} boards")

    print(f"\nAblated states: {sorted(ablated_states)}")
    print(f"  ({len(ablated_states)} states will redirect to initial state {initial_state})")

    if initial_state in ablated_states:
        print(f"  WARNING: Initial state {initial_state} is in ablated set!")

    # Run on all configurations
    fitness_distribution = defaultdict(int)
    total_fitness = 0
    config_count = 0
    total_ablations = 0

    print(f"\nRunning ablation study on {len(boards)} boards...")
    for board_idx, board in enumerate(boards):
        positions = find_starting_positions(board)

        for start_x, start_y in positions:
            for direction in range(4):
                fitness, ablation_count = run_agent_with_ablation(
                    actions, next_states, initial_state,
                    board, start_x, start_y, direction,
                    ablated_states
                )

                total_fitness += fitness
                config_count += 1
                fitness_distribution[fitness] += 1
                total_ablations += ablation_count

        if (board_idx + 1) % 500 == 0:
            print(f"  Processed {board_idx + 1}/{len(boards)} boards...")

    avg_fitness = total_fitness / config_count
    avg_ablations = total_ablations / config_count

    return {
        'avg_fitness': avg_fitness,
        'total_configs': config_count,
        'fitness_distribution': dict(fitness_distribution),
        'total_ablations': total_ablations,
        'avg_ablations_per_run': avg_ablations,
        'ablated_states': sorted(ablated_states),
        'num_ablated': len(ablated_states)
    }


def print_results(results, baseline_avg=None):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print("="*60)

    print(f"\nAblated {results['num_ablated']} states: {results['ablated_states']}")
    print(f"Total configurations: {results['total_configs']:,}")
    print(f"Total ablation redirects: {results['total_ablations']:,}")
    print(f"Avg ablations per run: {results['avg_ablations_per_run']:.2f}")

    print(f"\n{'='*60}")
    print("SCORE DISTRIBUTION")
    print("="*60)

    avg = results['avg_fitness']
    print(f"\nAverage fitness: {avg:.4f}")

    if baseline_avg:
        diff = avg - baseline_avg
        pct = 100 * diff / baseline_avg
        print(f"Baseline average: {baseline_avg:.4f}")
        print(f"Difference: {diff:+.4f} ({pct:+.2f}%)")

    dist = results['fitness_distribution']
    total = results['total_configs']

    print(f"\n{'Score':<8} {'Count':>10} {'Percent':>10} {'Cumulative':>12}")
    print("-" * 42)

    cumulative = 0
    for score in sorted(dist.keys(), reverse=True):
        count = dist[score]
        pct = 100 * count / total
        cumulative += count
        cum_pct = 100 * cumulative / total
        print(f"{score:<8} {count:>10,} {pct:>9.2f}% {cum_pct:>11.2f}%")


def load_state_clusters(filepath):
    """Load state clusters from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Try infomap first, then spectral
    if 'infomap' in data and 'clusters' in data['infomap']:
        clusters = data['infomap']['clusters']
    elif 'spectral' in data and 'clusters' in data['spectral']:
        clusters = data['spectral']['clusters']
    else:
        raise ValueError("No valid cluster data found")

    # Group states by cluster
    cluster_to_states = defaultdict(list)
    for state_str, cluster_id in clusters.items():
        cluster_to_states[cluster_id].append(int(state_str))

    return dict(cluster_to_states)


def main():
    parser = argparse.ArgumentParser(
        description='Ablation study: test impact of removing states'
    )
    parser.add_argument(
        '--agent', '-a',
        type=str,
        default='best/b-D2-4096-128-3000-1.txt',
        help='Agent file path'
    )
    parser.add_argument(
        '--boards', '-b',
        type=str,
        default='realboard.txt',
        help='Boards file path'
    )
    parser.add_argument(
        '--states', '-s',
        type=int,
        nargs='+',
        help='States to ablate (space-separated list)'
    )
    parser.add_argument(
        '--cluster', '-c',
        type=int,
        help='Ablate all states in this cluster (requires --cluster-file)'
    )
    parser.add_argument(
        '--cluster-file',
        type=str,
        default='state_clusters.json',
        help='State clusters JSON file'
    )
    parser.add_argument(
        '--exclude-cluster',
        type=int,
        help='Ablate all states EXCEPT those in this cluster'
    )
    parser.add_argument(
        '--num-states', '-n',
        type=int,
        default=128,
        help='Number of states in agent (default: 128)'
    )
    parser.add_argument(
        '--baseline',
        type=float,
        help='Baseline average fitness for comparison'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--run-baseline',
        action='store_true',
        help='Run baseline (no ablation) first'
    )

    args = parser.parse_args()

    # Determine which states to ablate
    ablated_states = set()

    if args.states:
        ablated_states = set(args.states)
    elif args.cluster is not None or args.exclude_cluster is not None:
        cluster_to_states = load_state_clusters(args.cluster_file)
        print(f"\nLoaded clusters from '{args.cluster_file}':")
        for cid, states in sorted(cluster_to_states.items()):
            print(f"  Cluster {cid}: {len(states)} states")

        if args.cluster is not None:
            ablated_states = set(cluster_to_states.get(args.cluster, []))
            print(f"\nAblating cluster {args.cluster}")
        elif args.exclude_cluster is not None:
            # Ablate all states NOT in the specified cluster
            keep_states = set(cluster_to_states.get(args.exclude_cluster, []))
            all_states = set()
            for states in cluster_to_states.values():
                all_states.update(states)
            ablated_states = all_states - keep_states
            print(f"\nKeeping only cluster {args.exclude_cluster}, ablating all others")
    else:
        print("Error: Must specify --states, --cluster, or --exclude-cluster")
        return

    if not ablated_states:
        print("Error: No states to ablate")
        return

    # Run baseline if requested
    baseline_avg = args.baseline
    if args.run_baseline:
        print("\n" + "="*60)
        print("RUNNING BASELINE (NO ABLATION)")
        print("="*60)
        baseline_results = run_ablation_study(
            args.agent, args.boards, set(), args.num_states
        )
        baseline_avg = baseline_results['avg_fitness']
        print(f"\nBaseline average fitness: {baseline_avg:.4f}")

    # Run ablation study
    print("\n" + "="*60)
    print("RUNNING ABLATION STUDY")
    print("="*60)
    results = run_ablation_study(
        args.agent, args.boards, ablated_states, args.num_states
    )

    # Print results
    print_results(results, baseline_avg)

    # Save results
    if args.output:
        results['baseline_avg'] = baseline_avg
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to '{args.output}'")


if __name__ == '__main__':
    main()
