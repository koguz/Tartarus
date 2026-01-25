#!/usr/bin/env python3
"""
Analyze FSM agent behavior on Tartarus boards.

This script:
1. Loads an FSM agent and runs it on all 74760 board configurations
2. Builds a state transition graph with edge weights
3. Collects per-state statistics (actions, combinations seen)
4. Saves state sequences for pattern analysis
"""

import numpy as np
from collections import defaultdict
import json
import pickle

# Constants (matching CUDA code)
NUM_BOARDS = 1869
CONFIGS_PER_BOARD = 40
TOTAL_CONFIGS = 74760  # 1869 * 40
NUM_STATES = 128
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

# Valid combination indices (inverted index mapping)
IIDX = [0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546]

# Build inverted index lookup
IX = [0] * 6561
for k, v in enumerate(IIDX):
    IX[v] = k

# Set for fast validity checking
IIDX_SET = set(IIDX)


def decode_combination(combo_raw):
    """Decode a raw combination value to 8 cell values (0=empty, 1=box, 2=wall)."""
    cells = []
    for i in range(8):
        cells.append(combo_raw % 3)
        combo_raw //= 3
    return cells


def describe_combination(combo_idx):
    """
    Get human-readable description of what the agent sees.

    Due to coordinate transposition (board[col*6+row] instead of board[row*6+col]),
    the agent's internal positions map to PHYSICAL directions as follows:
        Agent pos 0 (code "F")  -> Physical Left (L)
        Agent pos 1 (code "FR") -> Physical Back-Left (BL)
        Agent pos 2 (code "R")  -> Physical Back (B)
        Agent pos 3 (code "BR") -> Physical Back-Right (BR)
        Agent pos 4 (code "B")  -> Physical Right (R)
        Agent pos 5 (code "BL") -> Physical Front-Right (FR)
        Agent pos 6 (code "L")  -> Physical Front (F)
        Agent pos 7 (code "FL") -> Physical Front-Left (FL)

    We use PHYSICAL direction names in the output.
    """
    if combo_idx >= len(IIDX):
        return "invalid"
    combo_raw = IIDX[combo_idx]
    cells = decode_combination(combo_raw)
    # Physical direction names for each agent position
    phys_names = ['L', 'BL', 'B', 'BR', 'R', 'FR', 'F', 'FL']
    val_names = ['_', 'B', 'W']  # empty, box, wall
    return ''.join(f"{phys_names[i]}:{val_names[cells[i]]}" for i in range(8) if cells[i] != 0)


def has_box_in_front(combo_idx):
    """Check if the combination has a box directly in front."""
    if combo_idx >= len(IIDX):
        return False
    combo_raw = IIDX[combo_idx]
    return (combo_raw % 3) == 1  # Position 0 is front


def load_agent(filepath, num_states=128):
    """Load agent from file. Returns (actions, next_states, initial_state)."""
    with open(filepath, 'r') as f:
        data = list(map(int, f.read().split()))

    G = num_states * C + 1
    assert len(data) == 2 * G, f"Expected {2*G} values for {num_states} states, got {len(data)}"

    actions = []
    next_states = []
    for i in range(G):
        actions.append(data[2*i])
        next_states.append(data[2*i + 1])

    initial_state = next_states[-1]  # Last gene stores initial state
    return actions, next_states, initial_state


def load_boards(filepath):
    """Load board configurations from realboard.txt."""
    boards = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) >= 16:
                # Create 6x6 board (all zeros = empty)
                board = [0] * 36
                # Fill inner 4x4 grid
                for j in range(16):
                    coord_idx = 7 + (j // 4) * 6 + (j % 4)
                    board[coord_idx] = 1 if line[j] == '1' else 0
                boards.append(board)
    return boards


def find_starting_positions(board):
    """Find all empty positions in inner 4x4 grid."""
    positions = []
    for i, coord in enumerate(COORD):
        if board[coord] == 0:
            y = coord // 6
            x = coord % 6
            positions.append((x, y))
    return positions


def compute_sensor_input(board, cpx, cpy, direction):
    """Compute the sensor input (combination index) for agent."""
    cc = 0
    for m in range(8):
        sx = cpx + SCAN_X[direction][m]
        sy = cpy + SCAN_Y[direction][m]
        if sx < 0 or sy < 0 or sx >= 6 or sy >= 6:
            val = 2  # Wall (out of bounds)
        else:
            val = board[sy * 6 + sx]
        cc += POW3[m] * val
    if cc not in IIDX_SET:
        print(f"WARNING: Invalid cc={cc} at ({cpx},{cpy}) dir={direction}")
    return IX[cc], cc  # Return both mapped index and raw value


def run_agent_on_board(actions, next_states, initial_state, board_template, start_x, start_y, direction):
    """
    Run agent on a board for 80 steps.
    Returns: fitness, state_sequence, step_details
    """
    # Copy board
    board = board_template.copy()
    cpx, cpy = start_x, start_y
    dir_ = direction
    cs = initial_state

    state_sequence = [cs]
    step_details = []  # List of (state, combo_idx, combo_raw, action_taken, action_type)

    for move in range(80):
        # Compute sensor input
        combo_idx, combo_raw = compute_sensor_input(board, cpx, cpy, dir_)

        # Get gene
        gene_idx = cs * C + combo_idx
        action = actions[gene_idx]
        next_state = next_states[gene_idx]

        # Determine action type
        if action == 0:  # Forward
            # Check if there's a box in front
            box_in_front = has_box_in_front(combo_idx)
            if box_in_front:
                action_type = 'push'
            else:
                action_type = 'forward'
        elif action == 1:
            action_type = 'turn_left'
        else:
            action_type = 'turn_right'

        step_details.append({
            'state': cs,
            'combo_idx': combo_idx,
            'combo_raw': combo_raw,
            'action': action,
            'action_type': action_type,
            'next_state': next_state
        })

        # Execute action
        if action == 0:  # Forward
            cx = cpx + DIR_X[dir_]
            cy = cpy + DIR_Y[dir_]
            if 0 <= cx < 6 and 0 <= cy < 6:
                if board[cy * 6 + cx] == 0: 
                    # Move to empty cell
                    cpx, cpy = cx, cy
                elif board[cy * 6 + cx] == 1:
                    # Try to push box
                    dx = cx + DIR_X[dir_]
                    dy = cy + DIR_Y[dir_]
                    if 0 <= dx < 6 and 0 <= dy < 6 and board[dy * 6 + dx] == 0:
                        # Push successful
                        board[cy * 6 + cx] = 0
                        board[dy * 6 + dx] = 1
                        cpx, cpy = cx, cy
        elif action == 1:  # Turn left
            dir_ = TURN_LEFT[dir_]
        else:  # Turn right
            dir_ = TURN_RIGHT[dir_]

        cs = next_state
        state_sequence.append(cs)

    # Calculate fitness (boxes on edge)
    fitness = 0
    for i in range(6):
        for j in range(6):
            if board[i * 6 + j] == 1:
                if i == 0 or i == 5:
                    fitness += 1
                if j == 0 or j == 5:
                    fitness += 1

    return fitness, state_sequence, step_details


def analyze_agent(agent_path, boards_path, output_prefix='analysis', num_states=128):
    """Main analysis function."""
    print("Loading agent...")
    actions, next_states, initial_state = load_agent(agent_path, num_states)
    print(f"  Num states: {num_states}")
    print(f"  Initial state: {initial_state}")
    print(f"  Total genes: {len(actions)}")

    print("Loading boards...")
    boards = load_boards(boards_path)
    print(f"  Loaded {len(boards)} boards")

    # Initialize data structures
    # State transition graph: transition_counts[from_state][to_state] = count
    transition_counts = defaultdict(lambda: defaultdict(int))

    # Per-state statistics
    state_stats = {s: {
        'action_counts': {'forward': 0, 'push': 0, 'turn_left': 0, 'turn_right': 0},
        'combo_counts': defaultdict(int),  # combo_idx -> count
        'visit_count': 0
    } for s in range(num_states)}

    # All state sequences (for pattern analysis)
    all_sequences = []

    # Fitness distribution (score -> count)
    fitness_distribution = defaultdict(int)

    # Combination transition graph: combo_transitions[from_combo][to_combo] = count
    # Self-edges (from_combo == to_combo) are allowed
    combo_transitions = defaultdict(lambda: defaultdict(int))

    # Per-combination statistics: track states and actions for each combination
    combo_stats = {c: {
        'state_counts': defaultdict(int),  # state -> count (which states visited this combo)
        'action_counts': {'forward': 0, 'push': 0, 'turn_left': 0, 'turn_right': 0},
        'visit_count': 0
    } for c in range(C)}

    # Run all configurations
    total_fitness = 0
    config_count = 0

    failed_configs = []

    print("Running agent on all configurations...")
    for board_idx, board in enumerate(boards):
        # Find empty positions in inner 4x4
        positions = find_starting_positions(board)

        for pos_idx, (start_x, start_y) in enumerate(positions):
            for direction in range(4):
                fitness, state_seq, step_details = run_agent_on_board(
                    actions, next_states, initial_state,
                    board, start_x, start_y, direction
                )

                total_fitness += fitness
                config_count += 1
                fitness_distribution[fitness] += 1

                # failed ones.
                if fitness <= 6:
                    failed_configs.append({
                        'board_idx': board_idx,
                        'start_pos': [start_x, start_y],
                        'direction': direction,
                        'score': fitness,
                        'board_layout': board  # Optional: save the full board array
                    })

                # Store sequence
                all_sequences.append(state_seq)

                # Update transition counts
                for i in range(len(state_seq) - 1):
                    from_state = state_seq[i]
                    to_state = state_seq[i + 1]
                    if from_state != to_state:  # Ignore self-loops as requested
                        transition_counts[from_state][to_state] += 1

                # Update per-state statistics
                for detail in step_details:
                    s = detail['state']
                    state_stats[s]['visit_count'] += 1
                    state_stats[s]['action_counts'][detail['action_type']] += 1
                    state_stats[s]['combo_counts'][detail['combo_idx']] += 1

                # Update combination transition counts and per-combination statistics
                for i in range(len(step_details)):
                    detail = step_details[i]
                    combo_idx = detail['combo_idx']

                    # Update per-combination stats
                    combo_stats[combo_idx]['visit_count'] += 1
                    combo_stats[combo_idx]['state_counts'][detail['state']] += 1
                    combo_stats[combo_idx]['action_counts'][detail['action_type']] += 1

                    # Track transition to next combination (if there is a next step)
                    if i < len(step_details) - 1:
                        next_combo_idx = step_details[i + 1]['combo_idx']
                        combo_transitions[combo_idx][next_combo_idx] += 1

        if (board_idx + 1) % 200 == 0:
            print(f"  Processed {board_idx + 1}/{len(boards)} boards ({config_count} configs)")

    avg_fitness = total_fitness / config_count
    print(f"\nResults:")
    print(f"  Total configurations: {config_count}")
    print(f"  Average fitness: {avg_fitness:.4f}")

    # Fitness distribution
    print(f"\n" + "="*60)
    print("FITNESS DISTRIBUTION")
    print("="*60)
    max_score = max(fitness_distribution.keys()) if fitness_distribution else 0
    print(f"{'Score':>6} {'Count':>10} {'Percentage':>12} {'Cumulative':>12}")
    print("-" * 42)
    cumulative = 0
    for score in range(max_score, -1, -1):
        count = fitness_distribution.get(score, 0)
        cumulative += count
        pct = 100 * count / config_count
        cum_pct = 100 * cumulative / config_count
        if count > 0:
            print(f"{score:>6} {count:>10} {pct:>11.2f}% {cum_pct:>11.2f}%")

    # Save results
    print("\nSaving results...")

    # 1. State transition graph (as edge list for visualization)
    edges = []
    for from_state, targets in transition_counts.items():
        for to_state, weight in targets.items():
            edges.append({
                'source': from_state,
                'target': to_state,
                'weight': weight
            })

    with open(f'{output_prefix}_transitions.json', 'w') as f:
        json.dump({
            'nodes': list(range(num_states)),
            'edges': edges,
            'num_states': num_states,
            'total_configs': config_count
        }, f, indent=2)
    print(f"  Saved transition graph to {output_prefix}_transitions.json")

    # 1b. Fitness distribution
    with open(f'{output_prefix}_fitness_dist.json', 'w') as f:
        json.dump({
            'distribution': dict(fitness_distribution),
            'total_configs': config_count,
            'average_fitness': avg_fitness
        }, f, indent=2)
    print(f"  Saved fitness distribution to {output_prefix}_fitness_dist.json")

    # Save failed configurations
    with open(f'{output_prefix}_failed_boards.json', 'w') as f:
        json.dump(failed_configs, f, indent=2)
    print(f"Saved {len(failed_configs)} failed configurations to {output_prefix}_failed_boards.json")

    # 2. Per-state statistics
    stats_output = {}
    for s in range(num_states):
        stats = state_stats[s]
        # Convert defaultdict to regular dict for JSON
        combo_counts = dict(stats['combo_counts'])

        # Find most common combinations for this state
        top_combos = sorted(combo_counts.items(), key=lambda x: -x[1])[:10]

        stats_output[s] = {
            'visit_count': stats['visit_count'],
            'action_counts': stats['action_counts'],
            'top_combinations': [
                {
                    'combo_idx': c[0],
                    'count': c[1],
                    'description': describe_combination(c[0])
                }
                for c in top_combos
            ]
        }

    with open(f'{output_prefix}_state_stats.json', 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"  Saved state statistics to {output_prefix}_state_stats.json")

    # 3. State sequences (binary format for efficiency)
    with open(f'{output_prefix}_sequences.pkl', 'wb') as f:
        pickle.dump(all_sequences, f)
    print(f"  Saved {len(all_sequences)} state sequences to {output_prefix}_sequences.pkl")

    # 4. Combination transition graph (as edge list for visualization)
    combo_edges = []
    for from_combo, targets in combo_transitions.items():
        for to_combo, weight in targets.items():
            combo_edges.append({
                'source': from_combo,
                'target': to_combo,
                'weight': weight
            })

    with open(f'{output_prefix}_combo_transitions.json', 'w') as f:
        json.dump({
            'nodes': list(range(C)),
            'edges': combo_edges,
            'num_combinations': C,
            'total_configs': config_count
        }, f, indent=2)
    print(f"  Saved combination transition graph to {output_prefix}_combo_transitions.json")

    # 5. Per-combination statistics
    combo_stats_output = {}
    for c in range(C):
        stats = combo_stats[c]
        if stats['visit_count'] == 0:
            continue  # Skip unvisited combinations

        # Convert defaultdict to regular dict for JSON
        state_counts = dict(stats['state_counts'])

        # Find most common states for this combination
        top_states = sorted(state_counts.items(), key=lambda x: -x[1])[:10]

        combo_stats_output[c] = {
            'combo_raw': IIDX[c],
            'description': describe_combination(c),
            'visit_count': stats['visit_count'],
            'action_counts': stats['action_counts'],
            'num_states': len(state_counts),  # How many different states visited this combo
            'top_states': [
                {'state': s[0], 'count': s[1]}
                for s in top_states
            ],
            'all_states': state_counts  # Full mapping of state -> count
        }

    with open(f'{output_prefix}_combo_stats.json', 'w') as f:
        json.dump(combo_stats_output, f, indent=2)
    print(f"  Saved combination statistics to {output_prefix}_combo_stats.json")

    # 6. Summary report
    print("\n" + "="*60)
    print("STATE SUMMARY")
    print("="*60)

    # Sort states by visit count
    sorted_states = sorted(range(num_states), key=lambda s: -state_stats[s]['visit_count'])

    print(f"\nTop 20 most visited states:")
    print(f"{'State':>6} {'Visits':>10} {'Forward':>8} {'Push':>8} {'TurnL':>8} {'TurnR':>8}")
    print("-" * 58)

    for s in sorted_states[:20]:
        stats = state_stats[s]
        total_visits = stats['visit_count']

        print(f"{s:>6} {total_visits:>10} {stats['action_counts']['forward']:>8} "
              f"{stats['action_counts']['push']:>8} {stats['action_counts']['turn_left']:>8} "
              f"{stats['action_counts']['turn_right']:>8}")

    # Find states that push most
    print(f"\nTop 10 'pusher' states (by push count):")
    pusher_states = sorted(range(num_states), key=lambda s: -state_stats[s]['action_counts']['push'])
    for s in pusher_states[:10]:
        stats = state_stats[s]
        push_count = stats['action_counts']['push']
        total = stats['visit_count']
        if total > 0:
            print(f"  State {s}: {push_count} pushes ({100*push_count/total:.1f}% of {total} visits)")

    print(f"\n" + "="*60)
    print("TRANSITION GRAPH SUMMARY")
    print("="*60)

    # Count outgoing edges per state
    out_degree = {s: len(transition_counts[s]) for s in range(num_states)}
    in_degree = defaultdict(int)
    for from_s, targets in transition_counts.items():
        for to_s in targets:
            in_degree[to_s] += 1

    print(f"\nStates with highest out-degree (most diverse transitions):")
    for s in sorted(range(num_states), key=lambda x: -out_degree[x])[:10]:
        print(f"  State {s}: {out_degree[s]} different target states")

    print(f"\nStates with highest in-degree (most commonly transitioned to):")
    for s in sorted(range(num_states), key=lambda x: -in_degree[x])[:10]:
        total_incoming = sum(transition_counts[from_s][s] for from_s in transition_counts if s in transition_counts[from_s])
        print(f"  State {s}: transitions from {in_degree[s]} states ({total_incoming} total transitions)")

    print(f"\n" + "="*60)
    print("COMBINATION GRAPH SUMMARY")
    print("="*60)

    # Count edges and self-loops
    total_combo_edges = len(combo_edges)
    self_loop_count = sum(1 for e in combo_edges if e['source'] == e['target'])
    self_loop_weight = sum(e['weight'] for e in combo_edges if e['source'] == e['target'])
    total_edge_weight = sum(e['weight'] for e in combo_edges)

    print(f"\nGraph statistics:")
    print(f"  Total edges: {total_combo_edges} ({self_loop_count} self-loops)")
    print(f"  Total transitions: {total_edge_weight} ({self_loop_weight} self-loop transitions)")

    # Visited combinations
    visited_combos = [c for c in range(C) if combo_stats[c]['visit_count'] > 0]
    print(f"  Visited combinations: {len(visited_combos)} / {C}")

    # Out-degree and in-degree for combinations
    combo_out_degree = {c: len(combo_transitions[c]) for c in range(C)}
    combo_in_degree = defaultdict(int)
    for from_c, targets in combo_transitions.items():
        for to_c in targets:
            combo_in_degree[to_c] += 1

    print(f"\nTop 10 most visited combinations:")
    print(f"{'Combo':>6} {'Visits':>10} {'#States':>8} {'Forward':>8} {'Push':>6} {'TurnL':>6} {'TurnR':>6}  Description")
    print("-" * 90)

    sorted_combos = sorted(visited_combos, key=lambda c: -combo_stats[c]['visit_count'])
    for c in sorted_combos[:10]:
        stats = combo_stats[c]
        desc = describe_combination(c)
        print(f"{c:>6} {stats['visit_count']:>10} {len(stats['state_counts']):>8} "
              f"{stats['action_counts']['forward']:>8} {stats['action_counts']['push']:>6} "
              f"{stats['action_counts']['turn_left']:>6} {stats['action_counts']['turn_right']:>6}  {desc}")

    print(f"\nTop 10 'pusher' combinations (by push count):")
    pusher_combos = sorted(visited_combos, key=lambda c: -combo_stats[c]['action_counts']['push'])
    for c in pusher_combos[:10]:
        stats = combo_stats[c]
        push_count = stats['action_counts']['push']
        total = stats['visit_count']
        desc = describe_combination(c)
        if total > 0:
            print(f"  Combo {c}: {push_count} pushes ({100*push_count/total:.1f}% of {total} visits) - {desc}")

    print(f"\nCombinations with most diverse state usage (many states visit same combo):")
    diverse_combos = sorted(visited_combos, key=lambda c: -len(combo_stats[c]['state_counts']))
    for c in diverse_combos[:10]:
        stats = combo_stats[c]
        desc = describe_combination(c)
        print(f"  Combo {c}: {len(stats['state_counts'])} different states - {desc}")

    print(f"\nCombinations with highest out-degree (lead to many different combos):")
    for c in sorted(visited_combos, key=lambda x: -combo_out_degree[x])[:10]:
        desc = describe_combination(c)
        print(f"  Combo {c}: {combo_out_degree[c]} target combinations - {desc}")

    return transition_counts, state_stats, all_sequences, combo_transitions, combo_stats


if __name__ == '__main__':
    import sys

    agent_path = sys.argv[1] if len(sys.argv) > 1 else 'best/b-D2-4096-128-3000-1.txt'
    boards_path = sys.argv[2] if len(sys.argv) > 2 else 'realboard.txt'
    output_prefix = sys.argv[3] if len(sys.argv) > 3 else 'analysis'
    num_states = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    analyze_agent(agent_path, boards_path, output_prefix, num_states)
