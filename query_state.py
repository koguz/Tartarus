#!/usr/bin/env python3
"""
Query detailed information about a specific state.

Usage:
    python query_state.py <state_number> [analysis_prefix]

Example:
    python query_state.py 82
    python query_state.py 43 analysis
"""

import json
import sys
from collections import Counter

def load_data(prefix='analysis'):
    """Load all analysis data."""
    with open(f'{prefix}_transitions.json', 'r') as f:
        transitions = json.load(f)
    with open(f'{prefix}_state_stats.json', 'r') as f:
        state_stats = json.load(f)
    return transitions, state_stats


def query_state(state_id, prefix='analysis'):
    """Query all information about a specific state."""
    transitions, state_stats = load_data(prefix)

    state_key = str(state_id)

    print("=" * 70)
    print(f"STATE {state_id} ANALYSIS")
    print("=" * 70)

    # Get state stats
    stats = state_stats.get(state_key, {})

    if not stats:
        print(f"State {state_id} was never visited.")
        return

    # Basic stats
    visit_count = stats.get('visit_count', 0)
    print(f"\nVisit count: {visit_count:,}")

    # Action distribution
    print("\n" + "-" * 40)
    print("ACTION DISTRIBUTION")
    print("-" * 40)
    action_counts = stats.get('action_counts', {})
    total_actions = sum(action_counts.values())

    for action in ['push', 'forward', 'turn_left', 'turn_right']:
        count = action_counts.get(action, 0)
        pct = 100 * count / total_actions if total_actions > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {action:<12}: {count:>10,} ({pct:>5.1f}%) {bar}")

    # Dominant action
    if action_counts:
        dominant = max(action_counts.items(), key=lambda x: x[1])[0]
        print(f"\n  Dominant action: {dominant}")

    # Outgoing edges (transitions FROM this state)
    print("\n" + "-" * 40)
    print("OUTGOING TRANSITIONS (from this state)")
    print("-" * 40)

    outgoing = []
    for trans_key, weight in transitions.items():
        from_s, to_s = trans_key.split('->')
        if from_s == state_key:
            outgoing.append((int(to_s), weight))

    outgoing.sort(key=lambda x: -x[1])
    total_out = sum(w for _, w in outgoing)

    print(f"  Total outgoing transitions: {total_out:,}")
    print(f"  Unique destination states: {len(outgoing)}")
    print(f"\n  {'To State':<12} {'Weight':>12} {'Percent':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*10}")

    for to_state, weight in outgoing[:15]:  # Top 15
        pct = 100 * weight / total_out if total_out > 0 else 0
        print(f"  {to_state:<12} {weight:>12,} {pct:>9.1f}%")

    if len(outgoing) > 15:
        print(f"  ... and {len(outgoing) - 15} more destinations")

    # Incoming edges (transitions TO this state)
    print("\n" + "-" * 40)
    print("INCOMING TRANSITIONS (to this state)")
    print("-" * 40)

    incoming = []
    for trans_key, weight in transitions.items():
        from_s, to_s = trans_key.split('->')
        if to_s == state_key:
            incoming.append((int(from_s), weight))

    incoming.sort(key=lambda x: -x[1])
    total_in = sum(w for _, w in incoming)

    print(f"  Total incoming transitions: {total_in:,}")
    print(f"  Unique source states: {len(incoming)}")
    print(f"\n  {'From State':<12} {'Weight':>12} {'Percent':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*10}")

    for from_state, weight in incoming[:15]:  # Top 15
        pct = 100 * weight / total_in if total_in > 0 else 0
        print(f"  {from_state:<12} {weight:>12,} {pct:>9.1f}%")

    if len(incoming) > 15:
        print(f"  ... and {len(incoming) - 15} more sources")

    # Self-loop check
    self_loop = transitions.get(f"{state_key}->{state_key}", 0)
    if self_loop > 0:
        print(f"\n  Self-loop weight: {self_loop:,} ({100*self_loop/total_out:.1f}% of outgoing)")

    # Average perception (what the agent "sees" in this state)
    print("\n" + "-" * 40)
    print("AVERAGE PERCEPTION (what agent sees)")
    print("-" * 40)

    combo_counts = stats.get('combination_counts', {})
    if combo_counts:
        # Parse combinations and compute average perception
        # Combination is base-3 encoding of 8 cells: 0=empty, 1=box, 2=wall
        total_combos = sum(combo_counts.values())

        # Initialize average grid (3x3, center is agent)
        # Positions: 0=front-left, 1=front, 2=front-right, 3=left, 4=right, 5=back-left, 6=back, 7=back-right
        avg_perception = [0.0] * 8  # Average value for each of 8 cells

        for combo_str, count in combo_counts.items():
            combo = int(combo_str)
            # Decode base-3 to get 8 cell values
            cells = []
            temp = combo
            for _ in range(8):
                cells.append(temp % 3)
                temp //= 3

            # Weighted average
            for i, cell_val in enumerate(cells):
                avg_perception[i] += cell_val * count / total_combos

        # Display as 3x3 grid
        # Layout:  [0] [1] [2]
        #          [3] [A] [4]
        #          [5] [6] [7]
        print(f"\n  Unique combinations seen: {len(combo_counts)}")
        print(f"  Total observations: {total_combos:,}")
        print(f"\n  Average cell values (0=empty, 1=box, 2=wall):")
        print(f"\n       Front")
        print(f"    [{avg_perception[0]:4.2f}] [{avg_perception[1]:4.2f}] [{avg_perception[2]:4.2f}]")
        print(f"    [{avg_perception[3]:4.2f}] [ A  ] [{avg_perception[4]:4.2f}]")
        print(f"    [{avg_perception[5]:4.2f}] [{avg_perception[6]:4.2f}] [{avg_perception[7]:4.2f}]")
        print(f"       Back")

        # Most common combinations
        print(f"\n  Top 5 most common combinations:")
        sorted_combos = sorted(combo_counts.items(), key=lambda x: -x[1])
        for combo_str, count in sorted_combos[:5]:
            combo = int(combo_str)
            pct = 100 * count / total_combos
            # Decode for display
            cells = []
            temp = combo
            for _ in range(8):
                cells.append(temp % 3)
                temp //= 3
            cell_chars = ['.' if c == 0 else 'B' if c == 1 else '#' for c in cells]
            grid_str = f"{cell_chars[0]}{cell_chars[1]}{cell_chars[2]}/{cell_chars[3]}A{cell_chars[4]}/{cell_chars[5]}{cell_chars[6]}{cell_chars[7]}"
            print(f"    Combo {combo:>4}: {count:>8,} ({pct:>5.1f}%)  {grid_str}")

        # Return data structure for programmatic use
        perception_grid = [
            [avg_perception[0], avg_perception[1], avg_perception[2]],
            [avg_perception[3], None, avg_perception[4]],  # None = agent position
            [avg_perception[5], avg_perception[6], avg_perception[7]]
        ]

        print(f"\n  Perception as nested list (for code):")
        print(f"    [[{avg_perception[0]:.3f}, {avg_perception[1]:.3f}, {avg_perception[2]:.3f}],")
        print(f"     [{avg_perception[3]:.3f}, None,  {avg_perception[4]:.3f}],")
        print(f"     [{avg_perception[5]:.3f}, {avg_perception[6]:.3f}, {avg_perception[7]:.3f}]]")
    else:
        print("  No combination data available")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python query_state.py <state_number> [analysis_prefix]")
        print("Example: python query_state.py 82")
        sys.exit(1)

    state_id = int(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else 'analysis'

    query_state(state_id, prefix)
