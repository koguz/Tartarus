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

    push = action_counts.get('push', 0)
    forward = action_counts.get('forward', 0)
    turn_left = action_counts.get('turn_left', 0)
    turn_right = action_counts.get('turn_right', 0)
    turn = turn_left + turn_right

    print(f"\n  Detailed breakdown:")
    for action in ['push', 'forward', 'turn_left', 'turn_right']:
        count = action_counts.get(action, 0)
        pct = 100 * count / total_actions if total_actions > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"    {action:<12}: {count:>10,} ({pct:>5.1f}%) {bar}")

    print(f"\n  Summary (push / forward / turn):")
    for label, count in [('push', push), ('forward', forward), ('turn', turn)]:
        pct = 100 * count / total_actions if total_actions > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"    {label:<12}: {count:>10,} ({pct:>5.1f}%) {bar}")

    # Dominant action (using combined turn)
    combined = {'push': push, 'forward': forward, 'turn': turn}
    dominant = max(combined.items(), key=lambda x: x[1])[0]
    print(f"\n  Dominant action: {dominant}")

    # Outgoing edges (transitions FROM this state)
    print("\n" + "-" * 40)
    print("OUTGOING TRANSITIONS (from this state)")
    print("-" * 40)

    edges = transitions.get('edges', [])

    outgoing = []
    for edge in edges:
        if edge['source'] == state_id:
            outgoing.append((edge['target'], edge['weight']))

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
    for edge in edges:
        if edge['target'] == state_id:
            incoming.append((edge['source'], edge['weight']))

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
    self_loop = 0
    for edge in edges:
        if edge['source'] == state_id and edge['target'] == state_id:
            self_loop = edge['weight']
            break
    if self_loop > 0:
        print(f"\n  Self-loop weight: {self_loop:,} ({100*self_loop/total_out:.1f}% of outgoing)")

    # Average perception (what the agent "sees" in this state)
    print("\n" + "-" * 40)
    print("AVERAGE PERCEPTION (what agent sees)")
    print("-" * 40)

    top_combos = stats.get('top_combinations', [])
    if top_combos:
        # Positions in the raw combo (base-3 encoding):
        # 0=front, 1=front-right, 2=right, 3=back-right, 4=back, 5=back-left, 6=left, 7=front-left
        # IIDX lookup table (same as in analyze_agent.py)
        IIDX = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,540,541,542,543,544,545,546,547,548,549,550,551,567,568,569,570,571,572,573,574,575,576,577,578,594,595,596,597,598,599,600,601,602,603,604,605,621,622,623,624,625,626,627,628,629,630,631,632,648,649,650,651,652,653,654,655,656,657,658,659,675,676,677,678,679,680,681,682,683,684,685,686]

        def decode_combo(combo_idx):
            if combo_idx >= len(IIDX):
                return [0] * 8
            combo_raw = IIDX[combo_idx]
            cells = []
            for _ in range(8):
                cells.append(combo_raw % 3)
                combo_raw //= 3
            return cells

        total_observations = sum(c['count'] for c in top_combos)

        # Compute probability of each cell type (empty/box/wall) for each position
        # prob[position][type] where type: 0=empty, 1=box, 2=wall
        prob = [[0.0, 0.0, 0.0] for _ in range(8)]

        for combo_data in top_combos:
            cells = decode_combo(combo_data['combo_idx'])
            weight = combo_data['count'] / total_observations
            for i, val in enumerate(cells):
                prob[i][val] += weight

        print(f"\n  Unique combinations in top list: {len(top_combos)}")
        print(f"  Total observations: {total_observations:,}")

        # Display as 3x3 grid with probabilities
        # Mapping: positions are 0=F, 1=FR, 2=R, 3=BR, 4=B, 5=BL, 6=L, 7=FL
        # Grid layout:
        #   FL[7]  F[0]  FR[1]
        #   L[6]    A    R[2]
        #   BL[5]  B[4]  BR[3]
        grid_order = [7, 0, 1, 6, -1, 2, 5, 4, 3]  # -1 is agent position
        pos_labels = ['FL', 'F', 'FR', 'L', 'A', 'R', 'BL', 'B', 'BR']

        print(f"\n  Cell probabilities (% empty / % box / % wall):")
        print(f"\n       Front")
        for row in range(3):
            row_str = "    "
            for col in range(3):
                idx = row * 3 + col
                pos = grid_order[idx]
                if pos == -1:
                    row_str += "[  Agent  ] "
                else:
                    e = prob[pos][0] * 100
                    b = prob[pos][1] * 100
                    w = prob[pos][2] * 100
                    row_str += f"[{e:2.0f}/{b:2.0f}/{w:2.0f}] "
            print(row_str)
        print(f"       Back")
        print(f"\n  Legend: [%empty / %box / %wall]")

        # Most common combinations with visual grid
        print(f"\n  Top 5 most common combinations:")
        sorted_combos = sorted(top_combos, key=lambda x: -x['count'])
        for combo_data in sorted_combos[:5]:
            count = combo_data['count']
            pct = 100 * count / total_observations
            cells = decode_combo(combo_data['combo_idx'])
            # Create visual: . = empty, B = box, # = wall
            char_map = ['.', 'B', '#']
            # Grid: FL F FR / L A R / BL B BR -> positions 7,0,1 / 6,-,2 / 5,4,3
            grid = f"{char_map[cells[7]]}{char_map[cells[0]]}{char_map[cells[1]]}/" \
                   f"{char_map[cells[6]]}A{char_map[cells[2]]}/" \
                   f"{char_map[cells[5]]}{char_map[cells[4]]}{char_map[cells[3]]}"
            print(f"    {count:>8,} ({pct:>5.1f}%)  {grid}  (.=empty B=box #=wall)")

        # Perception as nested list (probabilities for each cell)
        print(f"\n  Probabilities as nested lists (for code):")
        print(f"    # Grid positions: [FL, F, FR], [L, agent, R], [BL, B, BR]")
        print(f"    # Each cell: [P(empty), P(box), P(wall)]")
        print(f"    [")
        print(f"      [{prob[7]}, {prob[0]}, {prob[1]}],")
        print(f"      [{prob[6]}, None, {prob[2]}],")
        print(f"      [{prob[5]}, {prob[4]}, {prob[3]}]")
        print(f"    ]")
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
