#!/usr/bin/env python3
"""
Analyze sequences to extract modular behaviors and their compositions.

This script finds:
1. Atomic behaviors (short, frequent patterns)
2. Composite behaviors (longer patterns made of atomic ones)
3. Maximal sequences (longest patterns that can't be extended without losing frequency)
4. Behavioral loops and entry/exit states

Usage:
    python analyze_behaviors.py [prefix] [options]

Options:
    --pattern X-Y-Z     Query a specific pattern
    --min-support N     Minimum support threshold for maximal sequences (default: 500)

Examples:
    python analyze_behaviors.py analysis
    python analyze_behaviors.py analysis --min-support 1000
    python analyze_behaviors.py analysis --pattern 75-42-118-2-22-94
"""

import pickle
import json
import sys
from collections import defaultdict, Counter


def load_data(prefix='analysis'):
    """Load sequences and state stats."""
    with open(f'{prefix}_sequences.pkl', 'rb') as f:
        sequences = pickle.load(f)
    with open(f'{prefix}_state_stats.json', 'r') as f:
        state_stats = json.load(f)
    return sequences, state_stats


def get_state_action(state_id, state_stats):
    """Get the dominant action for a state."""
    stats = state_stats.get(str(state_id), {})
    action_counts = stats.get('action_counts', {})
    if not action_counts:
        return '?'

    push = action_counts.get('push', 0)
    forward = action_counts.get('forward', 0)
    turn_left = action_counts.get('turn_left', 0)
    turn_right = action_counts.get('turn_right', 0)
    turn = turn_left + turn_right

    if push >= forward and push >= turn:
        return 'P'
    elif forward >= turn:
        return 'F'
    else:
        if turn_left > turn_right:
            return 'L'
        else:
            return 'R'


def pattern_to_actions(pattern, state_stats):
    """Convert a state pattern to action string."""
    return ''.join(get_state_action(s, state_stats) for s in pattern)


def extract_all_ngrams(sequence, min_n=2, max_n=12):
    """Extract all n-grams from a sequence."""
    ngrams = defaultdict(list)
    for n in range(min_n, min(max_n + 1, len(sequence) + 1)):
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i+n])
            ngrams[n].append((ngram, i))
    return ngrams


def find_frequent_patterns(sequences, min_n=2, max_n=12, min_support=500):
    """Find patterns that appear in at least min_support sequences."""
    pattern_counts = Counter()
    pattern_positions = defaultdict(list)  # pattern -> [(seq_idx, pos), ...]

    for seq_idx, seq in enumerate(sequences):
        seen_in_seq = set()
        for n in range(min_n, min(max_n + 1, len(seq) + 1)):
            for i in range(len(seq) - n + 1):
                pattern = tuple(seq[i:i+n])
                if pattern not in seen_in_seq:
                    pattern_counts[pattern] += 1
                    seen_in_seq.add(pattern)
                pattern_positions[pattern].append((seq_idx, i))

    # Filter by support
    frequent = {p: c for p, c in pattern_counts.items() if c >= min_support}
    return frequent, pattern_positions


def is_subpattern(short, long):
    """Check if short pattern appears within long pattern."""
    if len(short) >= len(long):
        return False
    short_str = ','.join(map(str, short))
    long_str = ','.join(map(str, long))
    return short_str in long_str


def find_pattern_in_sequence(pattern, sequence):
    """Find all occurrences of pattern in sequence."""
    positions = []
    pattern_len = len(pattern)
    for i in range(len(sequence) - pattern_len + 1):
        if tuple(sequence[i:i+pattern_len]) == pattern:
            positions.append(i)
    return positions


def count_pattern_support(pattern, sequences):
    """Count how many sequences contain this pattern."""
    count = 0
    for seq in sequences:
        if find_pattern_in_sequence(pattern, seq):
            count += 1
    return count


def find_maximal_sequences(sequences, min_support=500, seed_length=3, max_length=30):
    """
    Find maximal frequent sequences by growing patterns until they drop below threshold.

    A sequence is maximal if:
    1. It appears in at least min_support sequences
    2. No extension (left or right) maintains the min_support threshold

    Algorithm:
    1. Start with frequent seed patterns (length 3)
    2. Try to extend each pattern left and right
    3. Keep extending while support >= min_support
    4. Stop when extension drops below threshold
    5. Remove non-maximal patterns (those contained in longer ones)
    """
    print(f"  Finding seed patterns (length {seed_length})...")

    # Find seed patterns
    seed_counts = Counter()
    for seq in sequences:
        seen = set()
        for i in range(len(seq) - seed_length + 1):
            pattern = tuple(seq[i:i+seed_length])
            if pattern not in seen:
                seed_counts[pattern] += 1
                seen.add(pattern)

    seeds = [p for p, c in seed_counts.items() if c >= min_support]
    print(f"  Found {len(seeds)} seed patterns above threshold")

    # Build index: for each pattern, store which sequences contain it and at what positions
    def build_index(pattern):
        """Build index of where pattern appears."""
        index = []  # [(seq_idx, position), ...]
        for seq_idx, seq in enumerate(sequences):
            for i in range(len(seq) - len(pattern) + 1):
                if tuple(seq[i:i+len(pattern)]) == pattern:
                    index.append((seq_idx, i))
        return index

    def get_support_from_index(index):
        """Count unique sequences in index."""
        return len(set(seq_idx for seq_idx, _ in index))

    def try_extend_right(pattern, index):
        """Try to extend pattern by one state to the right."""
        extensions = Counter()
        for seq_idx, pos in index:
            seq = sequences[seq_idx]
            end_pos = pos + len(pattern)
            if end_pos < len(seq):
                next_state = seq[end_pos]
                extensions[next_state] += 1

        # Find best extension that maintains support
        for next_state, count in extensions.most_common():
            # Count unique sequences with this extension
            new_pattern = pattern + (next_state,)
            new_index = [(seq_idx, pos) for seq_idx, pos in index
                        if pos + len(pattern) < len(sequences[seq_idx])
                        and sequences[seq_idx][pos + len(pattern)] == next_state]
            support = get_support_from_index(new_index)
            if support >= min_support:
                return new_pattern, new_index
        return None, None

    def try_extend_left(pattern, index):
        """Try to extend pattern by one state to the left."""
        extensions = Counter()
        for seq_idx, pos in index:
            if pos > 0:
                prev_state = sequences[seq_idx][pos - 1]
                extensions[prev_state] += 1

        # Find best extension that maintains support
        for prev_state, count in extensions.most_common():
            new_pattern = (prev_state,) + pattern
            new_index = [(seq_idx, pos - 1) for seq_idx, pos in index
                        if pos > 0 and sequences[seq_idx][pos - 1] == prev_state]
            support = get_support_from_index(new_index)
            if support >= min_support:
                return new_pattern, new_index
        return None, None

    # Grow each seed pattern to maximal length
    print(f"  Growing patterns to maximal length...")
    maximal = {}  # pattern -> support

    for i, seed in enumerate(seeds):
        if (i + 1) % 100 == 0:
            print(f"    Processing seed {i+1}/{len(seeds)}...")

        # Build initial index
        index = build_index(seed)
        pattern = seed

        # Extend right as far as possible
        while len(pattern) < max_length:
            new_pattern, new_index = try_extend_right(pattern, index)
            if new_pattern is None:
                break
            pattern = new_pattern
            index = new_index

        # Extend left as far as possible
        while len(pattern) < max_length:
            new_pattern, new_index = try_extend_left(pattern, index)
            if new_pattern is None:
                break
            pattern = new_pattern
            index = new_index

        support = get_support_from_index(index)
        if pattern not in maximal or maximal[pattern] < support:
            maximal[pattern] = support

    # Remove non-maximal patterns (those contained in longer ones)
    print(f"  Removing non-maximal patterns...")
    patterns_list = sorted(maximal.keys(), key=len, reverse=True)
    truly_maximal = {}

    for pattern in patterns_list:
        # Check if this pattern is contained in any already-added (longer) pattern
        dominated = False
        for longer in truly_maximal:
            if is_subpattern(pattern, longer):
                dominated = True
                break
        if not dominated:
            truly_maximal[pattern] = maximal[pattern]

    print(f"  Found {len(truly_maximal)} maximal sequences")
    return truly_maximal


def classify_behavior(actions):
    """Classify a behavior based on its action sequence."""
    push_count = actions.count('P')
    forward_count = actions.count('F')
    turn_count = actions.count('L') + actions.count('R')
    total = len(actions)

    # Check for specific patterns
    if actions == 'PP' or actions == 'PPP':
        return 'consecutive-push'
    if 'PP' in actions:
        return 'multi-push'
    if push_count == 0 and turn_count > 0:
        return 'repositioning'
    if push_count == 1 and actions.endswith('P'):
        return 'approach-push'
    if push_count == 1 and actions.startswith('P'):
        return 'push-retreat'
    if push_count > 0 and turn_count > 0:
        return 'push-turn-combo'
    if forward_count > 0 and turn_count == 0 and push_count == 0:
        return 'straight-move'
    if turn_count > forward_count:
        return 'turning'

    return 'mixed'


def decompose_pattern(pattern, atomic_patterns, state_stats):
    """Try to decompose a pattern into atomic sub-patterns."""
    actions = pattern_to_actions(pattern, state_stats)
    decomposition = []

    # Sort atomic patterns by length (longest first) for greedy matching
    sorted_atoms = sorted(atomic_patterns, key=len, reverse=True)

    i = 0
    while i < len(pattern):
        matched = False
        for atom in sorted_atoms:
            atom_len = len(atom)
            if i + atom_len <= len(pattern):
                if tuple(pattern[i:i+atom_len]) == atom:
                    decomposition.append(atom)
                    i += atom_len
                    matched = True
                    break
        if not matched:
            # Single state as fallback
            decomposition.append((pattern[i],))
            i += 1

    return decomposition


def find_behavior_transitions(sequences, behaviors, state_stats):
    """Find which behaviors tend to follow which."""
    transitions = Counter()

    for seq in sequences:
        # Find all behavior occurrences in this sequence
        occurrences = []  # (start_pos, end_pos, behavior)
        for behavior in behaviors:
            positions = find_pattern_in_sequence(behavior, seq)
            for pos in positions:
                occurrences.append((pos, pos + len(behavior), behavior))

        # Sort by start position
        occurrences.sort(key=lambda x: x[0])

        # Find non-overlapping consecutive behaviors
        i = 0
        while i < len(occurrences) - 1:
            curr_end = occurrences[i][1]
            # Find next non-overlapping behavior
            for j in range(i + 1, len(occurrences)):
                if occurrences[j][0] >= curr_end:
                    # Found a transition
                    b1 = occurrences[i][2]
                    b2 = occurrences[j][2]
                    transitions[(b1, b2)] += 1
                    break
            i += 1

    return transitions


def analyze_behaviors(prefix='analysis', min_support=500):
    """Main analysis function."""
    print(f"Loading data from {prefix}...")
    print(f"Using minimum support threshold: {min_support}")
    sequences, state_stats = load_data(prefix)
    print(f"Loaded {len(sequences)} sequences")

    # Find frequent patterns of various lengths
    print("\nFinding frequent patterns...")
    frequent, positions = find_frequent_patterns(sequences, min_n=2, max_n=10, min_support=500)
    print(f"Found {len(frequent)} frequent patterns")

    # Group by length
    by_length = defaultdict(list)
    for pattern, count in frequent.items():
        by_length[len(pattern)].append((pattern, count))

    for length in sorted(by_length.keys()):
        by_length[length].sort(key=lambda x: -x[1])

    # Identify atomic behaviors (short, very frequent patterns)
    print("\n" + "=" * 70)
    print("ATOMIC BEHAVIORS (length 2-3, high frequency)")
    print("=" * 70)

    atomic_behaviors = []
    for length in [2, 3]:
        if length in by_length:
            print(f"\n--- Length {length} ---")
            print(f"{'Pattern':<30} {'Count':>10} {'Actions':<10} {'Type':<20}")
            print("-" * 75)
            for pattern, count in by_length[length][:20]:
                actions = pattern_to_actions(pattern, state_stats)
                behavior_type = classify_behavior(actions)
                pattern_str = '-'.join(map(str, pattern))
                print(f"{pattern_str:<30} {count:>10,} {actions:<10} {behavior_type:<20}")
                if count >= 2000:  # Very frequent = atomic
                    atomic_behaviors.append(pattern)

    print(f"\nIdentified {len(atomic_behaviors)} atomic behaviors")

    # Find composite behaviors (longer patterns)
    print("\n" + "=" * 70)
    print("COMPOSITE BEHAVIORS (length 4-8)")
    print("=" * 70)

    composite_behaviors = []
    for length in range(4, 9):
        if length in by_length and by_length[length]:
            print(f"\n--- Length {length} ---")
            print(f"{'Pattern':<40} {'Count':>8} {'Actions':<12} {'Type':<15}")
            print("-" * 80)
            for pattern, count in by_length[length][:10]:
                actions = pattern_to_actions(pattern, state_stats)
                behavior_type = classify_behavior(actions)
                pattern_str = '-'.join(map(str, pattern))
                print(f"{pattern_str:<40} {count:>8,} {actions:<12} {behavior_type:<15}")
                composite_behaviors.append(pattern)

    # Analyze composition of longer patterns
    print("\n" + "=" * 70)
    print("BEHAVIOR DECOMPOSITION")
    print("=" * 70)
    print("Showing how longer patterns decompose into atomic behaviors\n")

    # Take top composite behaviors and decompose them
    top_composites = []
    for length in range(5, 11):
        if length in by_length:
            top_composites.extend(by_length[length][:5])
    top_composites.sort(key=lambda x: -x[1])

    for pattern, count in top_composites[:15]:
        actions = pattern_to_actions(pattern, state_stats)
        pattern_str = '-'.join(map(str, pattern))
        print(f"\nPattern: {pattern_str}")
        print(f"Actions: {actions}")
        print(f"Frequency: {count:,} sequences")

        # Find which atomic behaviors are contained
        contained = []
        for atom in atomic_behaviors:
            if is_subpattern(atom, pattern):
                atom_actions = pattern_to_actions(atom, state_stats)
                contained.append((atom, atom_actions))

        if contained:
            print("Contains atomic behaviors:")
            for atom, atom_actions in contained:
                atom_str = '-'.join(map(str, atom))
                print(f"  - {atom_str} ({atom_actions})")

    # Find loops (patterns that start and end in same state)
    print("\n" + "=" * 70)
    print("BEHAVIORAL LOOPS (same start and end state)")
    print("=" * 70)

    loops = []
    for pattern, count in frequent.items():
        if len(pattern) >= 3 and pattern[0] == pattern[-1]:
            loops.append((pattern, count))

    loops.sort(key=lambda x: -x[1])
    print(f"\nFound {len(loops)} looping patterns")
    print(f"\n{'Pattern':<50} {'Count':>8} {'Actions':<15}")
    print("-" * 75)
    for pattern, count in loops[:20]:
        actions = pattern_to_actions(pattern, state_stats)
        pattern_str = '-'.join(map(str, pattern))
        print(f"{pattern_str:<50} {count:>8,} {actions:<15}")

    # Find "entry points" - states that commonly start patterns
    print("\n" + "=" * 70)
    print("COMMON ENTRY/EXIT STATES")
    print("=" * 70)

    entry_states = Counter()
    exit_states = Counter()
    for pattern, count in frequent.items():
        if len(pattern) >= 4:
            entry_states[pattern[0]] += count
            exit_states[pattern[-1]] += count

    print("\nMost common pattern entry states:")
    for state, count in entry_states.most_common(10):
        action = get_state_action(state, state_stats)
        print(f"  State {state} ({action}): {count:,}")

    print("\nMost common pattern exit states:")
    for state, count in exit_states.most_common(10):
        action = get_state_action(state, state_stats)
        print(f"  State {state} ({action}): {count:,}")

    # Find maximal sequences
    print("\n" + "=" * 70)
    print("MAXIMAL SEQUENCES (longest patterns above frequency threshold)")
    print("=" * 70)
    print("\nFinding maximal sequences that cannot be extended further...")
    print("(This may take a moment)\n")

    maximal_seqs = find_maximal_sequences(sequences, min_support=min_support, seed_length=3, max_length=30)

    # Sort by length (longest first), then by support
    sorted_maximal = sorted(maximal_seqs.items(), key=lambda x: (-len(x[0]), -x[1]))

    print(f"\n{'Pattern':<70} {'Len':>4} {'Support':>8} {'Actions'}")
    print("-" * 110)
    for pattern, support in sorted_maximal[:30]:  # Show top 30
        actions = pattern_to_actions(pattern, state_stats)
        pattern_str = '-'.join(map(str, pattern))
        if len(pattern_str) > 68:
            pattern_str = pattern_str[:65] + "..."
        print(f"{pattern_str:<70} {len(pattern):>4} {support:>8,} {actions}")

    # Also show some statistics about maximal sequences
    lengths = [len(p) for p in maximal_seqs.keys()]
    if lengths:
        print(f"\nMaximal sequence statistics:")
        print(f"  Total maximal sequences: {len(maximal_seqs)}")
        print(f"  Length range: {min(lengths)} - {max(lengths)}")
        print(f"  Average length: {sum(lengths)/len(lengths):.1f}")

        # Group by length
        by_len = defaultdict(int)
        for p in maximal_seqs:
            by_len[len(p)] += 1
        print(f"  Distribution by length:")
        for l in sorted(by_len.keys()):
            print(f"    Length {l}: {by_len[l]} sequences")

    # Save results
    results = {
        'atomic_behaviors': [
            {
                'pattern': list(p),
                'count': frequent[p],
                'actions': pattern_to_actions(p, state_stats),
                'type': classify_behavior(pattern_to_actions(p, state_stats))
            }
            for p in atomic_behaviors
        ],
        'composite_behaviors': [
            {
                'pattern': list(p),
                'count': c,
                'actions': pattern_to_actions(p, state_stats),
                'type': classify_behavior(pattern_to_actions(p, state_stats))
            }
            for p, c in top_composites[:50]
        ],
        'loops': [
            {
                'pattern': list(p),
                'count': c,
                'actions': pattern_to_actions(p, state_stats)
            }
            for p, c in loops[:50]
        ],
        'maximal_sequences': [
            {
                'pattern': list(p),
                'length': len(p),
                'support': s,
                'actions': pattern_to_actions(p, state_stats),
                'type': classify_behavior(pattern_to_actions(p, state_stats))
            }
            for p, s in sorted_maximal[:100]
        ]
    }

    output_file = f'{prefix}_behaviors.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")

    return results


def query_pattern(pattern_str, prefix='analysis'):
    """Query details about a specific pattern."""
    sequences, state_stats = load_data(prefix)
    pattern = tuple(map(int, pattern_str.split('-')))

    print(f"\n{'='*70}")
    print(f"PATTERN ANALYSIS: {pattern_str}")
    print(f"{'='*70}")

    # Basic info
    actions = pattern_to_actions(pattern, state_stats)
    print(f"\nActions: {actions}")
    print(f"Length: {len(pattern)} states")
    print(f"Type: {classify_behavior(actions)}")

    # Count occurrences
    total_occurrences = 0
    sequences_with_pattern = 0
    for seq in sequences:
        positions = find_pattern_in_sequence(pattern, seq)
        if positions:
            sequences_with_pattern += 1
            total_occurrences += len(positions)

    print(f"\nOccurrences:")
    print(f"  In {sequences_with_pattern:,} sequences ({100*sequences_with_pattern/len(sequences):.1f}%)")
    print(f"  Total occurrences: {total_occurrences:,}")
    print(f"  Avg per sequence: {total_occurrences/sequences_with_pattern:.2f}" if sequences_with_pattern > 0 else "")

    # State-by-state breakdown
    print(f"\nState-by-state breakdown:")
    print(f"  {'State':<8} {'Action':<8} {'Dominant %':<12}")
    print(f"  {'-'*30}")
    for i, state in enumerate(pattern):
        stats = state_stats.get(str(state), {})
        action_counts = stats.get('action_counts', {})
        total = sum(action_counts.values())
        action = get_state_action(state, state_stats)

        # Get dominant action percentage
        if action == 'P':
            pct = 100 * action_counts.get('push', 0) / total if total > 0 else 0
        elif action == 'F':
            pct = 100 * action_counts.get('forward', 0) / total if total > 0 else 0
        else:
            pct = 100 * (action_counts.get('turn_left', 0) + action_counts.get('turn_right', 0)) / total if total > 0 else 0

        print(f"  {state:<8} {action:<8} {pct:.1f}%")

    # Find sub-patterns
    print(f"\nSub-patterns contained:")
    frequent, _ = find_frequent_patterns(sequences, min_n=2, max_n=len(pattern)-1, min_support=1000)
    sub_patterns = []
    for sub, count in frequent.items():
        if is_subpattern(sub, pattern):
            sub_patterns.append((sub, count))

    sub_patterns.sort(key=lambda x: -x[1])
    for sub, count in sub_patterns[:10]:
        sub_str = '-'.join(map(str, sub))
        sub_actions = pattern_to_actions(sub, state_stats)
        print(f"  {sub_str} ({sub_actions}): {count:,} sequences")

    # Find super-patterns (patterns that contain this one)
    print(f"\nSuper-patterns containing this:")
    frequent_long, _ = find_frequent_patterns(sequences, min_n=len(pattern)+1, max_n=len(pattern)+4, min_support=500)
    super_patterns = []
    for sup, count in frequent_long.items():
        if is_subpattern(pattern, sup):
            super_patterns.append((sup, count))

    super_patterns.sort(key=lambda x: -x[1])
    for sup, count in super_patterns[:10]:
        sup_str = '-'.join(map(str, sup))
        sup_actions = pattern_to_actions(sup, state_stats)
        print(f"  {sup_str} ({sup_actions}): {count:,} sequences")


if __name__ == '__main__':
    prefix = 'analysis'
    pattern = None
    min_support = 500

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--pattern' and i + 1 < len(sys.argv):
            pattern = sys.argv[i + 1]
            i += 2
        elif arg == '--min-support' and i + 1 < len(sys.argv):
            min_support = int(sys.argv[i + 1])
            i += 2
        elif arg == '--help':
            print("Usage: python analyze_behaviors.py [prefix] [options]")
            print("\nOptions:")
            print("  --pattern X-Y-Z    Query a specific pattern")
            print("  --min-support N    Minimum support threshold (default: 500)")
            print("\nExamples:")
            print("  python analyze_behaviors.py analysis")
            print("  python analyze_behaviors.py analysis --min-support 1000")
            print("  python analyze_behaviors.py analysis --pattern 75-42-118-2-22-94")
            sys.exit(0)
        elif not arg.startswith('-'):
            prefix = arg
            i += 1
        else:
            i += 1

    if pattern:
        query_pattern(pattern, prefix)
    else:
        analyze_behaviors(prefix, min_support=min_support)
