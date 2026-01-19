#!/usr/bin/env python3
"""
Find recurring state transition patterns in agent behavior.

This script analyzes the state sequences from analyze_agent.py to find
common patterns (n-grams) that appear frequently across board runs.
These patterns represent emergent behaviors the agent has learned.

Usage:
    python find_patterns.py [analysis_prefix] [--min-length N] [--max-length M]
"""

import pickle
import json
import sys
from collections import defaultdict, Counter

def load_sequences(prefix='analysis'):
    """Load state sequences from pickle file."""
    with open(f'{prefix}_sequences.pkl', 'rb') as f:
        sequences = pickle.load(f)
    return sequences


def load_state_stats(prefix='analysis'):
    """Load state statistics for context."""
    with open(f'{prefix}_state_stats.json', 'r') as f:
        return json.load(f)


def extract_ngrams(sequence, n):
    """Extract all n-grams from a sequence."""
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i+n])
        ngrams.append(ngram)
    return ngrams


def find_frequent_ngrams(sequences, n, min_count=100):
    """Find the most frequent n-grams across all sequences."""
    ngram_counts = Counter()

    for seq in sequences:
        ngrams = extract_ngrams(seq, n)
        # Count each unique n-gram once per sequence (to avoid over-counting loops)
        unique_ngrams = set(ngrams)
        for ng in unique_ngrams:
            ngram_counts[ng] += 1

    # Filter by minimum count
    frequent = [(ng, count) for ng, count in ngram_counts.most_common() if count >= min_count]
    return frequent


def find_repeating_patterns_in_sequence(sequence, min_length=3, max_length=10):
    """
    Find patterns that repeat within a single sequence.
    Returns patterns that appear at least twice consecutively or multiple times.
    """
    patterns = []

    for length in range(min_length, max_length + 1):
        for start in range(len(sequence) - length * 2 + 1):
            pattern = tuple(sequence[start:start + length])
            # Check if pattern repeats immediately after
            next_segment = tuple(sequence[start + length:start + length * 2])
            if pattern == next_segment:
                # Found a repeating pattern, count total occurrences
                count = 0
                pos = start
                while pos + length <= len(sequence):
                    if tuple(sequence[pos:pos + length]) == pattern:
                        count += 1
                        pos += length
                    else:
                        break
                if count >= 2:
                    patterns.append((pattern, count, start))

    return patterns


def find_all_repeating_patterns(sequences, min_length=3, max_length=8, min_occurrences=100):
    """
    Find patterns that repeat within sequences, across all sequences.
    """
    pattern_stats = defaultdict(lambda: {'total_repeats': 0, 'sequences_with_pattern': 0})

    for seq_idx, seq in enumerate(sequences):
        patterns = find_repeating_patterns_in_sequence(seq, min_length, max_length)
        seen_patterns = set()

        for pattern, repeat_count, start_pos in patterns:
            pattern_stats[pattern]['total_repeats'] += repeat_count
            if pattern not in seen_patterns:
                pattern_stats[pattern]['sequences_with_pattern'] += 1
                seen_patterns.add(pattern)

    # Filter and sort
    results = []
    for pattern, stats in pattern_stats.items():
        if stats['sequences_with_pattern'] >= min_occurrences:
            results.append((pattern, stats['sequences_with_pattern'], stats['total_repeats']))

    results.sort(key=lambda x: -x[1])  # Sort by number of sequences
    return results


def get_pattern_action_info(pattern, state_stats):
    """Get action information for states in a pattern."""
    info = []
    for state in pattern:
        stats = state_stats.get(str(state), {})
        action_counts = stats.get('action_counts', {})
        push = action_counts.get('push', 0)
        forward = action_counts.get('forward', 0)
        turn_l = action_counts.get('turn_left', 0)
        turn_r = action_counts.get('turn_right', 0)
        turn = turn_l + turn_r
        total = push + forward + turn

        if total > 0:
            if push >= forward and push >= turn:
                dominant = 'P'
            elif forward >= turn:
                dominant = 'F'
            else:
                dominant = 'T'
        else:
            dominant = '?'

        info.append(dominant)

    return ''.join(info)


def analyze_patterns(prefix='analysis', min_length=3, max_length=8):
    """Main analysis function."""
    print(f"Loading sequences from {prefix}_sequences.pkl...")
    sequences = load_sequences(prefix)
    print(f"Loaded {len(sequences)} sequences")

    print(f"Loading state statistics...")
    state_stats = load_state_stats(prefix)

    total_states = sum(len(seq) for seq in sequences)
    print(f"Total state visits: {total_states:,}")

    # Find frequent n-grams of various lengths
    print("\n" + "="*70)
    print("FREQUENT STATE SEQUENCES (n-grams)")
    print("="*70)

    all_frequent_patterns = []

    for n in range(min_length, max_length + 1):
        print(f"\n--- {n}-grams (sequences of {n} states) ---")
        frequent = find_frequent_ngrams(sequences, n, min_count=1000)

        if frequent:
            print(f"Found {len(frequent)} patterns appearing in 1000+ sequences")
            print(f"{'Pattern':<40} {'Sequences':>10} {'Actions':<15}")
            print("-" * 70)

            for pattern, count in frequent[:15]:  # Top 15
                pattern_str = '-'.join(map(str, pattern))
                actions = get_pattern_action_info(pattern, state_stats)
                print(f"{pattern_str:<40} {count:>10,} {actions:<15}")
                all_frequent_patterns.append((pattern, count, n))
        else:
            print("No patterns found with sufficient frequency")

    # Find repeating patterns (loops)
    print("\n" + "="*70)
    print("REPEATING PATTERNS (loops within sequences)")
    print("="*70)

    print("\nSearching for patterns that repeat consecutively...")
    repeating = find_all_repeating_patterns(sequences, min_length, max_length, min_occurrences=500)

    if repeating:
        print(f"\nFound {len(repeating)} repeating patterns in 500+ sequences")
        print(f"{'Pattern':<40} {'Sequences':>10} {'Total Repeats':>15} {'Actions':<15}")
        print("-" * 85)

        for pattern, seq_count, total_repeats in repeating[:20]:  # Top 20
            pattern_str = '-'.join(map(str, pattern))
            actions = get_pattern_action_info(pattern, state_stats)
            print(f"{pattern_str:<40} {seq_count:>10,} {total_repeats:>15,} {actions:<15}")
    else:
        print("No significant repeating patterns found")

    # Save results
    results = {
        'frequent_ngrams': {
            n: [(list(p), c) for p, c in find_frequent_ngrams(sequences, n, min_count=500)[:50]]
            for n in range(min_length, max_length + 1)
        },
        'repeating_patterns': [
            {'pattern': list(p), 'sequences': s, 'total_repeats': t}
            for p, s, t in repeating[:100]
        ]
    }

    output_file = f'{prefix}_patterns.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {output_file}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Most common starting states
    start_states = Counter(seq[0] for seq in sequences)
    print(f"\nMost common starting states:")
    for state, count in start_states.most_common(5):
        pct = 100 * count / len(sequences)
        print(f"  State {state}: {count:,} ({pct:.1f}%)")

    # Most common ending states
    end_states = Counter(seq[-1] for seq in sequences)
    print(f"\nMost common ending states (after 80 moves):")
    for state, count in end_states.most_common(5):
        pct = 100 * count / len(sequences)
        print(f"  State {state}: {count:,} ({pct:.1f}%)")

    # Average unique states per sequence
    unique_per_seq = [len(set(seq)) for seq in sequences]
    avg_unique = sum(unique_per_seq) / len(unique_per_seq)
    print(f"\nAverage unique states visited per sequence: {avg_unique:.1f}")

    return results


if __name__ == '__main__':
    prefix = 'analysis'
    min_length = 3
    max_length = 8

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--min-length' and i + 1 < len(sys.argv):
            min_length = int(sys.argv[i + 1])
            i += 2
        elif arg == '--max-length' and i + 1 < len(sys.argv):
            max_length = int(sys.argv[i + 1])
            i += 2
        elif not arg.startswith('-'):
            prefix = arg
            i += 1
        else:
            i += 1

    analyze_patterns(prefix, min_length, max_length)
