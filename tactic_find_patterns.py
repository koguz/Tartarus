#!/usr/bin/env python3
"""
Find recurring tactic transition patterns in agent behavior.

This script analyzes the tactic sequences from analyze_agent.py to find
common patterns (n-grams) that appear frequently across board runs.
Tactics are compact representations of (perception, action) pairs.

Tactic naming: {B/C}{F/-}{S/-}{K/-}{W/O}_{F/T}
- B = box in front, C = clear front
- F = boxes in front area (FL, FR), - = none
- S = boxes on sides (L, R), - = none
- K = boxes behind (BL, B, BR), - = none
- W = wall present, O = open (no wall)
- F = forward/push action, T = turn action

Usage:
    python tactic_find_patterns.py [analysis_prefix] [--min-length N] [--max-length M]

Default: min-length=3, max-length=15

Output files:
    {prefix}_tactic_patterns.json - All patterns in JSON format
    {prefix}_tactic_patterns.txt - Human-readable summary (top patterns)
    {prefix}_tactic_patterns_all.txt - Complete list of ALL patterns
"""

import pickle
import json
import sys
from collections import defaultdict, Counter


def load_sequences(prefix='analysis'):
    """Load tactic sequences from pickle file."""
    with open(f'{prefix}_tactic_sequences.pkl', 'rb') as f:
        sequences = pickle.load(f)
    return sequences


def load_tactic_stats(prefix='analysis'):
    """Load tactic statistics for context."""
    with open(f'{prefix}_tactic_stats.json', 'r') as f:
        return json.load(f)


def decode_tactic(tactic):
    """Decode a tactic string into human-readable description."""
    if len(tactic) < 7 or tactic[5] != '_':
        return "invalid"

    parts = []

    # Box in front
    if tactic[0] == 'B':
        parts.append("box in front")
    else:
        parts.append("clear front")

    # Other boxes
    boxes = []
    if tactic[1] == 'F':
        boxes.append("front area")
    if tactic[2] == 'S':
        boxes.append("sides")
    if tactic[3] == 'K':
        boxes.append("behind")

    if boxes:
        parts.append("boxes: " + "+".join(boxes))
    else:
        parts.append("no other boxes")

    # Wall
    if tactic[4] == 'W':
        parts.append("walled")
    else:
        parts.append("open")

    # Action
    if tactic[6] == 'F':
        parts.append("→ FORWARD")
    else:
        parts.append("→ TURN")

    return ", ".join(parts)


def extract_ngrams(sequence, n):
    """Extract all n-grams from a sequence."""
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i+n])
        ngrams.append(ngram)
    return ngrams


def find_frequent_ngrams(sequences, n, min_count=100):
    """Find the most frequent n-grams across all sequences.
    Counts each n-gram once per sequence (presence/absence)."""
    ngram_counts = Counter()

    for seq in sequences:
        ngrams = extract_ngrams(seq, n)
        # Count each unique n-gram once per sequence
        unique_ngrams = set(ngrams)
        for ng in unique_ngrams:
            ngram_counts[ng] += 1

    # Filter by minimum count
    frequent = [(ng, count) for ng, count in ngram_counts.most_common() if count >= min_count]
    return frequent


def find_frequent_ngrams_total(sequences, n, min_count=100):
    """Find the most frequent n-grams counting ALL occurrences."""
    ngram_counts = Counter()
    ngram_sequence_counts = Counter()

    for seq in sequences:
        ngrams = extract_ngrams(seq, n)
        for ng in ngrams:
            ngram_counts[ng] += 1
        for ng in set(ngrams):
            ngram_sequence_counts[ng] += 1

    frequent = [(ng, ngram_counts[ng], ngram_sequence_counts[ng])
                for ng in ngram_counts if ngram_counts[ng] >= min_count]
    frequent.sort(key=lambda x: -x[1])
    return frequent


def find_repeating_patterns_in_sequence(sequence, min_length=3, max_length=15):
    """Find patterns that repeat consecutively within a single sequence."""
    patterns = []

    for length in range(min_length, max_length + 1):
        for start in range(len(sequence) - length * 2 + 1):
            pattern = tuple(sequence[start:start + length])
            next_segment = tuple(sequence[start + length:start + length * 2])
            if pattern == next_segment:
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


def find_all_repeating_patterns(sequences, min_length=3, max_length=15, min_occurrences=100):
    """Find patterns that repeat within sequences, across all sequences."""
    pattern_stats = defaultdict(lambda: {'total_repeats': 0, 'sequences_with_pattern': 0})

    for seq in sequences:
        patterns = find_repeating_patterns_in_sequence(seq, min_length, max_length)
        seen_patterns = set()

        for pattern, repeat_count, start_pos in patterns:
            pattern_stats[pattern]['total_repeats'] += repeat_count
            if pattern not in seen_patterns:
                pattern_stats[pattern]['sequences_with_pattern'] += 1
                seen_patterns.add(pattern)

    results = []
    for pattern, stats in pattern_stats.items():
        if stats['sequences_with_pattern'] >= min_occurrences:
            results.append((pattern, stats['sequences_with_pattern'], stats['total_repeats']))

    results.sort(key=lambda x: -x[1])
    return results


def get_pattern_description(pattern):
    """Get a brief description of the pattern's behavior."""
    actions = []
    for tactic in pattern:
        if len(tactic) >= 7:
            if tactic[0] == 'B' and tactic[6] == 'F':
                actions.append('PUSH')
            elif tactic[0] == 'C' and tactic[6] == 'F':
                actions.append('FWD')
            elif tactic[6] == 'T':
                actions.append('TRN')
            else:
                actions.append('?')
        else:
            actions.append('?')
    return '-'.join(actions)


def save_all_patterns_to_txt(prefix, results, tactic_stats):
    """Save ALL patterns to a comprehensive text file."""
    output_file = f'{prefix}_tactic_patterns_all.txt'

    with open(output_file, 'w') as f:
        f.write("="*90 + "\n")
        f.write("COMPLETE TACTIC PATTERN ANALYSIS (ALL PATTERNS)\n")
        f.write("="*90 + "\n\n")

        f.write("Tactic naming: {B/C}{F/-}{S/-}{K/-}{W/O}_{F/T}\n")
        f.write("  B=box in front, C=clear, F=front area boxes, S=side boxes, K=back boxes\n")
        f.write("  W=wall, O=open, _F=forward, _T=turn\n\n")

        # Frequent n-grams
        f.write("="*90 + "\n")
        f.write("ALL FREQUENT TACTIC SEQUENCES (n-grams)\n")
        f.write("="*90 + "\n\n")

        for n, patterns in sorted(results['frequent_ngrams'].items()):
            if not patterns:
                continue
            f.write(f"\n{'='*90}\n")
            f.write(f"{n}-grams (sequences of {n} tactics)\n")
            f.write(f"{'='*90}\n")
            f.write(f"Total patterns: {len(patterns)}\n\n")
            f.write(f"{'Pattern':<70} {'Sequences':>10} {'Behavior':<20}\n")
            f.write("-" * 105 + "\n")

            for pattern, count in patterns:
                pattern_str = ' → '.join(pattern)
                if len(pattern_str) > 68:
                    pattern_str = pattern_str[:65] + "..."
                behavior = get_pattern_description(pattern)
                f.write(f"{pattern_str:<70} {count:>10,} {behavior:<20}\n")

        # Total occurrences
        f.write("\n" + "="*90 + "\n")
        f.write("ALL TOTAL PATTERN OCCURRENCES (counting all instances)\n")
        f.write("="*90 + "\n\n")

        for n, patterns in sorted(results['total_occurrences'].items()):
            if not patterns:
                continue
            f.write(f"\n{'='*90}\n")
            f.write(f"{n}-grams (total occurrences)\n")
            f.write(f"{'='*90}\n")
            f.write(f"Total patterns: {len(patterns)}\n\n")
            f.write(f"{'Pattern':<55} {'Total':>10} {'Sequences':>10} {'Avg/Seq':>8} {'Behavior':<15}\n")
            f.write("-" * 105 + "\n")

            for p in patterns:
                pattern_str = ' → '.join(p['pattern'])
                if len(pattern_str) > 53:
                    pattern_str = pattern_str[:50] + "..."
                behavior = get_pattern_description(tuple(p['pattern']))
                f.write(f"{pattern_str:<55} {p['total']:>10,} {p['sequences']:>10,} {p['avg_per_seq']:>8.2f} {behavior:<15}\n")

        # Repeating patterns
        f.write("\n" + "="*90 + "\n")
        f.write("ALL REPEATING PATTERNS (loops within sequences)\n")
        f.write("="*90 + "\n\n")

        repeating = results['repeating_patterns']
        if repeating:
            f.write(f"Total repeating patterns: {len(repeating)}\n\n")
            f.write(f"{'Pattern':<55} {'Sequences':>10} {'Total Reps':>12} {'Behavior':<15}\n")
            f.write("-" * 95 + "\n")

            for p in repeating:
                pattern_str = ' → '.join(p['pattern'])
                if len(pattern_str) > 53:
                    pattern_str = pattern_str[:50] + "..."
                behavior = get_pattern_description(tuple(p['pattern']))
                f.write(f"{pattern_str:<55} {p['sequences']:>10,} {p['total_repeats']:>12,} {behavior:<15}\n")
        else:
            f.write("No repeating patterns found\n")

    print(f"Saved comprehensive results to {output_file}")


def save_patterns_to_txt(prefix, results, tactic_stats):
    """Save pattern analysis results to a human-readable text file (summary)."""
    output_file = f'{prefix}_tactic_patterns.txt'

    with open(output_file, 'w') as f:
        f.write("="*90 + "\n")
        f.write("TACTIC PATTERN ANALYSIS\n")
        f.write("="*90 + "\n\n")

        f.write("Tactic naming: {B/C}{F/-}{S/-}{K/-}{W/O}_{F/T}\n")
        f.write("  B=box in front, C=clear, F=front area boxes, S=side boxes, K=back boxes\n")
        f.write("  W=wall, O=open, _F=forward, _T=turn\n\n")

        # Frequent n-grams
        f.write("="*90 + "\n")
        f.write("FREQUENT TACTIC SEQUENCES (n-grams)\n")
        f.write("="*90 + "\n")

        for n, patterns in sorted(results['frequent_ngrams'].items()):
            if not patterns:
                continue
            f.write(f"\n--- {n}-grams (sequences of {n} tactics) ---\n")
            f.write(f"Found {len(patterns)} patterns appearing in 500+ sequences\n\n")
            f.write(f"{'Pattern':<70} {'Sequences':>10}\n")
            f.write("-" * 85 + "\n")

            for pattern, count in patterns[:20]:
                pattern_str = ' → '.join(pattern)
                if len(pattern_str) > 68:
                    pattern_str = pattern_str[:65] + "..."
                f.write(f"{pattern_str:<70} {count:>10,}\n")

        # Total occurrences
        f.write("\n" + "="*90 + "\n")
        f.write("TOTAL PATTERN OCCURRENCES (counting all instances)\n")
        f.write("="*90 + "\n")

        for n, patterns in sorted(results['total_occurrences'].items()):
            if not patterns:
                continue
            f.write(f"\n--- {n}-grams (total occurrences) ---\n")
            f.write(f"Found {len(patterns)} patterns with 5000+ total occurrences\n\n")
            f.write(f"{'Pattern':<55} {'Total':>10} {'Sequences':>10} {'Avg/Seq':>8}\n")
            f.write("-" * 90 + "\n")

            for p in patterns[:20]:
                pattern_str = ' → '.join(p['pattern'])
                if len(pattern_str) > 53:
                    pattern_str = pattern_str[:50] + "..."
                f.write(f"{pattern_str:<55} {p['total']:>10,} {p['sequences']:>10,} {p['avg_per_seq']:>8.2f}\n")

        # Repeating patterns
        f.write("\n" + "="*90 + "\n")
        f.write("REPEATING PATTERNS (loops within sequences)\n")
        f.write("="*90 + "\n\n")

        repeating = results['repeating_patterns']
        if repeating:
            f.write(f"Found {len(repeating)} repeating patterns in 500+ sequences\n\n")
            f.write(f"{'Pattern':<55} {'Sequences':>10} {'Total Reps':>12}\n")
            f.write("-" * 80 + "\n")

            for p in repeating[:30]:
                pattern_str = ' → '.join(p['pattern'])
                if len(pattern_str) > 53:
                    pattern_str = pattern_str[:50] + "..."
                f.write(f"{pattern_str:<55} {p['sequences']:>10,} {p['total_repeats']:>12,}\n")
        else:
            f.write("No significant repeating patterns found\n")

        # Pattern details
        f.write("\n" + "="*90 + "\n")
        f.write("DETAILED PATTERN DESCRIPTIONS\n")
        f.write("="*90 + "\n")
        f.write("Top 10 most frequent patterns with full descriptions:\n\n")

        top_patterns = []
        for n in [3, 4, 5]:
            if n in results['frequent_ngrams'] and results['frequent_ngrams'][n]:
                top_patterns.extend([(p, c, n) for p, c in results['frequent_ngrams'][n][:5]])

        top_patterns.sort(key=lambda x: -x[1])

        for pattern, count, n in top_patterns[:10]:
            f.write(f"\nPattern: {' → '.join(pattern)}\n")
            f.write(f"  Length: {n}-gram\n")
            f.write(f"  Appears in: {count:,} sequences\n")
            f.write(f"  Behavior: {get_pattern_description(pattern)}\n")
            f.write(f"  Steps:\n")
            for i, tactic in enumerate(pattern):
                desc = decode_tactic(tactic)
                f.write(f"    {i+1}. [{tactic}] {desc}\n")
            f.write("\n")

    print(f"Saved human-readable results to {output_file}")


def analyze_patterns(prefix='analysis', min_length=3, max_length=15):
    """Main analysis function."""
    print(f"Loading tactic sequences from {prefix}_tactic_sequences.pkl...")
    sequences = load_sequences(prefix)
    print(f"Loaded {len(sequences)} sequences")

    print(f"Loading tactic statistics...")
    tactic_stats = load_tactic_stats(prefix)

    total_tactics = sum(len(seq) for seq in sequences)
    print(f"Total tactic visits: {total_tactics:,}")

    # Count unique tactics used
    all_tactics = set()
    for seq in sequences:
        all_tactics.update(seq)
    print(f"Unique tactics observed: {len(all_tactics)}")

    # Find frequent n-grams
    print("\n" + "="*70)
    print("FREQUENT TACTIC SEQUENCES (n-grams)")
    print("="*70)

    for n in range(min_length, min(max_length + 1, 8)):  # Print up to 7-grams
        print(f"\n--- {n}-grams (sequences of {n} tactics) ---")
        frequent = find_frequent_ngrams(sequences, n, min_count=1000)

        if frequent:
            print(f"Found {len(frequent)} patterns appearing in 1000+ sequences")
            print(f"{'Pattern':<60} {'Sequences':>10}")
            print("-" * 75)

            for pattern, count in frequent[:10]:
                pattern_str = ' → '.join(pattern)
                if len(pattern_str) > 58:
                    pattern_str = pattern_str[:55] + "..."
                print(f"{pattern_str:<60} {count:>10,}")
        else:
            print("No patterns found with sufficient frequency")

    # Find n-grams with total occurrence counts
    print("\n" + "="*70)
    print("TOTAL PATTERN OCCURRENCES (counting all instances)")
    print("="*70)

    for n in range(min_length, min(max_length + 1, 6)):  # Print up to 5-grams
        print(f"\n--- {n}-grams (total occurrences) ---")
        frequent_total = find_frequent_ngrams_total(sequences, n, min_count=5000)

        if frequent_total:
            print(f"Found {len(frequent_total)} patterns with 5000+ total occurrences")
            print(f"{'Pattern':<50} {'Total':>10} {'Sequences':>10} {'Avg/Seq':>8}")
            print("-" * 85)

            for pattern, total_count, seq_count in frequent_total[:10]:
                pattern_str = ' → '.join(pattern)
                if len(pattern_str) > 48:
                    pattern_str = pattern_str[:45] + "..."
                avg_per_seq = total_count / seq_count if seq_count > 0 else 0
                print(f"{pattern_str:<50} {total_count:>10,} {seq_count:>10,} {avg_per_seq:>8.2f}")
        else:
            print("No patterns found with sufficient frequency")

    # Find repeating patterns
    print("\n" + "="*70)
    print("REPEATING PATTERNS (loops within sequences)")
    print("="*70)

    print("\nSearching for patterns that repeat consecutively...")
    repeating = find_all_repeating_patterns(sequences, min_length, max_length, min_occurrences=500)

    if repeating:
        print(f"\nFound {len(repeating)} repeating patterns in 500+ sequences")
        print(f"{'Pattern':<50} {'Sequences':>10} {'Total Reps':>12}")
        print("-" * 75)

        for pattern, seq_count, total_repeats in repeating[:15]:
            pattern_str = ' → '.join(pattern)
            if len(pattern_str) > 48:
                pattern_str = pattern_str[:45] + "..."
            print(f"{pattern_str:<50} {seq_count:>10,} {total_repeats:>12,}")
    else:
        print("No significant repeating patterns found")

    # Save results to JSON
    results = {
        'frequent_ngrams': {
            n: [(list(p), c) for p, c in find_frequent_ngrams(sequences, n, min_count=500)]
            for n in range(min_length, max_length + 1)
        },
        'total_occurrences': {
            n: [{'pattern': list(p), 'total': t, 'sequences': s, 'avg_per_seq': t/s if s > 0 else 0}
                for p, t, s in find_frequent_ngrams_total(sequences, n, min_count=1000)]
            for n in range(min_length, max_length + 1)
        },
        'repeating_patterns': [
            {'pattern': list(p), 'sequences': s, 'total_repeats': t}
            for p, s, t in repeating
        ]
    }

    output_file_json = f'{prefix}_tactic_patterns.json'
    with open(output_file_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {output_file_json}")

    # Save to TXT files
    save_patterns_to_txt(prefix, results, tactic_stats)
    save_all_patterns_to_txt(prefix, results, tactic_stats)

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Most common starting tactics
    start_tactics = Counter(seq[0] for seq in sequences)
    print(f"\nMost common starting tactics:")
    for tactic, count in start_tactics.most_common(5):
        pct = 100 * count / len(sequences)
        print(f"  {tactic}: {count:,} ({pct:.1f}%)")

    # Most common ending tactics
    end_tactics = Counter(seq[-1] for seq in sequences)
    print(f"\nMost common ending tactics (after 80 moves):")
    for tactic, count in end_tactics.most_common(5):
        pct = 100 * count / len(sequences)
        print(f"  {tactic}: {count:,} ({pct:.1f}%)")

    # Average unique tactics per sequence
    unique_per_seq = [len(set(seq)) for seq in sequences]
    avg_unique = sum(unique_per_seq) / len(unique_per_seq)
    print(f"\nAverage unique tactics per sequence: {avg_unique:.1f}")

    # Self-loop analysis
    total_transitions = 0
    self_loops = 0
    for seq in sequences:
        for i in range(len(seq) - 1):
            total_transitions += 1
            if seq[i] == seq[i + 1]:
                self_loops += 1

    print(f"\nSelf-loop analysis (same tactic repeated):")
    print(f"  Total transitions: {total_transitions:,}")
    print(f"  Self-loops: {self_loops:,} ({100*self_loops/total_transitions:.2f}%)")
    print(f"  Tactic changes: {total_transitions - self_loops:,} ({100*(total_transitions-self_loops)/total_transitions:.2f}%)")

    # Most common self-loop tactics
    self_loop_counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            if seq[i] == seq[i + 1]:
                self_loop_counts[seq[i]] += 1

    if self_loop_counts:
        print(f"\nMost common self-loop tactics (tactics that repeat consecutively):")
        for tactic, count in self_loop_counts.most_common(10):
            desc = decode_tactic(tactic)
            print(f"  {tactic}: {count:,} - {desc}")

    return results


if __name__ == '__main__':
    prefix = 'analysis'
    min_length = 3
    max_length = 15

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
