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


def tactic_to_number(tactic):
    """
    Convert tactic string to number (0-63).

    Encoding (6 bits):
      Bit 5 (32): B=1, C=0 (box in front)
      Bit 4 (16): F=1, -=0 (front area boxes)
      Bit 3 (8):  S=1, -=0 (side boxes)
      Bit 2 (4):  K=1, -=0 (back boxes)
      Bit 1 (2):  W=1, O=0 (wall)
      Bit 0 (1):  T=1, F=0 (turn action)

    Example: B---W_F = 100010 = 34, C-S-W_F = 001010 = 10
    """
    if len(tactic) < 7 or tactic[5] != '_':
        return -1

    n = 0
    if tactic[0] == 'B': n += 32
    if tactic[1] == 'F': n += 16
    if tactic[2] == 'S': n += 8
    if tactic[3] == 'K': n += 4
    if tactic[4] == 'W': n += 2
    if tactic[6] == 'T': n += 1

    return n


def number_to_tactic(n):
    """Convert number (0-63) back to tactic string."""
    if n < 0 or n > 63:
        return "invalid"

    p1 = 'B' if (n & 32) else 'C'
    p2 = 'F' if (n & 16) else '-'
    p3 = 'S' if (n & 8) else '-'
    p4 = 'K' if (n & 4) else '-'
    p5 = 'W' if (n & 2) else 'O'
    p6 = 'T' if (n & 1) else 'F'

    return f"{p1}{p2}{p3}{p4}{p5}_{p6}"


def pattern_to_numbers(pattern):
    """Convert a pattern (list of tactics) to a list of numbers."""
    return [tactic_to_number(t) for t in pattern]


def pattern_numbers_str(pattern):
    """Get a compact numeric representation of a pattern."""
    nums = pattern_to_numbers(pattern)
    return '-'.join(str(n) for n in nums)


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
        f.write("TACTIC PATTERN ANALYSIS\n")
        f.write("=======================\n\n")
        f.write("Numeric encoding (0-63): B=32, F=16, S=8, K=4, W=2, T=1\n")
        f.write("Example: B---W_F = 34, C-S-W_F = 10\n\n")
        f.write("Format: pattern, sequences\n\n")

        # Frequent n-grams
        for n, patterns in sorted(results['frequent_ngrams'].items()):
            if not patterns:
                continue
            f.write(f"\n{n}-GRAMS (by sequence count)\n")
            f.write("-" * 40 + "\n")

            for pattern, count in patterns:
                numeric_str = pattern_numbers_str(pattern)
                f.write(f"{numeric_str}, {count}\n")

        # Total occurrences
        f.write("\n\nTOTAL OCCURRENCES (pattern, total, sequences, avg)\n")
        f.write("-" * 50 + "\n")

        for n, patterns in sorted(results['total_occurrences'].items()):
            if not patterns:
                continue
            f.write(f"\n{n}-GRAMS\n")

            for p in patterns:
                numeric_str = pattern_numbers_str(tuple(p['pattern']))
                f.write(f"{numeric_str}, {p['total']}, {p['sequences']}, {p['avg_per_seq']:.2f}\n")

        # Repeating patterns
        f.write("\n\nREPEATING PATTERNS (pattern, sequences, total_repeats)\n")
        f.write("-" * 50 + "\n")

        repeating = results['repeating_patterns']
        if repeating:
            for p in repeating:
                numeric_str = pattern_numbers_str(tuple(p['pattern']))
                f.write(f"{numeric_str}, {p['sequences']}, {p['total_repeats']}\n")
        else:
            f.write("No repeating patterns found\n")

    print(f"Saved comprehensive results to {output_file}")


def save_patterns_to_txt(prefix, results, tactic_stats):
    """Save pattern analysis results to a human-readable text file (summary)."""
    output_file = f'{prefix}_tactic_patterns.txt'

    with open(output_file, 'w') as f:
        f.write("TACTIC PATTERN ANALYSIS (SUMMARY)\n")
        f.write("=================================\n\n")
        f.write("Numeric encoding (0-63): B=32, F=16, S=8, K=4, W=2, T=1\n")
        f.write("Example: B---W_F = 34, C-S-W_F = 10\n\n")
        f.write("Format: pattern, sequences\n\n")

        # Frequent n-grams - top 30 per length
        for n, patterns in sorted(results['frequent_ngrams'].items()):
            if not patterns:
                continue
            f.write(f"\n{n}-GRAMS (top 30)\n")
            f.write("-" * 40 + "\n")

            for pattern, count in patterns[:30]:
                numeric_str = pattern_numbers_str(pattern)
                f.write(f"{numeric_str}, {count}\n")

        # Repeating patterns - top 50
        f.write("\n\nREPEATING PATTERNS (top 50)\n")
        f.write("-" * 40 + "\n")
        f.write("pattern, sequences, total_repeats\n\n")

        repeating = results['repeating_patterns']
        if repeating:
            for p in repeating[:50]:
                numeric_str = pattern_numbers_str(tuple(p['pattern']))
                f.write(f"{numeric_str}, {p['sequences']}, {p['total_repeats']}\n")
        else:
            f.write("No repeating patterns found\n")

        # Pattern details - top 10 with descriptions
        f.write("\n\nTOP 10 PATTERNS WITH DESCRIPTIONS\n")
        f.write("=" * 50 + "\n")

        top_patterns = []
        for n in [3, 4, 5, 6]:
            if n in results['frequent_ngrams'] and results['frequent_ngrams'][n]:
                top_patterns.extend([(p, c, n) for p, c in results['frequent_ngrams'][n][:3]])

        top_patterns.sort(key=lambda x: -x[1])

        for pattern, count, n in top_patterns[:10]:
            numeric_str = pattern_numbers_str(pattern)
            f.write(f"\n{numeric_str} ({count} sequences)\n")
            for i, tactic in enumerate(pattern):
                num = tactic_to_number(tactic)
                desc = decode_tactic(tactic)
                f.write(f"  {i+1}. #{num} {tactic}: {desc}\n")

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
    print("\n" + "="*60)
    print("FREQUENT TACTIC SEQUENCES (pattern, sequences)")
    print("="*60)

    for n in range(min_length, min(max_length + 1, 13)):  # Print up to 12-grams
        print(f"\n{n}-grams:")
        frequent = find_frequent_ngrams(sequences, n, min_count=1000)

        if frequent:
            for pattern, count in frequent[:8]:
                numeric_str = pattern_numbers_str(pattern)
                print(f"{numeric_str}, {count}")
        else:
            print("(none)")

    # Find n-grams with total occurrence counts
    print("\n" + "="*60)
    print("TOTAL OCCURRENCES (pattern, total, sequences, avg)")
    print("="*60)

    for n in range(min_length, min(max_length + 1, 8)):  # Print up to 7-grams
        print(f"\n{n}-grams:")
        frequent_total = find_frequent_ngrams_total(sequences, n, min_count=5000)

        if frequent_total:
            for pattern, total_count, seq_count in frequent_total[:8]:
                numeric_str = pattern_numbers_str(pattern)
                avg_per_seq = total_count / seq_count if seq_count > 0 else 0
                print(f"{numeric_str}, {total_count}, {seq_count}, {avg_per_seq:.2f}")
        else:
            print("(none)")

    # Find repeating patterns
    print("\n" + "="*60)
    print("REPEATING PATTERNS (pattern, sequences, total_repeats)")
    print("="*60)

    repeating = find_all_repeating_patterns(sequences, min_length, max_length, min_occurrences=500)

    if repeating:
        print(f"Found {len(repeating)} repeating patterns\n")
        for pattern, seq_count, total_repeats in repeating[:15]:
            numeric_str = pattern_numbers_str(pattern)
            print(f"{numeric_str}, {seq_count}, {total_repeats}")
    else:
        print("No repeating patterns found")

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
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Most common starting tactics
    start_tactics = Counter(seq[0] for seq in sequences)
    print(f"\nStarting tactics (tactic, count):")
    for tactic, count in start_tactics.most_common(5):
        num = tactic_to_number(tactic)
        print(f"  {num}, {count}")

    # Most common ending tactics
    end_tactics = Counter(seq[-1] for seq in sequences)
    print(f"\nEnding tactics (tactic, count):")
    for tactic, count in end_tactics.most_common(5):
        num = tactic_to_number(tactic)
        print(f"  {num}, {count}")

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

    print(f"\nSelf-loops: {self_loops:,} / {total_transitions:,} ({100*self_loops/total_transitions:.1f}%)")

    # Most common self-loop tactics
    self_loop_counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            if seq[i] == seq[i + 1]:
                self_loop_counts[seq[i]] += 1

    if self_loop_counts:
        print(f"\nSelf-loop tactics (tactic, count):")
        for tactic, count in self_loop_counts.most_common(10):
            num = tactic_to_number(tactic)
            print(f"  {num}, {count}")

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
