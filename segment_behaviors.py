#!/usr/bin/env python3
"""
Segment long tactic patterns into reusable components using dynamic programming.

This script:
1. Loads tactic sequences and collapses repetitions
2. Computes n-gram frequencies on collapsed sequences
3. Uses DP to find optimal segmentation of long patterns
4. Outputs hierarchical behavior decomposition

Usage:
    python segment_behaviors.py [analysis_prefix] [--min-length N] [--threshold T]

Output:
    {prefix}_behaviors.txt - Segmented behaviors
    {prefix}_behaviors.json - Structured output
"""

import pickle
import json
import sys
from collections import Counter, defaultdict


def tactic_to_number(tactic):
    """Convert tactic string to number (0-63)."""
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
    """Convert number (0-63) to tactic string."""
    if n < 0 or n > 63:
        return "invalid"
    p1 = 'B' if (n & 32) else 'C'
    p2 = 'F' if (n & 16) else '-'
    p3 = 'S' if (n & 8) else '-'
    p4 = 'K' if (n & 4) else '-'
    p5 = 'W' if (n & 2) else 'O'
    p6 = 'T' if (n & 1) else 'F'
    return f"{p1}{p2}{p3}{p4}{p5}_{p6}"


def load_tactic_sequences(prefix='analysis'):
    """Load tactic sequences from pickle file."""
    with open(f'{prefix}_tactic_sequences.pkl', 'rb') as f:
        return pickle.load(f)


def collapse_sequence(seq):
    """Collapse consecutive repetitions in a sequence."""
    if not seq:
        return []
    collapsed = [seq[0]]
    for item in seq[1:]:
        if item != collapsed[-1]:
            collapsed.append(item)
    return collapsed


def collapse_all_sequences(sequences):
    """Collapse all sequences."""
    return [collapse_sequence(seq) for seq in sequences]


def compute_ngram_frequencies(sequences, max_n=12):
    """Compute frequencies of all n-grams (n=2 to max_n)."""
    frequencies = {}  # tuple -> count

    for n in range(2, max_n + 1):
        for seq in sequences:
            if len(seq) < n:
                continue
            for i in range(len(seq) - n + 1):
                ngram = tuple(seq[i:i+n])
                if ngram not in frequencies:
                    frequencies[ngram] = 0
                frequencies[ngram] += 1

    return frequencies


def pattern_to_numbers(pattern):
    """Convert pattern tuple to list of numbers."""
    return [tactic_to_number(t) if isinstance(t, str) else t for t in pattern]


def pattern_str(pattern):
    """Convert pattern to string representation."""
    nums = pattern_to_numbers(pattern)
    return '-'.join(str(n) for n in nums)


def dp_segment(sequence, frequencies, min_pattern_freq=100, segment_penalty=2.0, length_bonus=0.5):
    """
    Use dynamic programming to find optimal segmentation.

    Args:
        sequence: tuple of tactics to segment
        frequencies: dict mapping n-gram tuples to their frequencies
        min_pattern_freq: minimum frequency for a pattern to be in dictionary
        segment_penalty: penalty for each segment break (encourages fewer, longer segments)
        length_bonus: bonus multiplier for segment length (encourages longer segments)

    Returns:
        (score, segmentation) where segmentation is list of (start, end, pattern, freq)
    """
    n = len(sequence)
    if n == 0:
        return (0, [])

    # best[i] = (score, segmentation) for sequence[0:i]
    # score = sum of (log(freq) + length_bonus * length) - segment_penalty per segment
    # Higher score = better

    import math

    INF = float('-inf')
    best = [(INF, [])] * (n + 1)
    best[0] = (0, [])

    for i in range(1, n + 1):
        # Try all possible last segments ending at position i
        for j in range(0, i):
            segment = sequence[j:i]
            segment_len = i - j

            if segment_len == 1:
                # Single element - allow with penalty
                freq = frequencies.get(segment, 1)
                score = math.log(freq + 1) * 0.3 - segment_penalty
            else:
                # Multi-element pattern - check dictionary
                freq = frequencies.get(segment, 0)
                if freq < min_pattern_freq:
                    continue  # Skip patterns that are too rare
                # Score = log(freq) + bonus for length - penalty for creating a segment
                score = math.log(freq + 1) + (length_bonus * segment_len) - segment_penalty

            candidate_score = best[j][0] + score

            if candidate_score > best[i][0]:
                new_segmentation = best[j][1] + [(j, i, segment, freq)]
                best[i] = (candidate_score, new_segmentation)

    # If no valid segmentation found, fall back to single elements
    if best[n][0] == INF:
        segmentation = [(i, i+1, (sequence[i],), frequencies.get((sequence[i],), 1))
                        for i in range(n)]
        return (0, segmentation)

    return best[n]


def analyze_segmentation(segmentation, original_freq):
    """Analyze a segmentation to identify known vs specific components."""
    result = []
    for start, end, pattern, freq in segmentation:
        length = end - start
        ratio = freq / original_freq if original_freq > 0 else 0

        # Classify: known component (ratio > 2) vs specific (ratio close to 1)
        if ratio > 2:
            component_type = "known"
        elif ratio > 1.2:
            component_type = "shared"
        else:
            component_type = "specific"

        result.append({
            'start': start,
            'end': end,
            'pattern': pattern_str(pattern),
            'length': length,
            'frequency': freq,
            'ratio': ratio,
            'type': component_type
        })

    return result


def segment_long_patterns(sequences, frequencies, min_behavior_length=6,
                          min_behavior_freq=500, min_pattern_freq=100,
                          segment_penalty=2.0, length_bonus=0.5):
    """
    Find and segment all long patterns.

    Args:
        sequences: collapsed sequences
        frequencies: n-gram frequency dict
        min_behavior_length: minimum length to consider as behavior
        min_behavior_freq: minimum frequency for a behavior
        min_pattern_freq: minimum frequency for dictionary patterns
        segment_penalty: penalty for each segment (higher = fewer segments)
        length_bonus: bonus for longer segments (higher = prefer longer)

    Returns:
        List of segmented behaviors
    """
    behaviors = []

    # Find all long patterns that meet frequency threshold
    long_patterns = []
    for pattern, freq in frequencies.items():
        if len(pattern) >= min_behavior_length and freq >= min_behavior_freq:
            long_patterns.append((pattern, freq))

    # Sort by length (longest first), then by frequency
    long_patterns.sort(key=lambda x: (-len(x[0]), -x[1]))

    print(f"Found {len(long_patterns)} long patterns to segment")

    for pattern, freq in long_patterns:
        # Run DP segmentation
        score, segmentation = dp_segment(pattern, frequencies, min_pattern_freq,
                                         segment_penalty, length_bonus)

        # Analyze the segmentation
        analysis = analyze_segmentation(segmentation, freq)

        behaviors.append({
            'pattern': pattern_str(pattern),
            'length': len(pattern),
            'frequency': freq,
            'score': score,
            'segments': analysis,
            'num_segments': len(segmentation),
            'num_known': sum(1 for s in analysis if s['type'] == 'known'),
            'num_specific': sum(1 for s in analysis if s['type'] == 'specific')
        })

    return behaviors


def save_results(behaviors, prefix, frequencies):
    """Save results to files."""

    # Save JSON
    json_file = f'{prefix}_behaviors.json'
    with open(json_file, 'w') as f:
        json.dump(behaviors, f, indent=2)
    print(f"Saved {len(behaviors)} behaviors to {json_file}")

    # Save human-readable text
    txt_file = f'{prefix}_behaviors.txt'
    with open(txt_file, 'w') as f:
        f.write("BEHAVIOR SEGMENTATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Format: pattern (frequency) -> [segment1] + [segment2] + ...\n")
        f.write("Segment types: [known] = reusable, [specific] = unique to this behavior\n\n")

        # Group by length
        by_length = defaultdict(list)
        for b in behaviors:
            by_length[b['length']].append(b)

        for length in sorted(by_length.keys(), reverse=True):
            f.write(f"\n{'='*60}\n")
            f.write(f"{length}-GRAM BEHAVIORS\n")
            f.write(f"{'='*60}\n\n")

            for b in sorted(by_length[length], key=lambda x: -x['frequency']):
                f.write(f"{b['pattern']} (freq={b['frequency']})\n")

                # Show segmentation
                parts = []
                for seg in b['segments']:
                    type_marker = {'known': 'K', 'shared': 'S', 'specific': '.'}[seg['type']]
                    parts.append(f"[{seg['pattern']}]({type_marker},r={seg['ratio']:.1f})")

                f.write(f"  -> {' + '.join(parts)}\n")
                f.write(f"  Segments: {b['num_segments']} ({b['num_known']} known, {b['num_specific']} specific)\n\n")

    print(f"Saved human-readable results to {txt_file}")

    # Save simple CSV-like format
    csv_file = f'{prefix}_behaviors_simple.txt'
    with open(csv_file, 'w') as f:
        f.write("# pattern, frequency, num_segments, segmentation\n")
        for b in sorted(behaviors, key=lambda x: (-x['length'], -x['frequency'])):
            seg_str = ' + '.join(s['pattern'] for s in b['segments'])
            f.write(f"{b['pattern']}, {b['frequency']}, {b['num_segments']}, {seg_str}\n")

    print(f"Saved simple format to {csv_file}")


def main(prefix='analysis', min_behavior_length=6, min_behavior_freq=500,
         min_pattern_freq=100, max_ngram=12, segment_penalty=2.0, length_bonus=0.5):
    """Main analysis function."""

    print(f"Loading tactic sequences from {prefix}_tactic_sequences.pkl...")
    sequences = load_tactic_sequences(prefix)
    print(f"Loaded {len(sequences)} sequences")

    print("Collapsing repetitions...")
    collapsed = collapse_all_sequences(sequences)

    # Stats on collapsed sequences
    orig_lengths = [len(s) for s in sequences]
    collapsed_lengths = [len(s) for s in collapsed]
    print(f"  Original avg length: {sum(orig_lengths)/len(orig_lengths):.1f}")
    print(f"  Collapsed avg length: {sum(collapsed_lengths)/len(collapsed_lengths):.1f}")
    print(f"  Compression ratio: {sum(orig_lengths)/sum(collapsed_lengths):.2f}x")

    print(f"\nComputing n-gram frequencies (n=2 to {max_ngram})...")

    # Convert to tuples of numbers for consistency
    collapsed_numeric = []
    for seq in collapsed:
        numeric_seq = tuple(tactic_to_number(t) for t in seq)
        collapsed_numeric.append(numeric_seq)

    frequencies = compute_ngram_frequencies(collapsed_numeric, max_ngram)
    print(f"  Found {len(frequencies)} unique n-grams")

    # Stats on frequencies
    freq_by_length = defaultdict(list)
    for pattern, freq in frequencies.items():
        freq_by_length[len(pattern)].append(freq)

    print("\n  N-gram frequency summary:")
    for n in sorted(freq_by_length.keys()):
        freqs = freq_by_length[n]
        print(f"    {n}-grams: {len(freqs)} patterns, max={max(freqs)}, median={sorted(freqs)[len(freqs)//2]}")

    print(f"\nSegmenting behaviors (min_length={min_behavior_length}, min_freq={min_behavior_freq})...")
    print(f"  segment_penalty={segment_penalty}, length_bonus={length_bonus}")
    behaviors = segment_long_patterns(
        collapsed_numeric, frequencies,
        min_behavior_length=min_behavior_length,
        min_behavior_freq=min_behavior_freq,
        min_pattern_freq=min_pattern_freq,
        segment_penalty=segment_penalty,
        length_bonus=length_bonus
    )

    print(f"\nSegmentation complete!")
    print(f"  Total behaviors: {len(behaviors)}")

    # Summary stats
    if behaviors:
        avg_segments = sum(b['num_segments'] for b in behaviors) / len(behaviors)
        avg_known = sum(b['num_known'] for b in behaviors) / len(behaviors)
        print(f"  Avg segments per behavior: {avg_segments:.1f}")
        print(f"  Avg known components per behavior: {avg_known:.1f}")

    print("\nSaving results...")
    save_results(behaviors, prefix, frequencies)

    # Print top 10 behaviors
    print("\n" + "="*60)
    print("TOP 10 BEHAVIORS (by length, then frequency)")
    print("="*60)

    for b in behaviors[:10]:
        print(f"\n{b['pattern']} (freq={b['frequency']}, len={b['length']})")
        for seg in b['segments']:
            type_str = seg['type'].upper()
            print(f"  [{seg['pattern']}] {type_str} (freq={seg['frequency']}, ratio={seg['ratio']:.1f})")

    return behaviors


if __name__ == '__main__':
    prefix = 'analysis'
    min_behavior_length = 6
    min_behavior_freq = 500
    min_pattern_freq = 100
    segment_penalty = 2.0
    length_bonus = 0.5

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--min-length' and i + 1 < len(sys.argv):
            min_behavior_length = int(sys.argv[i + 1])
            i += 2
        elif arg == '--min-freq' and i + 1 < len(sys.argv):
            min_behavior_freq = int(sys.argv[i + 1])
            i += 2
        elif arg == '--min-pattern-freq' and i + 1 < len(sys.argv):
            min_pattern_freq = int(sys.argv[i + 1])
            i += 2
        elif arg == '--segment-penalty' and i + 1 < len(sys.argv):
            segment_penalty = float(sys.argv[i + 1])
            i += 2
        elif arg == '--length-bonus' and i + 1 < len(sys.argv):
            length_bonus = float(sys.argv[i + 1])
            i += 2
        elif arg == '--help':
            print(__doc__)
            print("\nOptions:")
            print("  --min-length N        Minimum behavior length (default: 6)")
            print("  --min-freq N          Minimum behavior frequency (default: 500)")
            print("  --min-pattern-freq N  Minimum pattern frequency for dictionary (default: 100)")
            print("  --segment-penalty F   Penalty per segment, higher = fewer segments (default: 2.0)")
            print("  --length-bonus F      Bonus per segment length, higher = longer segments (default: 0.5)")
            sys.exit(0)
        elif not arg.startswith('-'):
            prefix = arg
            i += 1
        else:
            i += 1

    main(prefix, min_behavior_length, min_behavior_freq, min_pattern_freq,
         segment_penalty=segment_penalty, length_bonus=length_bonus)
