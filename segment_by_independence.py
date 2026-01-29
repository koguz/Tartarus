#!/usr/bin/env python3
"""
Segment long tactic patterns using Independence Ratio Segmentation (IRS).

Algorithm: Independence Ratio Segmentation (IRS)
================================================
For a long pattern with frequency N, we find sub-sequences that exist
independently with frequency M. The ratio M/N indicates how "independent"
or "reusable" that sub-sequence is:

  - M/N >> 1: Sub-sequence appears in MANY other contexts → reusable component
  - M/N ≈ 1:  Sub-sequence mostly appears in THIS context → specific to this behavior
  - M/N < 1:  Shouldn't happen for valid sub-sequences

The algorithm:
1. For a long n-gram, check all possible sub-sequences from longest to shortest
2. Compute M/N ratio for each
3. Use DP to find optimal coverage favoring: longer segments × higher ratios

Usage:
    python segment_by_independence.py [analysis_prefix] [--ratio-threshold T]

Output:
    {prefix}_behaviors_irs.txt - Segmented behaviors
    {prefix}_behaviors_irs.json - Structured output
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
    frequencies = {}

    for n in range(1, max_n + 1):  # Include 1-grams
        for seq in sequences:
            if len(seq) < n:
                continue
            for i in range(len(seq) - n + 1):
                ngram = tuple(seq[i:i+n])
                if ngram not in frequencies:
                    frequencies[ngram] = 0
                frequencies[ngram] += 1

    return frequencies


def pattern_str(pattern):
    """Convert pattern to string representation."""
    nums = [tactic_to_number(t) if isinstance(t, str) else t for t in pattern]
    return '-'.join(str(n) for n in nums)


def irs_segment(sequence, frequencies, original_freq, ratio_threshold=2.0):
    """
    Independence Ratio Segmentation (IRS) using dynamic programming.

    For a pattern with frequency N, a sub-sequence with frequency M is
    considered "independent/reusable" if M/N > ratio_threshold.

    DP finds the optimal segmentation that maximizes: sum of (ratio × length)
    This naturally favors longer segments with good independence ratios.

    Args:
        sequence: tuple of tactics to segment
        frequencies: dict mapping n-gram tuples to their frequencies
        original_freq: N, the frequency of the full sequence
        ratio_threshold: minimum M/N ratio to be considered "independent"

    Returns:
        (score, segmentation) where segmentation is list of dicts
    """
    n = len(sequence)
    if n == 0:
        return (0, [])

    N = original_freq
    INF = float('-inf')

    # best[i] = (score, segmentation) for sequence[0:i]
    best = [(INF, [])] * (n + 1)
    best[0] = (0, [])

    for i in range(1, n + 1):
        # Try segments ending at position i, from longest to shortest
        for seg_len in range(i, 0, -1):
            j = i - seg_len
            segment = sequence[j:i]

            M = frequencies.get(segment, 0)
            if M == 0:
                continue

            ratio = M / N

            # Score: ratio × length
            # Longer segments with higher ratios get higher scores
            score = ratio * seg_len

            candidate_score = best[j][0] + score

            if candidate_score > best[i][0]:
                # Classify the segment
                if ratio >= ratio_threshold:
                    seg_type = "independent"
                elif ratio >= 1.2:
                    seg_type = "shared"
                else:
                    seg_type = "specific"

                new_seg = {
                    'start': j,
                    'end': i,
                    'pattern': pattern_str(segment),
                    'length': seg_len,
                    'freq': M,
                    'ratio': ratio,
                    'type': seg_type
                }
                new_segmentation = best[j][1] + [new_seg]
                best[i] = (candidate_score, new_segmentation)

    # Fallback if no segmentation found
    if best[n][0] == INF:
        segmentation = []
        for i in range(n):
            seg = (sequence[i],)
            M = frequencies.get(seg, 1)
            ratio = M / N
            segmentation.append({
                'start': i,
                'end': i + 1,
                'pattern': pattern_str(seg),
                'length': 1,
                'freq': M,
                'ratio': ratio,
                'type': 'specific'
            })
        return (0, segmentation)

    return best[n]


def segment_long_patterns(frequencies, min_behavior_length=6,
                          min_behavior_freq=500, ratio_threshold=2.0):
    """Find and segment all long patterns using IRS."""
    behaviors = []

    # Find all long patterns
    long_patterns = []
    for pattern, freq in frequencies.items():
        if len(pattern) >= min_behavior_length and freq >= min_behavior_freq:
            long_patterns.append((pattern, freq))

    # Sort by length (longest first), then by frequency
    long_patterns.sort(key=lambda x: (-len(x[0]), -x[1]))

    print(f"Found {len(long_patterns)} long patterns to segment")

    for pattern, freq in long_patterns:
        score, segmentation = irs_segment(pattern, frequencies, freq, ratio_threshold)

        behaviors.append({
            'pattern': pattern_str(pattern),
            'length': len(pattern),
            'frequency': freq,
            'score': score,
            'segments': segmentation,
            'num_segments': len(segmentation),
            'num_independent': sum(1 for s in segmentation if s['type'] == 'independent'),
            'num_specific': sum(1 for s in segmentation if s['type'] == 'specific')
        })

    return behaviors


def save_results(behaviors, prefix):
    """Save results to files."""

    # JSON
    json_file = f'{prefix}_behaviors_irs.json'
    with open(json_file, 'w') as f:
        json.dump(behaviors, f, indent=2)
    print(f"Saved {len(behaviors)} behaviors to {json_file}")

    # Human-readable
    txt_file = f'{prefix}_behaviors_irs.txt'
    with open(txt_file, 'w') as f:
        f.write("INDEPENDENCE RATIO SEGMENTATION (IRS) RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Segment types:\n")
        f.write("  [I] Independent: M/N > threshold, reusable component\n")
        f.write("  [S] Shared: 1.2 < M/N < threshold\n")
        f.write("  [.] Specific: M/N ≈ 1, unique to this behavior\n\n")

        by_length = defaultdict(list)
        for b in behaviors:
            by_length[b['length']].append(b)

        for length in sorted(by_length.keys(), reverse=True):
            f.write(f"\n{'='*60}\n")
            f.write(f"{length}-GRAM BEHAVIORS\n")
            f.write(f"{'='*60}\n\n")

            for b in sorted(by_length[length], key=lambda x: -x['frequency']):
                f.write(f"{b['pattern']} (freq={b['frequency']})\n")

                parts = []
                for seg in b['segments']:
                    marker = {'independent': 'I', 'shared': 'S', 'specific': '.'}[seg['type']]
                    parts.append(f"[{seg['pattern']}]({marker}, r={seg['ratio']:.1f})")

                f.write(f"  -> {' + '.join(parts)}\n\n")

    print(f"Saved human-readable results to {txt_file}")

    # Simple format
    simple_file = f'{prefix}_behaviors_irs_simple.txt'
    with open(simple_file, 'w') as f:
        f.write("# pattern, frequency, num_segments, segmentation\n")
        for b in sorted(behaviors, key=lambda x: (-x['length'], -x['frequency'])):
            seg_str = ' + '.join(s['pattern'] for s in b['segments'])
            f.write(f"{b['pattern']}, {b['frequency']}, {b['num_segments']}, {seg_str}\n")

    print(f"Saved simple format to {simple_file}")


def main(prefix='analysis', min_behavior_length=6, min_behavior_freq=500,
         ratio_threshold=2.0, max_ngram=12):
    """Main function."""

    print(f"Loading tactic sequences from {prefix}_tactic_sequences.pkl...")
    sequences = load_tactic_sequences(prefix)
    print(f"Loaded {len(sequences)} sequences")

    print("Collapsing repetitions...")
    collapsed = collapse_all_sequences(sequences)

    orig_len = sum(len(s) for s in sequences)
    coll_len = sum(len(s) for s in collapsed)
    print(f"  Compression: {orig_len} -> {coll_len} ({orig_len/coll_len:.2f}x)")

    print(f"\nComputing n-gram frequencies (n=1 to {max_ngram})...")
    collapsed_numeric = [tuple(tactic_to_number(t) for t in seq) for seq in collapsed]
    frequencies = compute_ngram_frequencies(collapsed_numeric, max_ngram)
    print(f"  Found {len(frequencies)} unique n-grams")

    print(f"\nSegmenting with IRS (ratio_threshold={ratio_threshold})...")
    behaviors = segment_long_patterns(
        frequencies,
        min_behavior_length=min_behavior_length,
        min_behavior_freq=min_behavior_freq,
        ratio_threshold=ratio_threshold
    )

    print(f"\nResults:")
    print(f"  Total behaviors: {len(behaviors)}")
    if behaviors:
        avg_seg = sum(b['num_segments'] for b in behaviors) / len(behaviors)
        avg_ind = sum(b['num_independent'] for b in behaviors) / len(behaviors)
        print(f"  Avg segments: {avg_seg:.1f}")
        print(f"  Avg independent components: {avg_ind:.1f}")

    save_results(behaviors, prefix)

    # Print top 10
    print("\n" + "="*60)
    print("TOP 10 BEHAVIORS")
    print("="*60)

    for b in behaviors[:10]:
        print(f"\n{b['pattern']} (freq={b['frequency']}, len={b['length']})")
        for seg in b['segments']:
            marker = {'independent': 'INDEP', 'shared': 'SHARED', 'specific': 'SPEC'}[seg['type']]
            print(f"  [{seg['pattern']}] {marker} (freq={seg['freq']}, ratio={seg['ratio']:.1f})")

    return behaviors


if __name__ == '__main__':
    prefix = 'analysis'
    min_behavior_length = 6
    min_behavior_freq = 500
    ratio_threshold = 2.0

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--min-length' and i + 1 < len(sys.argv):
            min_behavior_length = int(sys.argv[i + 1])
            i += 2
        elif arg == '--min-freq' and i + 1 < len(sys.argv):
            min_behavior_freq = int(sys.argv[i + 1])
            i += 2
        elif arg == '--ratio-threshold' and i + 1 < len(sys.argv):
            ratio_threshold = float(sys.argv[i + 1])
            i += 2
        elif arg == '--help':
            print(__doc__)
            print("\nOptions:")
            print("  --min-length N       Minimum behavior length (default: 6)")
            print("  --min-freq N         Minimum behavior frequency (default: 500)")
            print("  --ratio-threshold F  M/N ratio threshold for independence (default: 2.0)")
            sys.exit(0)
        elif not arg.startswith('-'):
            prefix = arg
            i += 1
        else:
            i += 1

    main(prefix, min_behavior_length, min_behavior_freq, ratio_threshold)
