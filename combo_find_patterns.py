#!/usr/bin/env python3
"""
Find recurring combination transition patterns in agent behavior.

This script analyzes the combination sequences from analyze_agent.py to find
common patterns (n-grams) that appear frequently across board runs.
These patterns represent emergent perceptual behaviors the agent has learned.

Usage:
    python combo_find_patterns.py [analysis_prefix] [--min-length N] [--max-length M]

Default: min-length=3, max-length=15

Output files:
    {prefix}_combo_patterns.json - All patterns in JSON format
    {prefix}_combo_patterns.txt - Human-readable summary (top patterns)
    {prefix}_combo_patterns_all.txt - Complete list of ALL patterns
"""

import pickle
import json
import sys
from collections import defaultdict, Counter

# Valid combination indices (maps combo_idx back to raw 8-cell encoding)
IIDX = [0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546]


def decode_combination(combo_idx):
    """Decode a combination index to 8 cell values (0=empty, 1=box, 2=wall)."""
    if combo_idx >= len(IIDX):
        return [0] * 8
    combo_raw = IIDX[combo_idx]
    cells = []
    for _ in range(8):
        cells.append(combo_raw % 3)
        combo_raw //= 3
    return cells


def describe_combination(combo_idx):
    """Get human-readable description of what the agent sees."""
    if combo_idx >= len(IIDX):
        return "invalid"
    combo_raw = IIDX[combo_idx]
    cells = decode_combination(combo_raw)
    # Physical direction names
    phys_names = ['L', 'BL', 'B', 'BR', 'R', 'FR', 'F', 'FL']
    val_names = ['_', 'B', 'W']  # empty, box, wall
    desc = ''.join(f"{phys_names[i]}:{val_names[cells[i]]} " for i in range(8) if cells[i] != 0)
    return desc.strip() if desc.strip() else "all_empty"


def load_sequences(prefix='analysis'):
    """Load combination sequences from pickle file."""
    with open(f'{prefix}_combo_sequences.pkl', 'rb') as f:
        sequences = pickle.load(f)
    return sequences


def load_combo_stats(prefix='analysis'):
    """Load combination statistics for context."""
    with open(f'{prefix}_combo_stats.json', 'r') as f:
        return json.load(f)


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
        # Count each unique n-gram once per sequence (to avoid over-counting loops)
        unique_ngrams = set(ngrams)
        for ng in unique_ngrams:
            ngram_counts[ng] += 1

    # Filter by minimum count
    frequent = [(ng, count) for ng, count in ngram_counts.most_common() if count >= min_count]
    return frequent


def find_frequent_ngrams_total(sequences, n, min_count=100):
    """Find the most frequent n-grams counting ALL occurrences.
    If a pattern appears 3 times in one sequence, it counts as 3."""
    ngram_counts = Counter()
    ngram_sequence_counts = Counter()  # How many sequences contain this pattern

    for seq in sequences:
        ngrams = extract_ngrams(seq, n)
        # Count every occurrence
        for ng in ngrams:
            ngram_counts[ng] += 1
        # Also track unique sequences containing this pattern
        for ng in set(ngrams):
            ngram_sequence_counts[ng] += 1

    # Filter by minimum total count
    frequent = [(ng, ngram_counts[ng], ngram_sequence_counts[ng])
                for ng in ngram_counts if ngram_counts[ng] >= min_count]
    frequent.sort(key=lambda x: -x[1])  # Sort by total occurrences
    return frequent


def find_repeating_patterns_in_sequence(sequence, min_length=3, max_length=15):
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


def find_all_repeating_patterns(sequences, min_length=3, max_length=15, min_occurrences=100):
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


def get_pattern_action_info(pattern, combo_stats):
    """Get action information for combinations in a pattern."""
    info = []
    for combo in pattern:
        stats = combo_stats.get(str(combo), {})
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


def save_all_patterns_to_txt(prefix, results, combo_stats):
    """Save ALL patterns to a comprehensive text file."""
    output_file = f'{prefix}_combo_patterns_all.txt'

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPLETE COMBINATION PATTERN ANALYSIS (ALL PATTERNS)\n")
        f.write("="*80 + "\n\n")

        # Frequent n-grams - ALL of them
        f.write("="*80 + "\n")
        f.write("ALL FREQUENT COMBINATION SEQUENCES (n-grams)\n")
        f.write("="*80 + "\n\n")

        for n, patterns in sorted(results['frequent_ngrams'].items()):
            if not patterns:
                continue
            f.write(f"\n{'='*80}\n")
            f.write(f"{n}-grams (sequences of {n} combinations)\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total patterns: {len(patterns)}\n\n")
            f.write(f"{'Pattern':<50} {'Sequences':>10} {'Actions':<15}\n")
            f.write("-" * 80 + "\n")

            for pattern, count in patterns:  # ALL patterns
                pattern_str = '-'.join(map(str, pattern))
                actions = get_pattern_action_info(pattern, combo_stats)
                f.write(f"{pattern_str:<50} {count:>10,} {actions:<15}\n")

        # Total occurrences - ALL of them
        f.write("\n" + "="*80 + "\n")
        f.write("ALL TOTAL PATTERN OCCURRENCES (counting all instances)\n")
        f.write("="*80 + "\n\n")

        for n, patterns in sorted(results['total_occurrences'].items()):
            if not patterns:
                continue
            f.write(f"\n{'='*80}\n")
            f.write(f"{n}-grams (total occurrences)\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total patterns: {len(patterns)}\n\n")
            f.write(f"{'Pattern':<45} {'Total Occ.':>12} {'In Sequences':>12} {'Avg/Seq':>10} {'Actions':<10}\n")
            f.write("-" * 95 + "\n")

            for p in patterns:  # ALL patterns
                pattern_str = '-'.join(map(str, p['pattern']))
                actions = get_pattern_action_info(tuple(p['pattern']), combo_stats)
                f.write(f"{pattern_str:<45} {p['total']:>12,} {p['sequences']:>12,} {p['avg_per_seq']:>10.2f} {actions:<10}\n")

        # Repeating patterns - ALL of them
        f.write("\n" + "="*80 + "\n")
        f.write("ALL REPEATING PATTERNS (loops within sequences)\n")
        f.write("="*80 + "\n\n")

        repeating = results['repeating_patterns']
        if repeating:
            f.write(f"Total repeating patterns: {len(repeating)}\n\n")
            f.write(f"{'Pattern':<45} {'Sequences':>10} {'Total Repeats':>15} {'Actions':<15}\n")
            f.write("-" * 90 + "\n")

            for p in repeating:  # ALL patterns
                pattern_str = '-'.join(map(str, p['pattern']))
                actions = get_pattern_action_info(tuple(p['pattern']), combo_stats)
                f.write(f"{pattern_str:<45} {p['sequences']:>10,} {p['total_repeats']:>15,} {actions:<15}\n")
        else:
            f.write("No repeating patterns found\n")

    print(f"Saved comprehensive results to {output_file}")


def save_patterns_to_txt(prefix, results, combo_stats):
    """Save pattern analysis results to a human-readable text file (summary version)."""
    output_file = f'{prefix}_combo_patterns.txt'

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMBINATION PATTERN ANALYSIS\n")
        f.write("="*80 + "\n\n")

        # Frequent n-grams
        f.write("="*80 + "\n")
        f.write("FREQUENT COMBINATION SEQUENCES (n-grams)\n")
        f.write("="*80 + "\n")
        f.write("These are common sequences of perceptions that appear across many boards.\n\n")

        for n, patterns in results['frequent_ngrams'].items():
            if not patterns:
                continue
            f.write(f"\n--- {n}-grams (sequences of {n} combinations) ---\n")
            f.write(f"Found {len(patterns)} patterns appearing in 500+ sequences\n\n")
            f.write(f"{'Pattern':<50} {'Sequences':>10} {'Actions':<15}\n")
            f.write("-" * 80 + "\n")

            for pattern, count in patterns[:20]:  # Top 20
                pattern_str = '-'.join(map(str, pattern))
                actions = get_pattern_action_info(pattern, combo_stats)
                f.write(f"{pattern_str:<50} {count:>10,} {actions:<15}\n")

        # Total occurrences
        f.write("\n" + "="*80 + "\n")
        f.write("TOTAL PATTERN OCCURRENCES (counting all instances)\n")
        f.write("="*80 + "\n")
        f.write("This counts every occurrence, even multiple times within one sequence.\n\n")

        for n, patterns in results['total_occurrences'].items():
            if not patterns:
                continue
            f.write(f"\n--- {n}-grams (total occurrences) ---\n")
            f.write(f"Found {len(patterns)} patterns with 5000+ total occurrences\n\n")
            f.write(f"{'Pattern':<45} {'Total Occ.':>12} {'In Sequences':>12} {'Avg/Seq':>10} {'Actions':<10}\n")
            f.write("-" * 95 + "\n")

            for p in patterns[:20]:  # Top 20
                pattern_str = '-'.join(map(str, p['pattern']))
                actions = get_pattern_action_info(tuple(p['pattern']), combo_stats)
                f.write(f"{pattern_str:<45} {p['total']:>12,} {p['sequences']:>12,} {p['avg_per_seq']:>10.2f} {actions:<10}\n")

        # Repeating patterns
        f.write("\n" + "="*80 + "\n")
        f.write("REPEATING PATTERNS (loops within sequences)\n")
        f.write("="*80 + "\n")
        f.write("These patterns repeat consecutively (like A-B-C-A-B-C-A-B-C).\n\n")

        repeating = results['repeating_patterns']
        if repeating:
            f.write(f"Found {len(repeating)} repeating patterns in 500+ sequences\n\n")
            f.write(f"{'Pattern':<45} {'Sequences':>10} {'Total Repeats':>15} {'Actions':<15}\n")
            f.write("-" * 90 + "\n")

            for p in repeating[:30]:  # Top 30
                pattern_str = '-'.join(map(str, p['pattern']))
                actions = get_pattern_action_info(tuple(p['pattern']), combo_stats)
                f.write(f"{pattern_str:<45} {p['sequences']:>10,} {p['total_repeats']:>15,} {actions:<15}\n")
        else:
            f.write("No significant repeating patterns found\n")

        # Pattern details section
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED PATTERN DESCRIPTIONS\n")
        f.write("="*80 + "\n")
        f.write("Top 10 most frequent patterns with full descriptions:\n\n")

        # Get top patterns from 3-grams and 4-grams
        top_patterns = []
        for n in [3, 4]:
            if n in results['frequent_ngrams'] and results['frequent_ngrams'][n]:
                top_patterns.extend([(p, c, n) for p, c in results['frequent_ngrams'][n][:5]])

        top_patterns.sort(key=lambda x: -x[1])

        for pattern, count, n in top_patterns[:10]:
            f.write(f"\nPattern: {'-'.join(map(str, pattern))}\n")
            f.write(f"  Length: {n}-gram\n")
            f.write(f"  Appears in: {count:,} sequences\n")
            f.write(f"  Actions: {get_pattern_action_info(pattern, combo_stats)}\n")
            f.write(f"  Details:\n")
            for i, combo in enumerate(pattern):
                desc = describe_combination(combo)
                f.write(f"    Step {i+1} (Combo {combo}): {desc}\n")
            f.write("\n")

    print(f"Saved human-readable results to {output_file}")


def analyze_patterns(prefix='analysis', min_length=3, max_length=15):
    """Main analysis function."""
    print(f"Loading combination sequences from {prefix}_combo_sequences.pkl...")
    sequences = load_sequences(prefix)
    print(f"Loaded {len(sequences)} sequences")

    print(f"Loading combination statistics...")
    combo_stats = load_combo_stats(prefix)

    total_combos = sum(len(seq) for seq in sequences)
    print(f"Total combination visits: {total_combos:,}")

    # Find frequent n-grams of various lengths
    print("\n" + "="*70)
    print("FREQUENT COMBINATION SEQUENCES (n-grams)")
    print("="*70)

    all_frequent_patterns = []

    for n in range(min_length, max_length + 1):
        print(f"\n--- {n}-grams (sequences of {n} combinations) ---")
        frequent = find_frequent_ngrams(sequences, n, min_count=1000)

        if frequent:
            print(f"Found {len(frequent)} patterns appearing in 1000+ sequences")
            print(f"{'Pattern':<50} {'Sequences':>10} {'Actions':<15}")
            print("-" * 80)

            for pattern, count in frequent[:15]:  # Top 15
                pattern_str = '-'.join(map(str, pattern))
                actions = get_pattern_action_info(pattern, combo_stats)
                print(f"{pattern_str:<50} {count:>10,} {actions:<15}")
                all_frequent_patterns.append((pattern, count, n))
        else:
            print("No patterns found with sufficient frequency")

    # Find n-grams with total occurrence counts
    print("\n" + "="*70)
    print("TOTAL PATTERN OCCURRENCES (counting all instances)")
    print("="*70)
    print("This counts every occurrence, even multiple times within one sequence.")

    for n in range(min_length, max_length + 1):
        print(f"\n--- {n}-grams (total occurrences) ---")
        frequent_total = find_frequent_ngrams_total(sequences, n, min_count=5000)

        if frequent_total:
            print(f"Found {len(frequent_total)} patterns with 5000+ total occurrences")
            print(f"{'Pattern':<45} {'Total Occ.':>12} {'In Sequences':>12} {'Avg/Seq':>10} {'Actions':<10}")
            print("-" * 95)

            for pattern, total_count, seq_count in frequent_total[:15]:  # Top 15
                pattern_str = '-'.join(map(str, pattern))
                actions = get_pattern_action_info(pattern, combo_stats)
                avg_per_seq = total_count / seq_count if seq_count > 0 else 0
                print(f"{pattern_str:<45} {total_count:>12,} {seq_count:>12,} {avg_per_seq:>10.2f} {actions:<10}")
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
        print(f"{'Pattern':<45} {'Sequences':>10} {'Total Repeats':>15} {'Actions':<15}")
        print("-" * 90)

        for pattern, seq_count, total_repeats in repeating[:20]:  # Top 20
            pattern_str = '-'.join(map(str, pattern))
            actions = get_pattern_action_info(pattern, combo_stats)
            print(f"{pattern_str:<45} {seq_count:>10,} {total_repeats:>15,} {actions:<15}")
    else:
        print("No significant repeating patterns found")

    # Save results to JSON (save ALL patterns, no limits)
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

    output_file_json = f'{prefix}_combo_patterns.json'
    with open(output_file_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {output_file_json}")

    # Save results to TXT (human-readable summary)
    save_patterns_to_txt(prefix, results, combo_stats)

    # Save ALL patterns to comprehensive TXT file
    save_all_patterns_to_txt(prefix, results, combo_stats)

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Most common starting combinations
    start_combos = Counter(seq[0] for seq in sequences)
    print(f"\nMost common starting combinations:")
    for combo, count in start_combos.most_common(5):
        pct = 100 * count / len(sequences)
        desc = describe_combination(combo)
        print(f"  Combo {combo} ({desc}): {count:,} ({pct:.1f}%)")

    # Most common ending combinations
    end_combos = Counter(seq[-1] for seq in sequences)
    print(f"\nMost common ending combinations (after 80 moves):")
    for combo, count in end_combos.most_common(5):
        pct = 100 * count / len(sequences)
        desc = describe_combination(combo)
        print(f"  Combo {combo} ({desc}): {count:,} ({pct:.1f}%)")

    # Average unique combinations per sequence
    unique_per_seq = [len(set(seq)) for seq in sequences]
    avg_unique = sum(unique_per_seq) / len(unique_per_seq)
    print(f"\nAverage unique combinations visited per sequence: {avg_unique:.1f}")

    # Check for consecutive repeats (same combo staying)
    total_transitions = 0
    self_loops = 0
    for seq in sequences:
        for i in range(len(seq) - 1):
            total_transitions += 1
            if seq[i] == seq[i + 1]:
                self_loops += 1

    print(f"\nSelf-loop analysis (agent staying in same combination):")
    print(f"  Total transitions: {total_transitions:,}")
    print(f"  Self-loops (same combo): {self_loops:,} ({100*self_loops/total_transitions:.2f}%)")
    print(f"  Combo changes: {total_transitions - self_loops:,} ({100*(total_transitions-self_loops)/total_transitions:.2f}%)")

    # Find most common self-loop combos
    self_loop_counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            if seq[i] == seq[i + 1]:
                self_loop_counts[seq[i]] += 1

    if self_loop_counts:
        print(f"\nMost common self-loop combinations (combos that repeat consecutively):")
        for combo, count in self_loop_counts.most_common(10):
            desc = describe_combination(combo)
            print(f"  Combo {combo} ({desc}): {count:,} consecutive repeats")

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
