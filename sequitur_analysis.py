"""
Sequitur Analysis for (Tactic, State) Sequences.

Applies the Sequitur grammar inference algorithm to discover
repeated behavioral patterns in the agent's (tactic, state) sequences.

Sequitur finds hierarchical structure by replacing repeated digrams
with grammar rules, building a compressed representation that reveals
the "building blocks" of behavior.
"""

import argparse
import pickle
import json
from collections import defaultdict


def tactic_to_decimal(tactic_str):
    """Convert tactic string to decimal (0-63)."""
    combo, action = tactic_str.split('_')
    n = 0
    if combo[0] == 'B': n += 32
    if combo[1] == 'F': n += 16
    if combo[2] == 'S': n += 8
    if combo[3] == 'K': n += 4
    if combo[4] == 'W': n += 2
    if action == 'T': n += 1
    return n


def decimal_to_tactic(n):
    """Convert decimal (0-63) to tactic string."""
    p1 = 'B' if (n & 32) else 'C'
    p2 = 'F' if (n & 16) else '-'
    p3 = 'S' if (n & 8) else '-'
    p4 = 'K' if (n & 4) else '-'
    p5 = 'W' if (n & 2) else 'O'
    p6 = 'T' if (n & 1) else 'F'
    return f"{p1}{p2}{p3}{p4}{p5}_{p6}"


class Symbol:
    """A symbol in the Sequitur grammar (terminal or non-terminal)."""
    def __init__(self, value, is_terminal=True):
        self.value = value
        self.is_terminal = is_terminal
        self.prev = None
        self.next = None
        self.rule = None  # For non-terminals, reference to the rule

    def __repr__(self):
        if self.is_terminal:
            return f"T({self.value})"
        else:
            return f"R{self.value}"


class Rule:
    """A grammar rule in Sequitur."""
    def __init__(self, rule_id):
        self.id = rule_id
        self.guard = Symbol(None, is_terminal=False)
        self.guard.rule = self
        self.guard.next = self.guard
        self.guard.prev = self.guard
        self.ref_count = 0  # How many times this rule is used

    def first(self):
        return self.guard.next

    def last(self):
        return self.guard.prev

    def append(self, symbol):
        """Append a symbol to the end of this rule."""
        symbol.prev = self.guard.prev
        symbol.next = self.guard
        self.guard.prev.next = symbol
        self.guard.prev = symbol

    def length(self):
        """Count symbols in this rule."""
        count = 0
        s = self.guard.next
        while s != self.guard:
            count += 1
            s = s.next
        return count

    def symbols(self):
        """Return list of symbols in this rule."""
        result = []
        s = self.guard.next
        while s != self.guard:
            result.append(s)
            s = s.next
        return result


class Sequitur:
    """
    Sequitur grammar inference algorithm.

    Maintains two invariants:
    1. Digram uniqueness: No digram appears more than once in the grammar
    2. Rule utility: Every rule is used at least twice
    """

    def __init__(self):
        self.main_rule = Rule(0)
        self.rules = {0: self.main_rule}
        self.next_rule_id = 1
        self.digram_index = {}  # (val1, val2) -> first symbol of digram

    def process(self, sequence):
        """Process a sequence of terminal symbols."""
        for item in sequence:
            self._append_terminal(item)

    def _append_terminal(self, value):
        """Append a terminal symbol to the main rule."""
        new_symbol = Symbol(value, is_terminal=True)
        self.main_rule.append(new_symbol)
        self._check_digram(new_symbol.prev)

    def _check_digram(self, symbol):
        """Check if a new digram needs to be processed."""
        if symbol == self.main_rule.guard or symbol.next == self.main_rule.guard:
            return

        digram = self._get_digram_key(symbol)
        if digram in self.digram_index:
            match = self.digram_index[digram]
            if match.next != symbol:  # Not the same digram
                self._process_match(symbol, match)
        else:
            self.digram_index[digram] = symbol

    def _get_digram_key(self, symbol):
        """Get a hashable key for a digram starting at symbol."""
        s1 = symbol
        s2 = symbol.next
        v1 = s1.value if s1.is_terminal else ('R', s1.value)
        v2 = s2.value if s2.is_terminal else ('R', s2.value)
        return (v1, v2)

    def _process_match(self, new_digram, existing_digram):
        """Handle a repeated digram."""
        # Check if existing digram is an entire rule
        if (existing_digram.prev.rule is not None and
            existing_digram.prev.rule == existing_digram.next.rule and
            existing_digram.prev.rule.length() == 2):
            # Use existing rule
            rule = existing_digram.prev.rule
            self._substitute(new_digram, rule)
        else:
            # Create new rule
            rule = Rule(self.next_rule_id)
            self.next_rule_id += 1
            self.rules[rule.id] = rule

            # Copy digram to new rule
            self._copy_digram_to_rule(existing_digram, rule)

            # Substitute both occurrences
            self._substitute(existing_digram, rule)
            self._substitute(new_digram, rule)

    def _copy_digram_to_rule(self, digram, rule):
        """Copy a digram's symbols to a rule."""
        s1 = digram
        s2 = digram.next

        new_s1 = Symbol(s1.value, s1.is_terminal)
        new_s2 = Symbol(s2.value, s2.is_terminal)

        if not s1.is_terminal:
            self.rules[s1.value].ref_count += 1
        if not s2.is_terminal:
            self.rules[s2.value].ref_count += 1

        rule.append(new_s1)
        rule.append(new_s2)

    def _substitute(self, digram, rule):
        """Replace a digram with a non-terminal."""
        s1 = digram
        s2 = digram.next

        # Remove old digrams from index
        if s1.prev != s1.prev.rule.guard if hasattr(s1.prev, 'rule') and s1.prev.rule else True:
            old_digram = self._get_digram_key(s1.prev) if s1.prev and s1.prev.next == s1 else None
            if old_digram and old_digram in self.digram_index:
                del self.digram_index[old_digram]

        current_digram = self._get_digram_key(s1)
        if current_digram in self.digram_index:
            del self.digram_index[current_digram]

        if s2.next and s2.next != s2.next.rule.guard if hasattr(s2.next, 'rule') and s2.next.rule else True:
            next_digram = self._get_digram_key(s2) if s2.next else None
            if next_digram and next_digram in self.digram_index:
                del self.digram_index[next_digram]

        # Decrease ref counts for non-terminals being removed
        if not s1.is_terminal:
            self.rules[s1.value].ref_count -= 1
            self._check_rule_utility(self.rules[s1.value])
        if not s2.is_terminal:
            self.rules[s2.value].ref_count -= 1
            self._check_rule_utility(self.rules[s2.value])

        # Create new non-terminal
        new_symbol = Symbol(rule.id, is_terminal=False)
        rule.ref_count += 1

        # Link new symbol
        new_symbol.prev = s1.prev
        new_symbol.next = s2.next
        s1.prev.next = new_symbol
        s2.next.prev = new_symbol

        # Check new digrams
        self._check_digram(new_symbol.prev)
        self._check_digram(new_symbol)

    def _check_rule_utility(self, rule):
        """Check if a rule is still useful (used at least twice)."""
        if rule.ref_count < 2 and rule.id != 0:
            # Inline the rule
            pass  # Simplified - full implementation would inline

    def get_grammar(self):
        """Return the grammar as a dictionary."""
        grammar = {}
        for rule_id, rule in self.rules.items():
            symbols = []
            for s in rule.symbols():
                if s.is_terminal:
                    symbols.append(s.value)
                else:
                    symbols.append(f"R{s.value}")
            grammar[f"R{rule_id}"] = symbols
        return grammar


def simple_sequitur(sequence, min_count=2):
    """
    Simplified Sequitur-like algorithm for finding repeated patterns.

    Instead of full grammar inference, finds repeated digrams and longer patterns.
    """
    # Count all digrams
    digram_counts = defaultdict(int)
    digram_positions = defaultdict(list)

    for i in range(len(sequence) - 1):
        digram = (sequence[i], sequence[i + 1])
        digram_counts[digram] += 1
        digram_positions[digram].append(i)

    # Filter to repeated digrams
    repeated_digrams = {d: c for d, c in digram_counts.items() if c >= min_count}

    # Try to extend digrams to longer patterns
    patterns = {}
    for digram, count in sorted(repeated_digrams.items(), key=lambda x: -x[1]):
        positions = digram_positions[digram]

        # Try to extend this pattern
        for length in range(3, 20):  # Try patterns up to length 20
            extended_count = 0
            pattern = None

            for pos in positions:
                if pos + length <= len(sequence):
                    candidate = tuple(sequence[pos:pos + length])
                    if pattern is None:
                        pattern = candidate
                        extended_count = 1
                    elif candidate == pattern:
                        extended_count += 1

            if extended_count >= min_count and pattern:
                pattern_key = pattern
                if pattern_key not in patterns or patterns[pattern_key] < extended_count:
                    patterns[pattern_key] = extended_count
            else:
                break

    return repeated_digrams, patterns


def analyze_sequences(tactic_sequences, state_sequences, min_pattern_count=100, max_sequences=None):
    """
    Analyze (tactic, state) sequences to find repeated patterns.
    """
    print("Combining tactic and state sequences...")

    # Combine into (tactic, state) sequences
    combined_sequences = []
    n_sequences = len(tactic_sequences)
    if max_sequences:
        n_sequences = min(n_sequences, max_sequences)

    for i in range(n_sequences):
        tactic_seq = tactic_sequences[i]
        state_seq = state_sequences[i]

        # State sequence has one extra element (initial state)
        # Align: state_seq[j] is the state when tactic_seq[j] is executed
        combined = []
        for j in range(len(tactic_seq)):
            tactic_dec = tactic_to_decimal(tactic_seq[j])
            state = state_seq[j]
            combined.append((tactic_dec, state))

        combined_sequences.append(combined)

    print(f"  Combined {len(combined_sequences)} sequences")

    # Concatenate all sequences with a separator
    all_symbols = []
    SEP = (-1, -1)  # Separator between sequences

    for seq in combined_sequences:
        all_symbols.extend(seq)
        all_symbols.append(SEP)

    print(f"  Total symbols: {len(all_symbols)}")

    # Find repeated patterns
    print(f"\nFinding repeated patterns (min count: {min_pattern_count})...")
    digrams, patterns = simple_sequitur(all_symbols, min_count=min_pattern_count)

    # Filter out patterns containing separator
    patterns = {p: c for p, c in patterns.items() if SEP not in p}
    digrams = {d: c for d, c in digrams.items() if SEP not in d}

    return digrams, patterns, combined_sequences


def format_pattern(pattern):
    """Format a pattern for display."""
    parts = []
    for tactic, state in pattern:
        parts.append(f"({tactic},{state})")
    return " -> ".join(parts)


def save_results(digrams, patterns, output_path):
    """Save results to JSON."""
    # Convert tuple keys to strings for JSON
    digrams_out = {
        f"({d[0][0]},{d[0][1]})->({d[1][0]},{d[1][1]})": count
        for d, count in sorted(digrams.items(), key=lambda x: -x[1])[:100]
    }

    patterns_out = []
    for pattern, count in sorted(patterns.items(), key=lambda x: (-len(x[0]), -x[1]))[:200]:
        patterns_out.append({
            'pattern': [{'tactic': t, 'state': s} for t, s in pattern],
            'length': len(pattern),
            'count': count,
            'readable': format_pattern(pattern)
        })

    output = {
        'top_digrams': digrams_out,
        'patterns': patterns_out,
        'stats': {
            'total_unique_digrams': len(digrams),
            'total_patterns_found': len(patterns)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(
        description='Sequitur analysis for (tactic, state) sequences'
    )
    parser.add_argument(
        '--tactic-sequences', '-t',
        type=str,
        default='analysis_tactic_sequences.pkl',
        help='Tactic sequences pickle file'
    )
    parser.add_argument(
        '--state-sequences', '-s',
        type=str,
        default='analysis_sequences.pkl',
        help='State sequences pickle file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='sequitur_patterns.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--min-count', '-m',
        type=int,
        default=100,
        help='Minimum pattern occurrence count (default: 100)'
    )
    parser.add_argument(
        '--max-sequences', '-n',
        type=int,
        default=None,
        help='Maximum number of sequences to analyze (default: all)'
    )

    args = parser.parse_args()

    # Load sequences
    print("Loading sequences...")
    with open(args.tactic_sequences, 'rb') as f:
        tactic_sequences = pickle.load(f)
    print(f"  Loaded {len(tactic_sequences)} tactic sequences")

    with open(args.state_sequences, 'rb') as f:
        state_sequences = pickle.load(f)
    print(f"  Loaded {len(state_sequences)} state sequences")

    # Analyze
    digrams, patterns, _ = analyze_sequences(
        tactic_sequences, state_sequences,
        min_pattern_count=args.min_count,
        max_sequences=args.max_sequences
    )

    # Print results
    print(f"\n{'='*60}")
    print("SEQUITUR ANALYSIS RESULTS")
    print("="*60)

    print(f"\nFound {len(digrams)} repeated digrams")
    print(f"Found {len(patterns)} repeated patterns (length >= 3)")

    print(f"\nTop 20 most common digrams:")
    for digram, count in sorted(digrams.items(), key=lambda x: -x[1])[:20]:
        (t1, s1), (t2, s2) = digram
        print(f"  ({t1},{s1}) -> ({t2},{s2}): {count:,} times")

    print(f"\nTop 20 longest/most common patterns:")
    sorted_patterns = sorted(patterns.items(), key=lambda x: (-len(x[0]), -x[1]))[:20]
    for pattern, count in sorted_patterns:
        print(f"  Length {len(pattern)}, count {count:,}:")
        print(f"    {format_pattern(pattern)}")

    # Analyze pattern structure
    print(f"\n{'='*60}")
    print("PATTERN LENGTH DISTRIBUTION")
    print("="*60)

    length_dist = defaultdict(int)
    for pattern in patterns:
        length_dist[len(pattern)] += 1

    for length in sorted(length_dist.keys()):
        print(f"  Length {length}: {length_dist[length]} patterns")

    # Save results
    save_results(digrams, patterns, args.output)


if __name__ == '__main__':
    main()
