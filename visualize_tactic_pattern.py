#!/usr/bin/env python3
"""
Visualize tactic patterns as a sequence of 3x3 grids.

Usage:
    python visualize_tactic_pattern.py --pattern B---W_F-B---W_F-B---W_T
    python visualize_tactic_pattern.py --pattern 34-34-35
    python visualize_tactic_pattern.py --numeric 34-34-35

The visualization shows:
- 3x3 grid with agent in center
- Orange cell in front if box in front (B)
- 50% transparent orange for front area (F), sides (S), back (K) boxes
- Thick border if wall present (W)
- Action (FORWARD or TURN) written on top

Tactic encoding:
    {B/C}{F/-}{S/-}{K/-}{W/O}_{F/T}
    B=box in front, C=clear, F=front area boxes, S=side boxes, K=back boxes
    W=wall, O=open, _F=forward, _T=turn

Numeric encoding (0-63):
    B=32, F=16, S=8, K=4, W=2, T=1
    Example: B---W_F = 100010 = 34
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def parse_tactic(s):
    """Parse a tactic - either as string or number."""
    s = s.strip()
    # Try as number first
    try:
        n = int(s)
        if 0 <= n <= 63:
            return number_to_tactic(n)
    except ValueError:
        pass
    # Otherwise treat as tactic string
    if len(s) >= 7 and s[5] == '_':
        return s
    return None


def parse_pattern(pattern_str):
    """Parse a pattern string like 'B---W_F-B---W_T' or '34-34-35'."""
    # Split by - but be careful with tactic strings that contain -
    parts = []
    current = ""

    for char in pattern_str:
        if char == '-':
            if current:
                # Check if current looks like a complete tactic or number
                parsed = parse_tactic(current)
                if parsed:
                    parts.append(parsed)
                    current = ""
                else:
                    current += char
            # else skip the dash
        else:
            current += char

    # Don't forget the last part
    if current:
        parsed = parse_tactic(current)
        if parsed:
            parts.append(parsed)

    return parts


def decode_tactic_flags(tactic):
    """Decode tactic string into component flags."""
    if len(tactic) < 7 or tactic[5] != '_':
        return None

    return {
        'box_front': tactic[0] == 'B',
        'front_area': tactic[1] == 'F',
        'sides': tactic[2] == 'S',
        'back': tactic[3] == 'K',
        'wall': tactic[4] == 'W',
        'turn': tactic[6] == 'T'
    }


def plot_tactic_sequence(tactics, output_file='tactic_pattern.png', title=None):
    """
    Visualize a sequence of tactics as 3x3 grids.

    Grid layout (agent facing up):
        [FL][F ][FR]     [7][0][1]
        [L ][A ][R ]  =  [6][X][2]
        [BL][B ][BR]     [5][4][3]
    """
    n_tactics = len(tactics)

    # Create figure
    fig, axes = plt.subplots(1, n_tactics, figsize=(2.5 * n_tactics, 4))
    if n_tactics == 1:
        axes = [axes]

    # Colors
    COLOR_EMPTY = np.array([1.0, 1.0, 1.0])           # White
    COLOR_BOX = np.array([0.9, 0.4, 0.1])             # Orange
    COLOR_BOX_AREA = np.array([0.9, 0.4, 0.1, 0.5])   # 50% transparent orange

    # Grid positions
    # 0=F (top middle), 1=FR (top right), 2=R (middle right), 3=BR (bottom right)
    # 4=B (bottom middle), 5=BL (bottom left), 6=L (middle left), 7=FL (top left)
    GRID_POS = {
        0: (0, 1),  # Front -> top middle
        1: (0, 2),  # Front-Right -> top right
        2: (1, 2),  # Right -> middle right
        3: (2, 2),  # Back-Right -> bottom right
        4: (2, 1),  # Back -> bottom middle
        5: (2, 0),  # Back-Left -> bottom left
        6: (1, 0),  # Left -> middle left
        7: (0, 0),  # Front-Left -> top left
    }

    # Position groups
    FRONT_POSITIONS = [0]           # F
    FRONT_AREA_POSITIONS = [7, 1]   # FL, FR
    SIDE_POSITIONS = [6, 2]         # L, R
    BACK_POSITIONS = [5, 4, 3]      # BL, B, BR

    for idx, tactic in enumerate(tactics):
        ax = axes[idx]
        flags = decode_tactic_flags(tactic)

        if flags is None:
            ax.text(0.5, 0.5, f"Invalid:\n{tactic}", ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        # Create 3x3 image (RGB)
        img = np.ones((3, 3, 3))  # Start with white

        # Fill cells based on tactic flags
        # Front cell - solid orange if box in front
        if flags['box_front']:
            row, col = GRID_POS[0]
            img[row, col] = COLOR_BOX

        # Front area (FL, FR) - 50% orange if front area boxes
        if flags['front_area']:
            for pos in FRONT_AREA_POSITIONS:
                row, col = GRID_POS[pos]
                # Blend with white for 50% transparency effect
                img[row, col] = 0.5 * COLOR_BOX + 0.5 * COLOR_EMPTY

        # Sides (L, R) - 50% orange if side boxes
        if flags['sides']:
            for pos in SIDE_POSITIONS:
                row, col = GRID_POS[pos]
                img[row, col] = 0.5 * COLOR_BOX + 0.5 * COLOR_EMPTY

        # Back (BL, B, BR) - 50% orange if back boxes
        if flags['back']:
            for pos in BACK_POSITIONS:
                row, col = GRID_POS[pos]
                img[row, col] = 0.5 * COLOR_BOX + 0.5 * COLOR_EMPTY

        # Center cell is agent (white)
        img[1, 1] = COLOR_EMPTY

        # Plot the grid
        ax.imshow(img, interpolation='nearest')

        # Grid lines - thicker if wall present
        line_width = 3 if flags['wall'] else 1
        line_color = 'darkred' if flags['wall'] else 'black'

        for i in range(4):
            ax.axhline(i - 0.5, color=line_color, linewidth=line_width)
            ax.axvline(i - 0.5, color=line_color, linewidth=line_width)

        # Arrow in center pointing up (agent facing direction)
        ax.annotate('', xy=(1, 0.7), xytext=(1, 1.3),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

        # Action text on top
        action_text = "TURN" if flags['turn'] else "FORWARD"
        action_color = 'blue' if flags['turn'] else 'green'

        # Title with step number, tactic code, numeric code, and action
        tactic_num = tactic_to_number(tactic)
        ax.set_title(f'Step {idx + 1}\n{tactic} (#{tactic_num})\n{action_text}',
                    fontsize=9, fontweight='bold', color=action_color)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_EMPTY, edgecolor='black', label='Empty / Agent'),
        mpatches.Patch(facecolor=COLOR_BOX, edgecolor='black', label='Box (front)'),
        mpatches.Patch(facecolor=tuple(0.5 * COLOR_BOX + 0.5 * COLOR_EMPTY), edgecolor='black', label='Boxes (area)'),
        mpatches.Patch(facecolor='white', edgecolor='darkred', linewidth=2, label='Wall nearby'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    # Overall title
    numeric_str = '-'.join(str(tactic_to_number(t)) for t in tactics)
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'Tactic Sequence: {numeric_str}\n(Arrow = agent facing up)',
                    fontsize=11, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved tactic pattern visualization to {output_file}")


def main():
    pattern_str = None
    output_file = None
    title = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ('--pattern', '--numeric', '-p') and i + 1 < len(sys.argv):
            pattern_str = sys.argv[i + 1]
            i += 2
        elif arg in ('--output', '-o') and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif arg in ('--title', '-t') and i + 1 < len(sys.argv):
            title = sys.argv[i + 1]
            i += 2
        elif arg == '--help':
            print(__doc__)
            sys.exit(0)
        else:
            # Assume it's the pattern if no flag
            pattern_str = arg
            i += 1

    if not pattern_str:
        print("Usage: python visualize_tactic_pattern.py --pattern B---W_F-B---W_F-B---W_T")
        print("       python visualize_tactic_pattern.py --pattern 34-34-35")
        print("       python visualize_tactic_pattern.py --help")
        sys.exit(1)

    # Parse the pattern
    tactics = parse_pattern(pattern_str)

    if not tactics:
        print(f"Error: Could not parse pattern '{pattern_str}'")
        sys.exit(1)

    print(f"Parsed {len(tactics)} tactics:")
    for i, t in enumerate(tactics):
        n = tactic_to_number(t)
        print(f"  {i+1}. {t} (#{n})")

    # Generate output filename if not specified
    if not output_file:
        numeric_str = '-'.join(str(tactic_to_number(t)) for t in tactics)
        output_file = f'tactic_pattern_{numeric_str}.png'

    # Plot
    plot_tactic_sequence(tactics, output_file, title)


if __name__ == '__main__':
    main()
