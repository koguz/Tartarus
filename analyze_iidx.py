#!/usr/bin/env python3
"""Analyze the iidx arrays and visualize differences"""

# iidx from kernel_2026.cu
iidx_2026 = [0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546]

# iidx from kernel_test_all.cu
iidx_test = [0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,111,112,117,118,120,121,159,160,242,243,245,246,251,252,254,255,270,271,273,274,279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546]

def to_base3(n, digits=8):
    """Convert decimal to base-3 representation with fixed digits"""
    result = []
    for _ in range(digits):
        result.append(n % 3)
        n //= 3
    return result  # LSB first (position 0 = front)

def visualize_3x3(val, scan_order="ccw45"):
    """
    Visualize the 8-neighborhood as a 3x3 grid.
    Center is agent (A), positions are numbered based on scan order.

    CCW 45-degree scan starting from Front:
    Position 0: Front
    Position 1: Front-Left (diagonal)
    Position 2: Left
    Position 3: Back-Left (diagonal)
    Position 4: Back
    Position 5: Back-Right (diagonal)
    Position 6: Right
    Position 7: Front-Right (diagonal)

    Assuming agent faces UP (North):
        7   0   1
        6   A   2
        5   4   3
    """
    digits = to_base3(val, 8)
    symbols = {0: '.', 1: 'B', 2: 'W'}  # empty, Box, Wall

    # Map position index to 3x3 grid location (row, col)
    # Agent faces UP (North)
    pos_to_grid = {
        0: (0, 1),  # Front (North)
        1: (0, 2),  # Front-Left becomes Front-Right in our view? Let me think...
        2: (1, 2),  # Left becomes Right
        3: (2, 2),  # Back-Left becomes Back-Right
        4: (2, 1),  # Back (South)
        5: (2, 0),  # Back-Right becomes Back-Left
        6: (1, 0),  # Right becomes Left
        7: (0, 0),  # Front-Right becomes Front-Left
    }

    # Actually, let me reconsider. The rotate_ccw rotates the direction vector CCW.
    # If agent faces North, after CCW 45deg rotation, it faces Northwest direction vector.
    # So the scan checks: Front, then rotates direction CCW by 45, checks that cell, etc.
    #
    # Let's say agent faces North (up the page):
    # - Position 0: looks North (front) -> cell at top-center
    # - Position 1: looks Northwest (45 CCW from N) -> cell at top-left
    # - Position 2: looks West (90 CCW from N) -> cell at left-center
    # - Position 3: looks Southwest (135 CCW from N) -> cell at bottom-left
    # - Position 4: looks South (180 CCW from N) -> cell at bottom-center
    # - Position 5: looks Southeast (225 CCW from N) -> cell at bottom-right
    # - Position 6: looks East (270 CCW from N) -> cell at right-center
    # - Position 7: looks Northeast (315 CCW from N) -> cell at top-right

    pos_to_grid_ccw45 = {
        0: (0, 1),  # Front (N) - top center
        1: (0, 0),  # NW - top left
        2: (1, 0),  # W - middle left
        3: (2, 0),  # SW - bottom left
        4: (2, 1),  # S (back) - bottom center
        5: (2, 2),  # SE - bottom right
        6: (1, 2),  # E - middle right
        7: (0, 2),  # NE - top right
    }

    grid = [['?', '?', '?'], ['?', 'A', '?'], ['?', '?', '?']]

    for pos, (r, c) in pos_to_grid_ccw45.items():
        grid[r][c] = symbols[digits[pos]]

    return grid

def print_grid(grid, title=""):
    if title:
        print(title)
    print("    Agent faces UP (^)")
    print("    +---+---+---+")
    for row in grid:
        print("    | " + " | ".join(row) + " |")
        print("    +---+---+---+")

print(f"Length of iidx_2026: {len(iidx_2026)}")
print(f"Length of iidx_test: {len(iidx_test)}")
print()

# Find all differences
print("=" * 60)
print("DIFFERENCES BETWEEN ARRAYS:")
print("=" * 60)
print(f"{'Index':>5} | {'kernel_2026':>11} | {'kernel_test':>11}")
print("-" * 40)

differences = []
for i in range(len(iidx_2026)):
    if iidx_2026[i] != iidx_test[i]:
        differences.append((i, iidx_2026[i], iidx_test[i]))
        print(f"{i:5d} | {iidx_2026[i]:11d} | {iidx_test[i]:11d}")

print()
print("=" * 60)
print("VISUALIZATION OF DIFFERENCES")
print("=" * 60)
print()
print("Legend: . = empty, B = box, W = wall, A = agent")
print()

for idx, val_2026, val_test in differences:
    print(f"Index {idx}:")
    print()

    print(f"  kernel_2026.cu value: {val_2026} = {to_base3(val_2026, 8)} (base 3)")
    grid = visualize_3x3(val_2026)
    print_grid(grid, f"  kernel_2026.cu ({val_2026}):")
    print()

    print(f"  kernel_test_all.cu value: {val_test} = {to_base3(val_test, 8)} (base 3)")
    grid = visualize_3x3(val_test)
    print_grid(grid, f"  kernel_test_all.cu ({val_test}):")
    print()
    print("-" * 40)
    print()
