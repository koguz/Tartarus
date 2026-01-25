import sys, math
import os
import time
from random import randint, choice
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CONFIGURATION INPUT
# ==========================================
# To replay a specific board, paste the output dictionary here.
# Example: OVERRIDE_CONFIG = {'board': [0, 0, ...], 'pos': [2, 3], 'dir': [0, 1]}
OVERRIDE_CONFIG = None
# ==========================================

# --- Inverted Index Mapping ---
IIDX = [
    0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40,78,79,81,82,84,85,90,91,93,94,108,109,
    111,112,117,118,120,121,159,160,243,244,246,247,252,253,255,256,270,271,273,274,
    279,280,282,283,321,322,324,325,327,328,333,334,336,337,351,352,354,355,360,361,
    363,364,402,403,702,703,705,706,711,712,714,715,726,727,729,730,732,733,738,739,
    741,742,756,757,759,760,765,766,768,769,807,808,810,811,813,814,819,820,822,823,
    837,838,840,841,846,847,849,850,888,972,973,975,976,981,982,984,985,999,1000,1002,
    1003,1008,1009,1011,1012,1050,1051,1053,1054,1056,1057,1062,1063,1065,1066,1080,
    1081,1083,1084,1089,1090,1092,1131,1431,1432,1434,1435,1440,1443,1455,2187,2188,
    2190,2191,2196,2197,2199,2200,2214,2215,2217,2218,2223,2224,2226,2227,2265,2266,
    2268,2269,2271,2272,2277,2278,2280,2281,2295,2296,2298,2299,2304,2305,2307,2308,
    2346,2347,2430,2431,2433,2434,2439,2440,2442,2443,2457,2458,2460,2461,2466,2467,
    2469,2470,2508,2509,2511,2512,2514,2515,2520,2521,2523,2524,2538,2539,2541,2542,
    2547,2548,2550,2589,2590,2889,2890,2892,2893,2898,2899,2901,2902,2913,2914,2916,
    2917,2919,2920,2925,2926,2928,2929,2943,2944,2946,2947,2952,2953,2955,2956,2994,
    2995,2997,2998,3000,3001,3006,3007,3009,3010,3024,3025,3027,3028,3033,3034,3036,
    3075,3159,3160,3162,3163,3168,3169,3171,3172,3186,3187,3189,3190,3195,3196,3198,
    3237,3238,3240,3241,3243,3244,3249,3250,3252,3267,3268,3270,3276,3318,3618,3619,
    3621,3622,3627,3630,3642,4382,4391,4409,4418,4454,4463,4472,4490,4499,4535,4625,
    4634,4652,4661,4697,4706,4715,4733,4742,4778,5111,5120,5138,5147,5183,5192,5219,
    5354,5363,5381,5390,5426,5435,5462,6318,6319,6321,6322,6326,6327,6328,6330,6331,
    6335,6345,6346,6348,6349,6353,6354,6355,6357,6358,6362,6399,6400,6402,6403,6407,
    6408,6411,6426,6427,6429,6430,6434,6435,6438,6534,6535,6537,6538,6543,6546
]

CC_TO_IDX = {val: i for i, val in enumerate(IIDX)}
INPUT_SIZE = len(IIDX)

def ternary(n):
    if n == 0: return '00000000'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums)).zfill(8)

def rotate(r, v):
    x = round(r[0] * math.cos(math.pi / v) - r[1] * math.sin(math.pi / v))
    y = round(r[0] * math.sin(math.pi / v) + r[1] * math.cos(math.pi / v))
    return [x, y]

def runboard(tartarus, cp, cd, cs, a, s, im, ima, imb, ime, saveImages: bool, output_dir=""):
    used_states = set()
    W, H = 600, 600
    TEXT_H = 50
    
    if saveImages and output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    for step in range(0, 81):
        used_states.add(cs)
        
        if saveImages:
            for x in range(0, 6):
                for y in range(0, 6):
                    if tartarus[x][y] == 0:
                        im.paste(ime, [y * 100, x * 100])
                    else:
                        im.paste(imb, [y * 100, x * 100])
            im.paste(ima, [cp[0] * 100, cp[1] * 100])

        cc = 0
        cd_temp = deepcopy(cd) 
        for m in range(0, 8):
            cx = cp[0] + cd_temp[0]
            cy = cp[1] + cd_temp[1]
            if (cx < 0 or cy < 0 or cx >= 6 or cy >= 6):
                cc = cc + pow(3, m) * 2
            else:
                cc = cc + pow(3, m) * tartarus[cy][cx]
            cd_temp = rotate(cd_temp, 4)

        try:
            mapped_idx = CC_TO_IDX[cc]
        except KeyError:
            print(f"CRITICAL ERROR: Encounted a board configuration (cc={cc}) that is not in IIDX.")
            mapped_idx = 0 
        
        flat_idx = INPUT_SIZE * cs + mapped_idx
        
        try:
            action = a[flat_idx]
            next_state = s[flat_idx]
        except IndexError:
             print(f"INDEX ERROR: flat_idx {flat_idx} out of range.")
             break

        if saveImages and output_dir:
            canvas = Image.new("RGB", (W, H + TEXT_H), (30, 30, 30))
            canvas.paste(im, (0, 0))
            draw = ImageDraw.Draw(canvas)
            
            action_str = ["MOVE", "LEFT", "RIGHT"][action]
            cc_ternary = ternary(cc)
            
            info_text = (f"Step: {step:02d} | State: {cs} -> {next_state}\n"
                         f"Input: {cc} ({cc_ternary}) | Action: {action_str}")
            
            draw.text((20, H + 10), info_text, fill=(255, 255, 255), font=font)
            fname = "state-%02d.png" % step
            canvas.save(os.path.join(output_dir, fname))

        cs = next_state

        if action == 0:
            cx = cp[0] + cd[0]
            cy = cp[1] + cd[1]
            if (cx >= 0 and cy >= 0 and cx < 6 and cy < 6):
                if tartarus[cy][cx] == 0:
                    cp = [cx, cy]
                else:
                    dx = cx + cd[0]
                    dy = cy + cd[1]
                    if (dx >= 0 and dy >= 0 and dx < 6 and dy < 6 and tartarus[dy][dx] == 0):
                        tartarus[cy][cx] = 0
                        tartarus[dy][dx] = 1
                        cp = [cx, cy]
        elif action == 1:
            cd = rotate(cd, 0.66)
            ima = ima.transpose(Image.ROTATE_90)
        elif action == 2:
            cd = rotate(cd, 2)
            ima = ima.transpose(Image.ROTATE_270)

    print(f"Unique states used: {len(used_states)}")
    fitness = 0
    for x in range(0, 6):
        for y in range(0, 6):
            if tartarus[y][x] == 1:
                if y == 0 or y == 5:
                    fitness = fitness + 1
                if x == 0 or x == 5:
                    fitness = fitness + 1
    return fitness

# --- Main Execution ---

agent_path = "best/b-D2-4096-128-3000-1.txt"
print(f"Loading agent from: {agent_path}")

try:
    with open(agent_path) as f:
        d = f.read()
except FileNotFoundError:
    print(f"Error: Could not find agent file at {agent_path}")
    sys.exit(1)

v = d.split()
aa = v[0::2]
ss = v[1::2]

a = [int(x) for x in aa]
s = [int(x) for x in ss]

print("Agent loaded.")
print(f"State table size: {len(s)}")

try:
    imb = Image.open("images/box.png").convert("RGB")
    iman = Image.open("images/agentn.png").convert("RGB")
    imae = Image.open("images/agente.png").convert("RGB")
    imaw = Image.open("images/agentw.png").convert("RGB")
    imas = Image.open("images/agents.png").convert("RGB")
    ime = Image.open("images/empty.png").convert("RGB")
except FileNotFoundError as e:
    print(f"Error loading images: {e}")
    sys.exit(1)

im = Image.new("RGB", (600, 600), (255, 255, 255))

tartarus = []
cp = []
cd = []

if OVERRIDE_CONFIG:
    print(">>> USING OVERRIDE CONFIGURATION <<<")
    # Reconstruct 2D board from 1D array
    flat_board = OVERRIDE_CONFIG['board']
    tartarus = [flat_board[i*6:(i+1)*6] for i in range(6)]
    cp = OVERRIDE_CONFIG['pos']
    cd = OVERRIDE_CONFIG['dir']
    
else:
    print("Generating random board...")
    tartarus = [[0 for x in range(0,6)] for x in range(0,6)]
    i = 0
    while i < 6:
        x = randint(1,4)
        y = randint(1,4)
        if tartarus[y][x] == 0:
            tartarus[y][x] = 1
            i = i + 1

    tamam = True
    while tamam:
        cpx = randint(1, 4)
        cpy = randint(1, 4)
        if tartarus[cpy][cpx] == 0:
            cp = [cpx, cpy]
            tamam = False

    cd = choice([[-1, 0], [1, 0], [0, 1], [0, -1]])
    
    # Print the replay configuration
    flat_board = [cell for row in tartarus for cell in row]
    config_dict = {'board': flat_board, 'pos': cp, 'dir': cd}
    print("-" * 60)
    print("TO REPLAY THIS BOARD, COPY THE LINE BELOW INTO 'OVERRIDE_CONFIG':")
    print(config_dict)
    print("-" * 60)

# Set image based on direction (Logic moved here so it works for both override and random)
if cd == [-1, 0]:
    ima = imaw
elif cd == [1, 0]:
    ima = imae
elif cd == [0, 1]:
    ima = imas
elif cd == [0, -1]:
    ima = iman

cs = s[-1]

timestamp = int(time.time())
run_dir = f"runs/run_{timestamp}"
print(f"Running simulation... Saving images to: {run_dir}")

score = runboard(tartarus, cp, cd, cs, a, s, im, ima, imb, ime, True, run_dir)

print(f"Simulation complete. Final Score: {score}")