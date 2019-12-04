import sys, math
from random import randint, choice


def rotate(r, v):
    x = round(r[0] * math.cos(math.pi / v) - r[1] * math.sin(math.pi / v))
    y = round(r[0] * math.sin(math.pi / v) + r[1] * math.cos(math.pi / v))
    return [x, y]

def runboard(tartarus, cp, cd, cs, a, s):
    used_states = set()
    for step in range(0, 81):
        used_states.add(cs)
        cc = 0
        for m in range(0, 8):
            cx = cp[0] + cd[0]
            cy = cp[1] + cd[1]
            if (cx < 0 or cy < 0 or cx >= 6 or cy >= 6):
                cc = cc + pow(3, m) * 2  # wall = 2
            else:
                cc = cc + pow(3, m) * tartarus[cy][cx]
            cd = rotate(cd, 4)

        action = a[6561 * cs + cc]
        cs = s[6561 * cs + cc]

        if action == 0:
            cx = cp[0] + cd[0]
            cy = cp[1] + cd[1]
            if (cx >= 0 and cy >= 0 and cx < 6 and cy < 6):
                if tartarus[cy][cx] == 0:
                    cp = [cx, cy]
                else:  # there is a box
                    dx = cx + cd[0]
                    dy = cy + cd[1]
                    if (dx >= 0 and dy >= 0 and dx < 6 and dy < 6 and tartarus[dy][dx] == 0):
                        tartarus[cy][cx] = 0
                        tartarus[dy][dx] = 1
                        cp = [cx, cy]
        elif action == 1:
            cd = rotate(cd, 0.66)  # right
        elif action == 2:
            cd = rotate(cd, 2)  # left
    fitness = 0
    for x in range(0, 6):
        for y in range(0, 6):
            if tartarus[y][x] == 1:
                if y == 0 or y == 5:
                    fitness = fitness + 1
                if x == 0 or x == 5:
                    fitness = fitness + 1
    return fitness

# BEST-r-1280-10-451.txt
# "C:/Users/Kaya/source/repos/Tartarus00/x64/Release/BEST-r-1280-10-451.txt"
# C:/Users/Kaya/source/repos/Tartarus00/BEST-r-256-4-10.txt

with open("E:/google_drive/Research/kaya/tartarus/results/BEST-r-1280-10-451.txt") as f:
    d = f.read()

v = d.split()
aa = v[0::2]
ss = v[1::2]

a = [int(x) for x in aa]
s = [int(x) for x in ss]

print("agent loaded.")
print(len(s))

boards = list()

# randomly generate tartarus boards until a good score is received...

say = 0
toplam = 0
while say < 5000:
    tartarus = [[0 for x in range(0,6)] for x in range(0,6)]
    devam = True
    while devam:
        i = 0
        while i < 6:
            x = randint(1,4)
            y = randint(1,4)
            if tartarus[y][x] == 0:
                tartarus[y][x] = 1
                i = i + 1

        Y = list()
        for x in tartarus[1:5]:
            Y = Y + x[1:5]
        
        l=0
        for p in list(range(0, len(Y))):
            if Y[p] > 0:
                l = l + pow(2, p)
                
        if l not in boards:
            boards.append(l)
            devam = False

    tamam = True
    while tamam:
        cpx = randint(1, 4)
        cpy = randint(1, 4)
        if tartarus[cpy][cpx] == 0:
            cp = [cpx, cpy]
            tamam = False

    cd = choice([[-1, 0], [1, 0], [0, 1], [0, -1]])
    cs = s[-1]

    sonuc = runboard(tartarus, cp, cd, cs, a, s)
    
    if say % 100 == 0:
        print(say)
        if say > 0:
            print(toplam/say)
    toplam = toplam + sonuc
    say = say + 1
    
print(toplam/say)



