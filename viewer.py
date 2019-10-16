import sys, math
from random import randint, choice
from copy import deepcopy
from PIL import Image


def rotate(r, v):
    x = round(r[0] * math.cos(math.pi / v) - r[1] * math.sin(math.pi / v))
    y = round(r[0] * math.sin(math.pi / v) + r[1] * math.cos(math.pi / v))
    return [x, y]

def runboard(tartarus, cp, cd, cs, a, s, im, ima, imb, ime, saveImages: bool):
    for step in range(0, 81):
        if saveImages:
            for x in range(0, 6):
                for y in range(0, 6):
                    if tartarus[x][y] == 0:
                        im.paste(ime, [y * 100, x * 100])
                    else:
                        im.paste(imb, [y * 100, x * 100])
            im.paste(ima, [cp[0] * 100, cp[1] * 100])

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
            ima = ima.transpose(Image.ROTATE_90)
        elif action == 2:
            cd = rotate(cd, 2)  # left
            ima = ima.transpose(Image.ROTATE_270)

        if saveImages:
            fname = "state-%02d.png" % step
            im.save(fname)

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

with open("C:/Users/Kaya/source/repos/Tartarus00/x64/Release/BEST-r-1280-10-451.txt") as f:
    d = f.read()

v = d.split()
aa = v[0::2]
ss = v[1::2]

a = [int(x) for x in aa]
s = [int(x) for x in ss]

print("agent loaded.")
print(len(s))

imb = Image.open("box.png").convert("RGB")
iman = Image.open("agentn.png").convert("RGB")
imae = Image.open("agente.png").convert("RGB")
imaw = Image.open("agentw.png").convert("RGB")
imas = Image.open("agents.png").convert("RGB")
ime = Image.open("empty.png").convert("RGB")

im = Image.new("RGB", (600, 600), (255, 255, 255))

# tartarus = [
#     [0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 1, 0],
#     [0, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0]
# ]

# tartarus = [
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 1, 0],
#     [0, 1, 0, 0, 0, 0],
#     [0, 1, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0]
# ]
#
# cp = [1, 4]
# cd = [-1, 0]
# cs = s[-1]
# ima = imaw
# runboard(tartarus, cp, cd, cs, a, s, im, ima, imb, ime, True)
# exit(0)

# randomly generate tartarus boards until a good score is received...

devam = True
while devam:
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
    if cd == [-1, 0]:
        ima = imaw
    elif cd == [1, 0]:
        ima = imae
    elif cd == [0, 1]:
        ima = imas
    elif cd == [0, -1]:
        ima = iman
    cs = s[-1]

    cp2 = deepcopy(cp)
    cd2 = deepcopy(cd)
    cs2 = s[-1]

    yedek = deepcopy(tartarus)

    sonuc = runboard(tartarus, cp, cd, cs, a, s, im, ima, imb, ime, False)
    print(sonuc)
    if sonuc > 7:
        print("found!")
        runboard(yedek, cp2, cd2, cs2, a, s, im, ima, imb, ime, True)
        devam = False
        # runboard(yedek, cp2, cd2, cs2, a, s, True )


# im.paste(ima.transpose(Image.ROTATE_90), [100, 100])
# im.paste(imb, [300, 100])





