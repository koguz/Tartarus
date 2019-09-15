import sys, math
from PIL import Image


def rotate(r, v):
    x = round(r[0] * math.cos(math.pi / v) - r[1] * math.sin(math.pi / v))
    y = round(r[0] * math.sin(math.pi / v) + r[1] * math.cos(math.pi / v))
    return [x, y]


# BEST-r-1280-10-451.txt

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
ima = Image.open("agent.png").convert("RGB")
ime = Image.open("empty.png").convert("RGB")

im = Image.new("RGB", (600, 600), (255, 255, 255))

tartarus = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0]
]

cp = [2, 1]
cd = [-1, 0]
cs = s[-1]

for step in range(0, 81):
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

    fname = "state-%02d.png" % step
    im.save(fname)

# im.paste(ima.transpose(Image.ROTATE_90), [100, 100])
# im.paste(imb, [300, 100])





