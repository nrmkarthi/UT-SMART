import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle, FancyBboxPatch, Circle, Arc, Wedge
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(7.16, 3.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 54); ax.axis("off")
INK = "#2b2a27"

# ---------------- background: sky, sea gradient, seabed
ax.add_patch(Rectangle((0, 42), 100, 12, fc="#eef5fb", ec="none", zorder=0))
sea = LinearSegmentedColormap.from_list("sea", ["#0f3d66", "#2a78d6", "#9ec5f4"])
grad = np.linspace(0, 1, 256).reshape(-1, 1)
ax.imshow(grad, extent=(0, 100, 0, 42), origin="lower", cmap=sea, aspect="auto", zorder=0, alpha=0.95)
xs = np.linspace(0, 100, 400)
wave = 42 + 0.35 * np.sin(xs * 0.9)
ax.fill_between(xs, wave, 42.6, color="#9ec5f4", zorder=1, lw=0)
ax.plot(xs, wave, color="#5598e7", lw=0.8, zorder=1)
bed = 5 + 1.6 * np.sin(xs * 0.13) + 0.8 * np.sin(xs * 0.41 + 1)
ax.fill_between(xs, 0, bed, color="#5a4a35", zorder=1, lw=0)
ax.fill_between(xs, bed - 1.2, bed, color="#8a7350", zorder=1, lw=0)

# ---------------- space / backhaul
def satellite(x, y):
    ax.add_patch(Rectangle((x - 1.2, y - 0.9), 2.4, 1.8, fc="#c3c2b7", ec=INK, lw=0.6, zorder=5))
    for dx in (-4.6, 1.6):
        ax.add_patch(Rectangle((x + dx, y - 0.6), 3.0, 1.2, fc="#256abf", ec=INK, lw=0.5, zorder=5))
    ax.text(x - 5.2, y - 0.3, "Satellite", ha="right", va="center", fontsize=6.3, color=INK, zorder=6)
satellite(40, 51)

def bolt(x1, y1, x2, y2, col="#eb6834"):
    n_seg = max(6, int(np.hypot(x2 - x1, y2 - y1) / 1.6)) | 1
    t = np.linspace(0, 1, n_seg)
    px = x1 + (x2 - x1) * t; py = y1 + (y2 - y1) * t
    nx, ny = -(y2 - y1), (x2 - x1); n = np.hypot(nx, ny); nx, ny = nx / n, ny / n
    off = np.array([0] + [(-1) ** i for i in range(n_seg - 2)] + [0]) * 0.45
    ax.plot(px + nx * off, py + ny * off, color=col, lw=1.3, zorder=4)

bolt(38.5, 50.0, 25.9, 46.4)
bolt(44.6, 51.2, 84.0, 51.2)
ax.text(31, 46.0, "RF / electromagnetic\nbackhaul", fontsize=5.6, color="#9a3d13", zorder=6, linespacing=0.95)

# control centre on island (right)
isl = Ellipse((90, 43.2), 17, 3.2, fc="#6f9e4f", ec="#4e7a35", lw=0.6, zorder=3)
ax.add_patch(isl)
for i, (bx, bh) in enumerate([(85.5, 2.8), (87.5, 3.6), (89.6, 2.4), (91.4, 4.0), (93.4, 3.0)]):
    ax.add_patch(Rectangle((bx, 43.6), 1.6, bh, fc="#c3c2b7", ec=INK, lw=0.4, zorder=4))
ax.text(91.5, 49.6, "Mission control centre", ha="center", fontsize=6.3, color=INK, zorder=6)
# cellular tower
ax.plot([79, 79], [43, 48.5], color=INK, lw=0.9, zorder=4)
ax.plot([77.8, 79, 80.2], [43, 48.5, 43], color=INK, lw=0.6, zorder=4)
for r in (0.9, 1.6):
    ax.add_patch(Arc((79, 48.6), r * 2, r * 2, theta1=30, theta2=150, color=INK, lw=0.6, zorder=4))
ax.text(77.6, 46.0, "4G/5G", ha="right", fontsize=5.8, color=INK, zorder=6)

# ---------------- surface: buoys, gateway, ship
def buoy(x, label=None, gw=False):
    y = 42.4
    ax.add_patch(Ellipse((x, y), 3.0, 1.2, fc="#eda100", ec=INK, lw=0.6, zorder=6))
    ax.add_patch(Polygon([[x - 0.9, y + 0.3], [x + 0.9, y + 0.3], [x + 0.4, y + 2.4], [x - 0.4, y + 2.4]],
                         fc="#f6c75a", ec=INK, lw=0.6, zorder=6))
    ax.plot([x, x], [y + 2.4, y + 3.8], color=INK, lw=0.7, zorder=6)
    ax.plot([x, x], [y - 0.5, y - 1.7], color=INK, lw=0.7, zorder=6)
    if gw:
        ax.add_patch(Rectangle((x - 0.8, y + 0.7), 1.6, 1.0, fc="#2a78d6", ec=INK, lw=0.4, zorder=7))
    if label:
        ax.text(x, y + 4.3, label, ha="center", fontsize=6.0, color=INK, zorder=7)

buoy(25.5, None, gw=True)
ax.text(23.6, 44.6, "Surface gateway\n+ edge server", ha="right", fontsize=6.0, color=INK, zorder=7, linespacing=0.95)
buoy(11)
buoy(54, "Relay buoy")
ax.plot([11, 25.5, 54], [42.4, 42.4, 42.4], ls=(0, (2, 2)), color="#eda100", lw=0.8, zorder=5)

# support ship
hull = Polygon([[61, 42.2], [74, 42.2], [75.5, 44.0], [59.5, 44.0]], fc="#55544f", ec=INK, lw=0.6, zorder=6)
ax.add_patch(hull)
ax.add_patch(Rectangle((64.5, 44.0), 5, 1.8, fc="#e6e5e0", ec=INK, lw=0.5, zorder=6))
ax.add_patch(Rectangle((66, 45.8), 2, 1.0, fc="#e6e5e0", ec=INK, lw=0.5, zorder=6))
ax.text(67.3, 47.6, "Support vessel", ha="center", fontsize=6.0, color=INK, zorder=7)

# ---------------- AUVs
def auv(x, y, s=1.0, bad=False, flip=False):
    col = "#e5484d" if bad else "#f2f1ec"
    d = -1 if flip else 1
    ax.add_patch(Ellipse((x, y), 6.0 * s, 1.7 * s, fc=col, ec=INK, lw=0.7, zorder=8))
    ax.add_patch(Ellipse((x + d * 2.4 * s, y), 1.3 * s, 1.5 * s, fc="#eda100", ec=INK, lw=0.5, zorder=9))
    ax.add_patch(Polygon([[x - d * 3.0 * s, y], [x - d * 4.1 * s, y + 1.1 * s], [x - d * 4.1 * s, y - 1.1 * s]],
                         fc="#55544f", ec=INK, lw=0.5, zorder=8))
    ax.add_patch(Polygon([[x - d * 0.6 * s, y + 0.8 * s], [x + d * 0.4 * s, y + 0.8 * s], [x - d * 0.2 * s, y + 1.7 * s]],
                         fc="#55544f", ec=INK, lw=0.4, zorder=8))

A = {1: (14, 29), 2: (36, 34), 3: (50, 23), 4: (64, 31)}
# patrol trajectories
for (cx, cy, w, h) in [(14, 24, 20, 10), (37, 28, 18, 11), (60, 26, 20, 11)]:
    ax.add_patch(Ellipse((cx, cy), w, h, fc="none", ec="#111111", ls=(0, (4, 3)), lw=0.8, zorder=7))
ax.text(14, 17.0, "Patrol trajectory", fontsize=5.8, color="#ffffff", ha="center", zorder=9)

auv(*A[1]); auv(*A[2], flip=True); auv(*A[3]); auv(*A[4], bad=True, flip=True)
labels = {1: "AUV 1 (cluster edge node)", 2: "AUV 2", 3: "AUV 3", 4: "Compromised AUV"}
offs = {1: (0, 2.0), 2: (0, 2.0), 3: (0, -3.0), 4: (0, 2.0)}
for k, (x, y) in A.items():
    ax.text(x + offs[k][0], y + offs[k][1], labels[k], ha="center", fontsize=6.0,
            color="#ffdede" if k == 4 else "#ffffff", weight="bold", zorder=10)

# acoustic links (to gateway and between AUVs)
def acoustic(p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], ls=(0, (3, 2)), color="#f6c75a", lw=1.0, zorder=7)
acoustic((25.5, 40.7), A[1]); acoustic((25.5, 40.7), A[2]); acoustic((54, 40.7), A[3])
acoustic((54, 40.7), A[4]); acoustic(A[1], A[2]); acoustic(A[2], A[3])
ax.text(17.2, 37.0, "Acoustic\n(kbit/s)", fontsize=5.8, color="#ffe9a8", ha="center", zorder=9)

# ---------------- seabed: sensors, wreck/pipeline, rocks
def sensor(x):
    yb = 5 + 1.6 * np.sin(x * 0.13) + 0.8 * np.sin(x * 0.41 + 1)
    ax.plot([x - 0.8, x, x + 0.8], [yb, yb + 2.0, yb], color="#c3c2b7", lw=0.8, zorder=6)
    ax.add_patch(Circle((x, yb + 2.3), 0.55, fc="#1baf7a", ec=INK, lw=0.5, zorder=7))
    for r in (1.2, 2.0):
        ax.add_patch(Arc((x, yb + 2.3), r * 2, r * 2, theta1=40, theta2=140, color="#bdf2d9", lw=0.6, zorder=6))
    return yb + 2.3

sens = [sensor(x) for x in (6, 11, 18, 24, 31, 40, 46, 57, 63, 70)]
ax.text(20, 1.2, "Seabed sensors with semantic encoders (1× / 4× / 16× / 64×)",
        fontsize=6.2, color="#ffffff", zorder=9)

# sensor -> AUV acoustic uplinks (thin)
for sx, sy, a in [(11, sens[1], 1), (18, sens[2], 1), (40, sens[5], 2), (46, sens[6], 3), (63, sens[8], 4)]:
    ax.plot([sx, A[a][0]], [sy, A[a][1] - 0.8], ls=(0, (1, 1.5)), color="#f6c75a", lw=0.7, zorder=6)
# optical short-range link AUV3 -> sensor
ax.fill([50, 55.6, 58.4], [22.2, sens[7] + 0.4, sens[7] + 0.4], color="#7cf29a", alpha=0.45, zorder=6, lw=0)
ax.text(49.5, 16.5, "Optical\n(<60 m, Mb/s)", ha="right", fontsize=5.8, color="#d7ffe2", zorder=9)

# wreck / pipeline inspection target
ax.add_patch(Polygon([[30, 8.1], [36, 8.4], [38.5, 10.6], [31, 10.8]], fc="#3a3935", ec="#111", lw=0.6, zorder=5))
ax.add_patch(Rectangle((33, 10.6), 2.2, 1.4, fc="#3a3935", ec="#111", lw=0.5, zorder=5))
ax.add_patch(Rectangle((76, 6.0), 20, 1.1, fc="#8a8984", ec="#111", lw=0.5, zorder=5))
ax.text(34.3, 13.0, "Wreck (inspection)", fontsize=5.8, color="#ffffff", ha="center", zorder=9)
ax.text(86, 8.0, "Pipeline", fontsize=5.8, color="#ffffff", ha="center", zorder=9)

# ocean current arrows
for (x, y) in [(64, 17.5), (67, 14.5)]:
    ax.annotate("", (x + 5, y - 1.5), (x, y), arrowprops=dict(arrowstyle="-|>", color="#cde2fb", lw=1.0), zorder=7)
ax.text(62.5, 18.8, "Ocean current", fontsize=5.8, color="#e7f1fb", zorder=9)

# ---------------- inset: AUV edge node
bx, by, bw, bh = 76.5, 10.5, 23, 27
ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.3,rounding_size=1.2",
                            fc="#fcfcfb", ec=INK, lw=0.8, zorder=11))
ax.text(bx + bw / 2, by + bh - 2.0, "AUV edge-computing node", ha="center", fontsize=6.6, weight="bold", zorder=12)
# processing chip
cx, cy = bx + bw / 2, by + 11.5
ax.add_patch(FancyBboxPatch((cx - 4.2, cy - 3.0), 8.4, 6.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                            fc="#184f95", ec=INK, lw=0.6, zorder=12))
for i in range(5):
    for side in (-1, 1):
        ax.plot([cx - 3.4 + i * 1.7] * 2, [cy + side * 3.0, cy + side * 3.7], color="#c3c2b7", lw=0.6, zorder=12)
ax.text(cx, cy + 0.8, "Edge processor", ha="center", fontsize=5.6, color="white", zorder=13)
ax.text(cx, cy - 1.4, "inference · twin\nMAPPO · trust", ha="center", fontsize=5.0, color="#cde2fb",
        zorder=13, linespacing=1.0)
mods = [("Acoustic modem", bx + 1.0, by + 20.5), ("Optical modem", bx + 12.4, by + 20.5),
        ("Imaging sonar", bx + 1.0, by + 4.2), ("Stereo camera", bx + 12.4, by + 4.2),
        ("INS / DVL nav.", bx + 1.0, by + 0.6), ("Vehicle control", bx + 12.4, by + 0.6)]
for name, x, y in mods:
    ax.add_patch(FancyBboxPatch((x, y), 9.6, 2.4, boxstyle="round,pad=0.15,rounding_size=0.4",
                                fc="#e7f1fb", ec="#55544f", lw=0.5, zorder=12))
    ax.text(x + 4.8, y + 1.2, name, ha="center", va="center", fontsize=5.3, zorder=13)
    ty = y + 2.4 if y < cy else y
    tx = x + 4.8
    ax.plot([tx, cx + (tx - cx) * 0.35], [ty, cy + (3.0 if y > cy else -3.0)], color="#55544f", lw=0.6, zorder=11.5)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig("fig_scenario.png", dpi=300)
print("ok")
