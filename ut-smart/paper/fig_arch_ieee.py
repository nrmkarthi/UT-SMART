import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})
fig, ax = plt.subplots(figsize=(7.16, 3.75))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
K = "black"
LW = 0.7


def box(x, y, w, h, txt, fill="white", fs=7.2, bold=False, ls="-", lw=LW, r=0.0):
    if r:
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=K, lw=lw, ls=ls)
    else:
        p = Rectangle((x, y), w, h, fc=fill, ec=K, lw=lw, ls=ls)
    ax.add_patch(p)
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                weight="bold" if bold else "normal", linespacing=1.15)
    return (x, y, w, h)


def group(x, y, w, h, title, fill="#f4f4f4"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=K, lw=0.8, ls=(0, (4, 2))))
    ax.text(x + 0.8, y + h - 0.6, title, ha="left", va="top", fontsize=7.6, weight="bold")


def arr(p, q, txt="", dx=0, dy=0.9, ls="-", fs=6.6, ha="center"):
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                          color=K, lw=LW, ls=ls, shrinkA=0, shrinkB=0))
    if txt:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, txt, ha=ha, va="center", fontsize=fs)


def poly(pts, txt="", tpos=None, ls="-", fs=6.6, ha="center"):
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=K, lw=LW, ls=ls)
    ax.annotate("", pts[-1], pts[-2], arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                                       color=K, lw=LW, ls=ls, shrinkA=0, shrinkB=0))
    if txt:
        ax.text(*tpos, txt, ha=ha, va="center", fontsize=fs)


def step(x, y, n):
    ax.add_patch(Circle((x, y), 1.05, fc="black", ec=K, lw=0.5, zorder=5))
    ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=6.4, color="white", weight="bold", zorder=6)


# ---------------------------------------------------------------- sensor
group(0.5, 13.5, 16.5, 36.5, "Seabed sensor $j$")
box(2, 39.5, 13.5, 5.5, "Raw sonar / image\n$S$ bits")
box(2, 31, 13.5, 6, "Feature extractor\n(lightweight CNN)")
box(2, 22.5, 13.5, 6, "Semantic encoder\n$\\kappa_k\\in\\{1,4,16,64\\}$")
arr((8.75, 39.5), (8.75, 37.0))
arr((8.75, 31.0), (8.75, 28.5))

# ---------------------------------------------------------------- AUV agent
group(19, 13.5, 53, 36.5, "AUV agent $c$ (decentralized execution)")
ax.add_patch(Rectangle((20.5, 31), 30, 14.6, fc="white", ec=K, lw=0.6))
ax.text(21.2, 45.0, "Uncertainty-aware digital twin", ha="left", va="top", fontsize=7.0, style="italic")
box(21.8, 37.0, 27.6, 5.4, "Kalman predict/update of $\\mathcal{N}(m_l,P_l)$\non-demand probe if $\\sqrt{P_j}>\\tau_s$", fill="#e9e9e9")
box(21.8, 32.0, 27.6, 4.0, "What-if evaluator: $\\hat D(a),\\ \\hat D_{\\mathrm{LCB}}(a),\\ A(a),\\ \\hat E(a)$", fill="#e9e9e9")
arr((35.6, 37.0), (35.6, 36.0))

box(53.5, 31, 17, 14.6, "", fill="white")
ax.text(54.2, 45.0, "Shared actor", ha="left", va="top", fontsize=7.0, style="italic")
box(55, 37.0, 14, 5.4, "$f_\\theta(g,\\phi_a)$\nMLP 27-64-64-1", fill="#e9e9e9")
box(55, 32.0, 14, 3.8, "$a^\\star=\\arg\\max_a \\pi_\\theta(a\\,|\\,o)$", fill="#e9e9e9")
arr((62, 37.0), (62, 35.8))
arr((50.5, 39.7), (55, 39.7))
ax.text(52.0, 41.0, "$\\phi_a$", ha="center", va="center", fontsize=6.6)
ax.plot([49.4, 50.5, 50.5], [34.0, 34.0, 39.7], color=K, lw=LW)

box(21.8, 15.5, 15.5, 7.5, "Cluster acoustic channel\n(FIFO, $R_s(t)$) /\noptical link", fill="white")
box(39.5, 15.5, 14.5, 7.5, "Edge inference\n$C_{k,e}/F_{\\mathrm{AUV}}$", fill="white")
box(56.2, 15.5, 14.2, 7.5, "Outbound acoustic\nqueue $\\to e^\\star$", fill="white")
arr((37.3, 19.25), (39.5, 19.25))
arr((54.0, 19.25), (56.2, 19.25))
arr((65.5, 32.0), (65.5, 23.0))
ax.text(66.2, 27.6, "$a^\\star=(k^\\star,e^\\star)$", ha="left", va="center", fontsize=6.6, rotation=90)
poly([(8.75, 22.5), (8.75, 19.25), (21.8, 19.25)])
ax.text(14.6, 20.5, "$b=S/\\kappa_k$", ha="center", fontsize=6.6)
arr((27.0, 23.0), (27.0, 31.0), ls=(0, (3, 2)))
ax.text(27.6, 27.0, "piggy-backed SNR $z$", ha="left", va="center", fontsize=6.0)
box(40.5, 25.2, 18.5, 4.2, "Event-triggered broadcast\n$|\\Delta W_c|>15$ s or 120 s", fs=6.2)
arr((47.0, 23.0), (47.0, 25.2))
poly([(59.0, 27.3), (61.0, 27.3), (61.0, 23.0)])

# ---------------------------------------------------------------- right side
group(74.5, 30.5, 25, 19.5, "Surface gateway")
box(76, 40.5, 22, 5.0, "Inference (large model)\n$A_{\\mathrm{GW}}=0.95$", fill="white")
box(76, 31.8, 22, 7.4, "Trust aggregation\n$T_i=0.3R_i+0.3B_i+0.4S_i$\n(Beta reputation, spot checks)", fill="#e9e9e9", fs=6.9)
group(74.5, 13.5, 25, 12.5, "Neighbour AUVs $i\\neq c$", fill="#fafafa")
box(76, 15.0, 22, 6.0, "Advertised backlog $W_i$,\nedge inference", fill="white")
poly([(70.4, 21.5), (73.0, 21.5), (73.0, 43.0), (76.0, 43.0)])
ax.text(72.2, 40.5, "to GW", ha="center", va="center", fontsize=6.0, rotation=90)
arr((70.4, 18.0), (76.0, 18.0))
ax.text(73.2, 19.0, "to $n^\\star$", ha="center", fontsize=6.2)
arr((76.0, 36.0), (70.5, 36.0))
ax.text(74.4, 37.0, "$\\mathbf{T}$", ha="center", fontsize=6.8)
poly([(95.0, 21.0), (95.0, 28.2), (68.5, 28.2), (68.5, 31.0)], ls=(0, (3, 2)))
ax.text(86.0, 29.2, "advertised loads $W_i$ (possibly stale)", fontsize=6.0, ha="center")

# ---------------------------------------------------------------- training box
group(0.5, 0.5, 99, 11.0, "Centralized training (offline, simulator / twin)", fill="white")
box(3, 2.0, 17, 6.0, "Centralized critic\n$V_\\psi(s)$", fill="#e9e9e9")
box(23.5, 2.0, 21.5, 6.0, "$\\tilde r_\\tau=r_\\tau-\\lambda_D\\mathbb{1}_{\\mathrm{miss}}-\\lambda_A\\mathbb{1}_{\\mathrm{acc}}$\n$\\hat A_\\tau=\\tilde r_\\tau-V_\\psi(s_\\tau)$", fill="#e9e9e9", fs=6.8)
box(48.5, 2.0, 18, 6.0, "PPO-clip update\nof $\\theta$ ($\\epsilon=0.2$)", fill="#e9e9e9")
box(70, 2.0, 27, 6.0, "Dual ascent\n$\\lambda\\leftarrow[\\lambda+\\eta_\\lambda(\\hat P-\\epsilon)]^+$", fill="#e9e9e9")
arr((20, 5.0), (23.5, 5.0)); arr((45, 5.0), (48.5, 5.0)); arr((70, 5.0), (66.5, 5.0))
poly([(57.5, 8.0), (57.5, 12.6), (62.0, 12.6), (62.0, 13.5)], "", None, ls=(0, (3, 2)))
ax.text(58.2, 10.0, "$\\theta$", fontsize=6.8)

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig("fig_arch.png", dpi=400)
fig.savefig("fig_arch.pdf")
print("ok")
