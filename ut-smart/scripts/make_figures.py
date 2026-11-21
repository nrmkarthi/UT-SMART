#!/usr/bin/env python3
"""Aggregate results/*.csv (mean, 95% CI over seeds) into figures and tables.

Produces figures/fig_{convergence,sensors,auvs,channel,attack,sync}.png and
results/ablation.md, results/summary.md (with paired t-test p-values).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utsmart.config import LABELS  # noqa: E402

try:
    from scipy import stats as sstats
except ImportError:  # pragma: no cover
    sstats = None

STYLE = {
    "utsmart": dict(color="#1f5fbf", marker="o", lw=2.2),
    "mappo": dict(color="#d9822b", marker="s"),
    "greedy_sem": dict(color="#2a9d55", marker="^"),
    "raw_local": dict(color="#8a8a8a", marker="v"),
    "debr": dict(color="#b23a48", marker="D"),
    "wo_trust": dict(color="#1f5fbf", marker="o", ls="--", alpha=0.7),
}


def read_csv(path):
    import csv
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    t = sstats.t.ppf(0.975, len(x) - 1) if sstats else 2.262
    return float(x.mean()), float(t * x.std(ddof=1) / np.sqrt(len(x)))


def series(rows, method, metric, **filt):
    sel = [r for r in rows if r["method"] == method and all(r.get(k) == v for k, v in filt.items())]
    xs = sorted({r["value"] for r in sel})
    m, e = [], []
    for x in xs:
        mu, ci = ci95([r[metric] for r in sel if r["value"] == x])
        m.append(mu)
        e.append(ci)
    return np.array(xs), np.array(m), np.array(e)


def panel_fig(rows, metrics, xlabel, path, methods=None, scale=None, **filt):
    methods = methods or [m for m in STYLE if any(r["method"] == m for r in rows)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.3))
    for ax, (metric, ylabel) in zip(np.atleast_1d(axes), metrics):
        f = (scale or {}).get(metric, 1.0)
        for meth in methods:
            xs, m, e = series(rows, meth, metric, **filt)
            if len(xs) == 0:
                continue
            ax.errorbar(xs, m * f, yerr=e * f, capsize=3, label=LABELS[meth], ms=5, **STYLE[meth])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def convergence(train_dir, path):
    files = sorted(glob.glob(os.path.join(train_dir, "train_*_s*.csv")))
    if not files:
        return
    by_var = {}
    for f in files:
        v = os.path.basename(f)[6:].rsplit("_s", 1)[0]
        rows = read_csv(f)
        y = np.array([r["success_rate"] for r in rows])
        k = 20
        ma = np.convolve(y, np.ones(k) / k, mode="valid")
        by_var.setdefault(v, []).append(ma)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = plt.cm.tab10.colors
    for i, (v, curves) in enumerate(sorted(by_var.items())):
        n = min(len(c) for c in curves)
        C = np.stack([c[:n] for c in curves])
        x = np.arange(n) + 20
        ax.plot(x, C.mean(0), color=colors[i % 10], label=LABELS.get(v, v), lw=2 if v == "utsmart" else 1.2)
        ax.fill_between(x, C.min(0), C.max(0), color=colors[i % 10], alpha=0.15)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Success rate (MA-20)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def sync_fig(rows, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for mode, mk, col, unit in (("periodic", "s", "#d9822b", "s"), ("ondemand", "o", "#1f5fbf", "dB")):
        sel = [r for r in rows if r["mode"] == mode]
        xs = sorted({r["value"] for r in sel})
        px, py = [], []
        for x in xs:
            pts = [r for r in sel if r["value"] == x]
            px.append(np.mean([r["control_kbit_per_task"] for r in pts]))
            py.append(np.mean([r["twin_rmse"] for r in pts]))
            ax.annotate(f"{x:g}{unit}", (px[-1], py[-1]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.plot(px, py, marker=mk, color=col, label="periodic" if mode == "periodic" else "on-demand")
    ax.set_xscale("log")
    ax.set_xlabel("Control overhead (kbit/task)")
    ax.set_ylabel("Twin SNR RMSE (dB)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def paired_p(rows, a, b, metric, **filt):
    if sstats is None:
        return float("nan")
    xa = {r["seed"]: r[metric] for r in rows if r["method"] == a and all(r.get(k) == v for k, v in filt.items())}
    xb = {r["seed"]: r[metric] for r in rows if r["method"] == b and all(r.get(k) == v for k, v in filt.items())}
    s = sorted(set(xa) & set(xb))
    if len(s) < 2 or np.allclose([xa[i] for i in s], [xb[i] for i in s]):
        return float("nan")
    return float(sstats.ttest_rel([xa[i] for i in s], [xb[i] for i in s]).pvalue)


def ablation_table(rows, path):
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    lines = ["| Variant | Success, gateway (%) | Success, AUV-only (%) | p vs UT-SMART (AUV-only) | "
             "Corrupted, AUV-only (%) | Control (kbit/task) | Twin RMSE (dB) | Energy, AUV-only (J/task) |",
             "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        g = [r for r in rows if r["method"] == m and r["regime"] == "gateway"]
        e = [r for r in rows if r["method"] == m and r["regime"] == "auv_only"]
        sg, cg = ci95([100 * r["success_rate"] for r in g])
        se, ce = ci95([100 * r["success_rate"] for r in e])
        p = paired_p(rows, "utsmart", m, "success_rate", regime="auv_only") if m != "utsmart" else float("nan")
        lines.append(f"| {LABELS[m]} | {sg:.1f} ± {cg:.1f} | {se:.1f} ± {ce:.1f} | "
                     f"{'—' if np.isnan(p) else f'{p:.3g}'} | "
                     f"{100 * np.mean([r['corrupt_rate'] for r in e]):.1f} | "
                     f"{np.mean([r['control_kbit_per_task'] for r in g]):.2f} | "
                     f"{np.mean([r['twin_rmse'] for r in g]):.2f} | "
                     f"{np.mean([r['energy_per_task'] for r in e]):.1f} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


def default_point_table(rows, path):
    """Default point (M=100, N=5, gateway-assisted) from the sensor sweep."""
    sel = [r for r in rows if r["value"] == 100]
    metrics = [("success_rate", "Success (%)", 100), ("miss_rate", "Miss (%)", 100),
               ("acc_fail_rate", "Below A_min (%)", 100), ("corrupt_rate", "Corrupted (%)", 100),
               ("mean_delay", "Delay (s)", 1), ("energy_per_task", "Energy (J)", 1),
               ("traffic_kbit_per_task", "Traffic (kbit)", 1), ("control_kbit_per_task", "Control (kbit)", 1),
               ("twin_rmse", "Twin RMSE (dB)", 1)]
    lines = ["| Method | " + " | ".join(m[1] for m in metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for meth in STYLE:
        rr = [r for r in sel if r["method"] == meth]
        if not rr:
            continue
        cells = []
        for key, _, f in metrics:
            mu, ci = ci95([f * r[key] for r in rr])
            cells.append(f"{mu:.1f} ± {ci:.1f}" if key == "success_rate" else f"{mu:.2f}")
        lines.append(f"| {LABELS[meth]} | " + " | ".join(cells) + " |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = lambda n: read_csv(os.path.join(a.results, f"{n}.csv")) if os.path.exists(
        os.path.join(a.results, f"{n}.csv")) else None
    pct = {"success_rate": 100, "miss_rate": 100, "corrupt_rate": 100}
    convergence(os.path.join(a.results, "train"), os.path.join(a.out, "fig_convergence.png"))
    if (r := R("sensors")):
        panel_fig(r, [("success_rate", "Mission success (%)"), ("energy_per_task", "Energy (J/task)"),
                      ("control_kbit_per_task", "Control overhead (kbit/task)")],
                  "Number of sensors M", os.path.join(a.out, "fig_sensors.png"), scale=pct)
        default_point_table(r, os.path.join(a.results, "summary.md"))
    if (r := R("auvs")):
        panel_fig(r, [("success_rate", "Mission success (%)"), ("mean_delay", "Mean delay (s)"),
                      ("traffic_kbit_per_task", "Acoustic traffic (kbit/task)")],
                  "Number of AUVs N", os.path.join(a.out, "fig_auvs.png"), scale=pct)
    if (r := R("channel")):
        panel_fig(r, [("success_rate", "Mission success (%)"), ("miss_rate", "Deadline miss (%)"),
                      ("twin_rmse", "Twin SNR RMSE (dB)")],
                  "Shadowing σ (dB)", os.path.join(a.out, "fig_channel.png"), scale=pct)
    if (r := R("attack")):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.3))
        for ax, (regime, metric, yl) in zip(axes, [("gateway", "success_rate", "Success, gateway-assisted (%)"),
                                                  ("auv_only", "success_rate", "Success, AUV-only edge (%)"),
                                                  ("auv_only", "corrupt_rate", "Corrupted accepted (%)")]):
            for meth in STYLE:
                xs, m, e = series(r, meth, metric, regime=regime)
                if len(xs):
                    ax.errorbar(100 * xs, 100 * m, yerr=100 * e, capsize=3, label=LABELS[meth], ms=5,
                                **STYLE[meth])
            ax.set_xlabel("Compromised AUVs (%)")
            ax.set_ylabel(yl)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig_attack.png"), dpi=200)
        plt.close(fig)
        print("wrote", os.path.join(a.out, "fig_attack.png"))
    if (r := R("sync")):
        sync_fig(r, os.path.join(a.out, "fig_sync.png"))
    if (r := R("ablation")):
        ablation_table(r, os.path.join(a.results, "ablation.md"))


if __name__ == "__main__":
    main()
