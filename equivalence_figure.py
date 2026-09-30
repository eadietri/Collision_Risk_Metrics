#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

CSV = "equivalence.csv"
R = 20.0

df = pd.read_csv(CSV)
x, y = df["x"].to_numpy(), df["y"].to_numpy()
dcpa, tcpa = df["dcpa"].to_numpy(), df["tcpa"].to_numpy()

holds_d = dcpa <= R
holds_t = tcpa > 0


def to_color(holds, mag, clip_pct=95):
    c = np.full(len(mag), 0.0)
    for mask, lo_c, hi_c in [(holds, -1.0, 0.0), (~holds, 0.0, 1.0)]:
        if not mask.any():
            continue
        vals = mag[mask]
        clip_val = np.percentile(vals, clip_pct)
        clipped = np.clip(vals, 0, clip_val)
        normed = clipped / clip_val if clip_val > 0 else np.zeros(mask.sum())
        c[mask] = lo_c + normed * (hi_c - lo_c)
    return c


dcpa_c = to_color(holds_d, dcpa)
tcpa_c = to_color(holds_t, np.abs(tcpa))
risk_c = np.maximum(dcpa_c, tcpa_c)

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
XI, YI = np.meshgrid(xi, yi)
zi_d = griddata((x, y), dcpa, (XI, YI), method='linear')
zi_t = griddata((x, y), tcpa, (XI, YI), method='linear')

panels = [
    (dcpa_c, "cpa_dcpa.pdf",
     lambda ax: ax.contour(XI, YI, zi_d, levels=[R], colors='black', linewidths=2, zorder=10)),
    (tcpa_c, "cpa_tcpa.pdf",
     lambda ax: ax.contour(XI, YI, zi_t, levels=[0], colors='black', linewidths=2, zorder=10)),
    (risk_c, "cpa_combined.pdf",
     lambda ax: ax.contour(XI, YI, np.minimum(R - zi_d, zi_t), levels=[0], colors='black', linewidths=2, zorder=10)),
]

# --- Individual PDFs (no labels, no ticks) ---
for data, out, draw_contour in panels:
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.scatter(x, y, c=data, cmap="PuRd", s=30, alpha=0.8,
               edgecolors="none", linewidths=0.15, vmin=-1, vmax=1)
    draw_contour(ax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out, bbox_inches="tight", pad_inches=0, facecolor='white')
    plt.close(fig)
    print(f"Wrote {out}")

# --- Combined PNG (with labels) ---
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharex=True, sharey=True, constrained_layout=True)

for (data, _, draw_contour), ax in zip(panels, axes):
    ax.scatter(x, y, c=data, cmap="PuRd", s=30, alpha=0.8,
               edgecolors="black", linewidths=0.15, vmin=-1, vmax=1)
    draw_contour(ax)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

OUT = "artifacts/cpa_scatter.png"
fig.savefig(OUT, dpi=200)
plt.close(fig)
print(f"Wrote {OUT}")