"""
Visualise deterministic vs interval TCPA/DCPA.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as EllipsePatch

from examples.uncertain_cpa_metrics import solve_tcpa_interval, solve_dcpa_interval, Interval
from utils.set_utils import Ellipse, Interval, Ship, Circle, MinkowskiSum
from utils.metric_utils import TCPA, DCPA, VelocityObstacle, point_in_cone, point_in_inflated_cone
from utils.plotting_utils import plot_collision_cone, plot_inflated_cone

def combined_safety_radius(ego, obstacle):
    """Combined disc clearance (own + target radii). Circular domains only."""
    return ego.R + obstacle.R

def _sample_shape(S, n_samples, rng):
    """
    Sample n_samples points from shape S.

    Supports:
      - Ellipse (delegates to _sample_ellipse)
      - Interval (delegates to _sample_box)
    """
    if Interval._is_interval(S):
        return S._sample_box(n_samples, rng)
    return S.sample_ellipse_interior(n_samples, rng)

def plot_intervals(tcpa_det, dcpa_det, tcpa_iv, dcpa_iv):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    for ax, det, iv, label, unit in zip(
        axes,
        [tcpa_det, dcpa_det],
        [tcpa_iv, dcpa_iv],
        ["TCPA", "DCPA"],
        ["s", "m"],
    ):
        lo, hi = iv
        margin = 0.15 * max(hi - lo, abs(det) * 0.1, 0.1)
        ax.barh(0, hi - lo, left=lo, height=0.4,
                color="steelblue", alpha=0.35, label="Interval")
        ax.plot([lo, hi], [0, 0], "s", color="steelblue", ms=8)
        ax.plot(det, 0, "D", color="red", ms=10, zorder=5,
                label="Deterministic")
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(-0.8, 0.8)
        ax.set_yticks([])
        ax.set_xlabel(f"{label} [{unit}]")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{label}: {det:.2f} ∈ [{lo:.2f}, {hi:.2f}]")

    fig.tight_layout()
    return fig


def plot_tcpa_dcpa_cloud(P, V, tcpa_iv, dcpa_iv, R, n_samples=10000):
    PURD_DARK = "#CE1256"
    TEAL = "#16A085"
    """
    joint TCPA-DCPA scatter from sampled uncertainty.

    P, V may each be an Ellipse or an axis-aligned box (Interval, or any
    object exposing `.lo`/`.hi`); sampling is dispatched accordingly.
    """
    rng = np.random.default_rng()
    ps = _sample_shape(P, n_samples, rng)
    vs = _sample_shape(V, n_samples, rng)

    tcpa_vals = np.zeros(n_samples)
    dcpa_vals = np.empty(n_samples)
    
    for i in range(n_samples):
        p, v = ps[i], vs[i]
        vv = v @ v
        tcpa_vals[i] = TCPA(p, v)
        dcpa_vals[i] = DCPA(p, v)

    # Collision condition: |DCPA| <= R AND TCPA > 0
    hit = (np.abs(dcpa_vals) <= R) & (tcpa_vals > 0)

    # Inside vo condition box
    in_box = ((tcpa_vals >= tcpa_iv[0]) & (tcpa_vals <= tcpa_iv[1]) &
              (dcpa_vals >= dcpa_iv[0]) & (dcpa_vals <= dcpa_iv[1]))

    # Zoom out: robust percentiles + padding
    xpad = 0.25 * (tcpa_vals.max() - tcpa_vals.min())
    xlo, xhi = tcpa_vals.min() - xpad, tcpa_vals.max() + xpad
    ypad = 0.25 * (dcpa_vals.max() - dcpa_vals.min())
    ymax = max(abs(dcpa_vals.min()), abs(dcpa_vals.max())) + ypad

    def _draw(ax):

        ax.axhspan(dcpa_iv[0], dcpa_iv[1], color=TEAL, alpha=0.15, zorder=1)
        ax.axvspan(xlo, tcpa_iv[0], color="white", zorder=1)
        ax.axvspan(tcpa_iv[1], xhi, color="white", zorder=1)
        ax.axvline(tcpa_iv[0], color=TEAL,  lw=2, zorder=1)
        ax.axvline(tcpa_iv[1], color=TEAL, lw=2, zorder=1)
        ax.axhline(dcpa_iv[0], color=TEAL, lw=2, zorder=1)
        ax.axhline(dcpa_iv[1], color=TEAL,  lw=2, zorder=1)

        ax.axhspan(-R, R, color="white", zorder=1)
        ax.axhline(-R, color=PURD_DARK, lw=2, zorder=1)
        ax.axhline(R, color=PURD_DARK, lw=2, zorder=1)
        ax.axhspan(-R, R, color=PURD_DARK, alpha=0.15, zorder=1)

        ax.scatter(tcpa_vals[~hit & ~in_box], dcpa_vals[~hit & ~in_box], s=4, alpha=0.2,
                   color=TEAL, zorder=2)
        ax.scatter(tcpa_vals[~hit & in_box], dcpa_vals[~hit & in_box], s=4, alpha=0.3,
                   color=TEAL, zorder=2)
        ax.scatter(tcpa_vals[hit], dcpa_vals[hit], s=4, alpha=0.4,
                   color=PURD_DARK, zorder=3)

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-ymax, ymax)
        ax.grid(True, alpha=0.25, zorder=3)
        print("points inside interval box:", np.sum(hit)/n_samples)

    # PNG with ticks
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw(ax)
    fig.tight_layout()

    # PDF without ticks/labels
    fig_pdf, ax_pdf = plt.subplots(figsize=(9, 5))
    fig_pdf.patch.set_facecolor('white')
    ax_pdf.set_facecolor('white')
    _draw(ax_pdf)
    ax_pdf.set_xticks([])
    ax_pdf.set_yticks([])
    ax_pdf.spines[:].set_visible(False)
    # Force grid using minor ticks since major ticks are removed
    # ax_pdf.set_axisbelow(True)
    ax_pdf.xaxis.set_zorder(1.5)
    ax_pdf.yaxis.set_zorder(1.5)
    ax_pdf.xaxis.set_major_locator(plt.AutoLocator())
    ax_pdf.yaxis.set_major_locator(plt.AutoLocator())
    ax_pdf.tick_params(length=0, labelbottom=False, labelleft=False)
    ax_pdf.grid(True, alpha=0.25, zorder=3)
    fig_pdf.tight_layout()

    print(f"xlo={xlo:.4f}, xhi={xhi:.4f}, ylo={-ymax:.4f}, yhi={ymax:.4f}")

    return fig, fig_pdf


def plot_enlarged_velocity_obstacle(ego, obstacle, original_VO, VO_with_position_uncertainty, VO_with_velocity_uncertainty, vos, ves, n_samples):
    """
    Deterministic VO vs. the position-uncertainty-inflated ('enlarged') VO.
    Both are cones (VelocityObstacle instances) sharing the same apex.
    """
    PURD_DARK = "#CE1256"
    TEAL = "#8E44AD"

    apex = original_VO.apex
    inside = point_in_cone(original_VO.cone, obstacle.vel, ves)
    inside_uvo = point_in_inflated_cone(VO_with_velocity_uncertainty, obstacle.vel, ves)
    n_in, n_out = inside.sum(), (~inside).sum()
    n_in_uvo, n_out_uvo = inside_uvo.sum(), (~inside_uvo).sum()

    # frame computed once, shared by both figures (cones are unbounded,
    # so they don't contribute finite points -- matplotlib clips the wedge)
    frame = np.vstack([[0.0, 0.0], apex, ego.vel, ves])
    lo, hi = frame.min(axis=0)-2, 2+frame.max(axis=0)
    center = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo).max() * 1.15, 1.0)

    def _draw(ax, with_labels):
        ax.scatter(ves[~inside, 0], ves[~inside, 1], s=4, alpha=0.5,
                   color=TEAL, zorder=4,
                   label=f"outside VO (n={n_out})" if with_labels else None)
        ax.scatter(ves[inside, 0], ves[inside, 1], s=4, alpha=0.5,
                   color=PURD_DARK, zorder=4,
                   label=f"inside VO (n={n_in})" if with_labels else None)

        plot_collision_cone(original_VO.cone, ax, apex=original_VO.apex, color=PURD_DARK,
                            alpha=0.5, lw=1.5, ls="--",
                            label="deterministic VO" if with_labels else None)
        # plot_collision_cone(VO_with_position_uncertainty.cone, ax,
        #                     apex=VO_with_position_uncertainty.apex, color=PURD_DARK,
        #                     alpha=0.25, lw=1.5,
        #                     label="enlarged VO" if with_labels else None)
        plot_inflated_cone(VO_with_velocity_uncertainty, ax,
                    apex=VO_with_position_uncertainty.apex, color=PURD_DARK,
                    alpha=0.25, lw=1.5,
                    label="velocity enlarged VO" if with_labels else None)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.grid(True, alpha=0.3)
        print("points inside VO:", n_in, "prob of collision in VO: ", n_in/n_samples)
        print("points outside UVO:", n_in_uvo, "prob of collision in UVO: ", n_in_uvo/n_samples)

    # ---- PNG: full annotation ----
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw(ax, with_labels=True)
    ax.set_xlabel(r"$v_{East}$ [m/s]")
    ax.set_ylabel(r"$v_{North}$ [m/s]")
    ax.set_title("Velocity space: VO enlarged by position uncertainty")
    ax.legend(fontsize=9)
    fig.tight_layout()

    # ---- PDF: bare, no labels/legend/ticks/spines ----
    fig_pdf, ax_pdf = plt.subplots(figsize=(6, 6))
    _draw(ax_pdf, with_labels=False)
    ax_pdf.spines[:].set_visible(False)
    ax_pdf.set_axisbelow(True)
    ax_pdf.xaxis.set_major_locator(plt.AutoLocator())
    ax_pdf.yaxis.set_major_locator(plt.AutoLocator())
    ax_pdf.tick_params(length=0, labelbottom=False, labelleft=False)
    ax_pdf.grid(True, alpha=0.25)
    fig_pdf.tight_layout()

    print(f"xlo={center[0] - half:.4f}, xhi={center[0] + half:.4f}, "
          f"ylo={center[1] - half:.4f}, yhi={center[1] + half:.4f}")
    return fig, fig_pdf
 

if __name__ == "__main__":
    # ---- encounter --------------------------------------------------------
    ego = Ship([0.0, 0.0], [2.0, 0.0], 10.0, shape=Circle(10.0)) #Ship([0.0, 0.0], [2.0, 0.0], R=10.0, shape=Circle(10.0))
    obstacle = Ship([100.0, 100.0], [0.0, -5.0], 10.0, shape=Circle(10.0))#Ship([100.0, 100.0], [0.0, -5.0], R=10.0, shape=Circle(10.0))

    # ---- relative state -------------------------------------------
    p_rel = obstacle.pos - ego.pos
    v_rel = obstacle.vel - ego.vel
 
    P = Ellipse.from_axes(30.0, 25.0, center=p_rel, theta=0.0)
    Ve = Ellipse.from_axes(1.5, 0.8, center=ego.vel, theta=0.0)
    Vo = Ellipse.from_axes(1.5, 0.8, center=obstacle.vel, theta=0.0)
    Kv = Ellipse(Vo.Q, center=(0.0, 0.0), r=Vo.r) 
    # Calculate convex interval for Vrel = Vo - Ve
    n_samples = 10000
    rng = np.random.default_rng()
    ves = Ve.sample_ellipse_interior(n_samples, rng)
    vos = Vo.sample_ellipse_interior(n_samples, rng)

    # vectorized instead of the per-sample loop
    vrels = vos - ves  # shape (n_samples, 2)

    # collapse the sample cloud into an axis-aligned box (Interval)
    vrel_lo = vrels.min(axis=0)
    vrel_hi = vrels.max(axis=0)
    Vrel = Interval(vrel_lo, vrel_hi)
 
    # ---- deterministic ----------------------------------------------------
    tcpa_det = TCPA(p_rel, v_rel)
    dcpa_det = DCPA(p_rel, v_rel)
 
    # ---- VO ---------------------------------------------------------------
    original_VO =  VelocityObstacle.from_ships(ego, obstacle)
    VO_with_position_uncertainty = VelocityObstacle(MinkowskiSum(original_VO.O, P), obstacle.vel)
    VO_with_velocity_uncertainty = VO_with_position_uncertainty.inflate(Kv)
 
    # ---- intervals --------------------------------------------------------
    tcpa_iv = solve_tcpa_interval(P, Vrel)
    dcpa_iv = solve_dcpa_interval(P, Vrel)
 
    print(f"Deterministic:  TCPA={tcpa_det:.4f} s,  DCPA={dcpa_det:.4f} m")
    print(f"Interval:       TCPA∈[{tcpa_iv[0]:.4f}, {tcpa_iv[1]:.4f}] s")
    print(f"                DCPA∈[{dcpa_iv[0]:.4f}, {dcpa_iv[1]:.4f}] m")
 
    # ---- figures ----------------------------------------------------------
    out = Path("artifacts/interval_plots")
    out.mkdir(parents=True, exist_ok=True)

    fig_vo_png, fig_vo_pdf = plot_enlarged_velocity_obstacle(ego, obstacle, original_VO, VO_with_position_uncertainty, VO_with_velocity_uncertainty, vos, ves, n_samples)
    fig_vo_png.savefig(out / "enlarged_vo.png", dpi=150, bbox_inches="tight")
    fig_vo_pdf.savefig(out / "enlarged_vo.pdf")
    
    fig_png, fig_pdf = plot_tcpa_dcpa_cloud(
        P, Vrel, tcpa_iv, dcpa_iv, R=combined_safety_radius(ego, obstacle))
    fig_png.savefig(out / "tcpa_dcpa_cloud.png", dpi=150, bbox_inches="tight")
    fig_pdf.savefig(out / "tcpa_dcpa_cloud.pdf", bbox_inches="tight",
                    pad_inches=0, facecolor="white")
 
    plt.close("all")
    print(f"\nFigures saved to {out}/")


 