import numpy as np
from matplotlib.patches import Polygon 

_BIG = 1e6 

def plot_collision_cone(cone, ax, apex=(0.0, 0.0),
                        color="#d62728", alpha=0.15, lw=1.5, ls="-", label=None):
    apex = np.asarray(apex, dtype=float)
    if cone.contains_origin:
        ax.annotate("collision cone = every direction", apex, color=color,
                    fontsize=9, ha="center")
        ax.plot(*apex, "o", color=color, ms=6)
        return ax

    e1, e2 = cone.edges
    wedge = Polygon([apex, apex + e1 * _BIG, apex + e2 * _BIG], closed=True,
                     facecolor=color, edgecolor="none", alpha=alpha, label=label)
    ax.add_artist(wedge)

    for e in (e1, e2):
        far = apex + e * _BIG          # ray endpoint, not a through-line
        ax.plot([apex[0], far[0]], [apex[1], far[1]],
                 color=color, lw=lw, ls=ls)

    return ax


def plot_inflated_cone(icone, ax, apex=(0.0, 0.0), color="#d62728",
                       alpha=0.15, lw=1.5, ls="-", label=None):
    apex = np.asarray(apex, dtype=float)
    if icone.contains_origin:
        ax.annotate("inflated cone = every direction", apex, color=color,
                    fontsize=9, ha="center")
        ax.plot(*apex, "o", color=color, ms=6)
        return ax

    (p1, p2), (e1, e2) = icone.tangent_points, icone.edges
    verts = np.vstack([apex + p1 + e1 * _BIG, apex + icone.cap, apex + p2 + e2 * _BIG])
    ax.add_artist(Polygon(verts, closed=True, facecolor=color, edgecolor="none",
                          alpha=alpha, label=label))
    for p, e in ((p1, e1), (p2, e2)):
        far = apex + p + e * _BIG
        ax.plot([apex[0] + p[0], far[0]], [apex[1] + p[1], far[1]],
                color=color, lw=lw, ls=ls)
    cap = apex + icone.cap
    ax.plot(cap[:, 0], cap[:, 1], color=color, lw=lw, ls=ls)
    return ax