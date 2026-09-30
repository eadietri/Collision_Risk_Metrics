
import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq

from utils.set_utils import MinkowskiSum

def TCPA(prel, vrel):
    """Time to closest point of approach (s). Negative => already past it."""
    speed_sq = float(np.dot(vrel, vrel))
    if speed_sq < 1e-12:              # parallel, no relative motion
        return 0.0
    return -float(np.dot(prel, vrel)) / speed_sq

def DCPA(prel, vrel):
    """Signed distance at closest point of approach (m)."""
    speed_sq = float(np.dot(vrel, vrel))
    if speed_sq < 1e-12:
        return float(np.linalg.norm(prel))
    cross = float(vrel[0] * prel[1] - vrel[1] * prel[0])         # v x r (signed)
    return cross / float(np.sqrt(speed_sq))


def _dir(phi):
    return np.array([np.cos(phi), np.sin(phi)])


class Translated:
    """A shape shifted by a fixed offset: {x + t : x in S}."""

    def __init__(self, S, offset):
        self.S = S
        self.offset = np.asarray(offset, dtype=float)

    def support(self, d):
        return self.S.support(d) + self.offset @ np.asarray(d, dtype=float)

    def support_point(self, d):
        return self.S.support_point(d) + self.offset

    def reflect(self):
        return Translated(self.S.reflect(), -self.offset)

    def __repr__(self):
        return f"Translated(S={self.S!r}, offset={self.offset.tolist()})"


@dataclass
class CollisionCone:
    contains_origin: bool = False
    normals: np.ndarray | None = None   # (2,2) outward tangent normals
    edges: np.ndarray | None = None     # (2,2) unit rays along cone edges
    contacts: np.ndarray | None = None  # (2,2) tangent contact points on O


def collision_cone(O, n_grid=1440):
    """Collision cone for convex obstacle O, via the two tangents from the origin.

    Returns CollisionCone(contains_origin=True) if the origin is inside O.
    """
    g = lambda phi: O.support(_dir(phi))

    phis = np.linspace(0, 2 * np.pi, n_grid)
    vals = np.array([g(p) for p in phis])
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0] + 1
    if len(idx) == 0:
        return CollisionCone(contains_origin=True)

    roots = [brentq(g, phis[i - 1], phis[i]) for i in idx]
    normals = np.array([_dir(r) for r in roots])
    contacts = np.array([O.support_point(d) for d in normals])
    edges = contacts / np.linalg.norm(contacts, axis=1, keepdims=True)

    return CollisionCone(normals=normals, edges=edges, contacts=contacts)

@dataclass
class InflatedCone:
    """Cone (+) K: a wedge with edges offset outward and a rounded corner."""
    contains_origin: bool = False
    tangent_points: np.ndarray | None = None  # (2,2) offset points p1, p2
    edges: np.ndarray | None = None            # (2,2) same ray directions e1, e2 (unchanged)
    cap: np.ndarray | None = None              # (n,2) arc of K's boundary rounding the corner


def inflate_cone(cone, K, n_cap=60):
    """Minkowski sum of a CollisionCone (wedge) with a bounded convex shape K.

    Offsets each edge outward by K's support point in its outward-normal
    direction, and rounds the corner with the arc of K's boundary swept
    between the two normals (the arc facing away from the cone interior).
    """
    if cone.contains_origin:
        return InflatedCone(contains_origin=True)

    d1, d2 = cone.normals
    e1, e2 = cone.edges

    p1 = K.support_point(d1)
    p2 = K.support_point(d2)

    phi1 = np.arctan2(d1[1], d1[0])
    phi2 = np.arctan2(d2[1], d2[0])
    delta = (phi2 - phi1) % (2 * np.pi)
    if delta > np.pi:                    # always take the minor (outward) arc
        phi1, phi2 = phi2, phi1
        delta = 2 * np.pi - delta
    phis = phi1 + np.linspace(0.0, delta, n_cap)
    cap = np.array([K.support_point(_dir(p)) for p in phis])

    return InflatedCone(tangent_points=np.array([p1, p2]),
                        edges=np.array([e1, e2]), cap=cap)

class VelocityObstacle:
    def __init__(self, O, v_target, n_grid=1440):
        self.apex = np.asarray(v_target, dtype=float)
        self.cone = collision_cone(O, n_grid=n_grid)
        self.O = O

    @classmethod
    def from_ships(self, own, target, n_grid=1440):
        """O = D_target (+) (-D_own), translated to relpos = target.pos - own.pos."""
        relpos = target.pos - own.pos
        O = Translated(MinkowskiSum(target.shape, own.shape.reflect()), relpos)
        return self(O, target.vel, n_grid=n_grid)

    def inflate(self, K, n_cap=60):
        """VO (+) K in velocity space -- Stage 2 of the definition, applied
        directly to the cone (not re-derived via tangent search)."""
        return inflate_cone(self.cone, K, n_cap=n_cap)

    @property
    def edges(self):
        return self.cone.edges

    def contains(self, v_own):
        """True if v_own puts the relative velocity inside the collision cone."""
        if self.cone.contains_origin:
            return True

        w = np.asarray(v_own, dtype=float) - self.apex
        e1, e2 = self.cone.edges
        cross_between = e1[0] * e2[1] - e1[1] * e2[0]
        cross1 = e1[0] * w[1] - e1[1] * w[0]
        cross2 = e2[0] * w[1] - e2[1] * w[0]
        return bool(cross1 * cross_between >= 0 and cross2 * cross_between <= 0)

from matplotlib.path import Path

def point_in_inflated_cone(icone, apex, points, far=1e4):
    """Vectorized: True where each point in `points` (n,2) lies inside the
    inflated (rounded) VO wedge icone, whose apex sits at `apex`."""
    apex = np.asarray(apex, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))

    if icone.contains_origin:
        return np.ones(len(points), dtype=bool)

    p1, p2 = icone.tangent_points
    e1, e2 = icone.edges

    # boundary: far tip on edge1 -> p1 -> cap arc (p1 to p2) -> p2 -> far tip on edge2
    # (the segment closing the two far tips is arbitrary/at "infinity" — irrelevant
    # as long as `far` exceeds the range of points you're testing)
    poly = np.vstack([
        p1 + far * e1,
        p1,
        icone.cap,
        p2,
        p2 + far * e2,
    ])

    path = Path(poly)
    return path.contains_points(points - apex)


def point_in_cone(cone, apex, points):
    """Vectorized: True where each point in `points` (n,2) lies inside the
    (unbounded) wedge defined by cone.edges."""
    apex = np.asarray(apex, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))

    if cone.contains_origin:
        return np.ones(len(points), dtype=bool)

    e1, e2 = cone.edges
    d = points - apex  # (n, 2)

    cross1 = e1[0] * d[:, 1] - e1[1] * d[:, 0]
    cross2 = e2[0] * d[:, 1] - e2[1] * d[:, 0]
    cross_between = e1[0] * e2[1] - e1[1] * e2[0]

    return (cross1 * cross_between >= 0) & (cross2 * cross_between <= 0)