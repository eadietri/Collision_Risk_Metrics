import numpy as np


class Ellipse:
    """ 
    Ellipse: {x : (x-c)^T Q^{-1} (x-c) <= r}.
    Parameterized by a 2x2 SPD shape matrix Q, a radius scalar r,
    and a center c. Support function and support point are O(1).
    """

    def __init__(self, Q, center=(0.0, 0.0), r=1.0):
        self.Q = np.asarray(Q, dtype=float)
        self.r = float(r)
        self.center = np.asarray(center, dtype=float)

    @classmethod
    def from_axes(cls, a, b, center=(0.0, 0.0), theta=0.0):
        """Construct from semi-axes (a, b), center, and rotation theta (rad)."""
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        D = np.diag([a * a, b * b])
        Q = R @ D @ R.T
        return cls(Q, center=center, r=1.0)

    @property
    def rQ(self):
        return self.r * self.Q

    def support(self, d):
        d = np.asarray(d, dtype=float)
        quad = d @ (self.rQ @ d)
        return float(self.center @ d + np.sqrt(quad))

    def support_point(self, d):
        d = np.asarray(d, dtype=float)
        rQ = self.rQ
        rQd = rQ @ d
        denom = np.sqrt(d @ rQd)
        return self.center + rQd / denom

    def reflect(self):                     # an ellipse is centrally symmetric
        return Ellipse(self.Q, -self.center, self.r)

    def sample_boundary(self, n=180):
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        dirs = np.column_stack([np.cos(theta), np.sin(theta)])
        return np.array([self.support_point(d) for d in dirs])

    def sample_ellipse_interior(self, number_of_points=1, rng=None):
        """Uniform samples from interior of an Ellipse."""
        if rng is None:
            rng = np.random.default_rng()
        z = rng.standard_normal((number_of_points, 2))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        r = np.sqrt(rng.uniform(size=(number_of_points, 1)))
        L = np.linalg.cholesky(self.rQ)          # rQ = L L^T, plays the role of M
        return self.center + (r * z) @ L.T

    def __repr__(self):
        return (f"Ellipse(Q={self.Q.tolist()}, "
                f"center={self.center.tolist()}, r={self.r})")

    def __str__(self):
        return type(self).__name__

class Interval:
    """
    Axis-aligned box (interval) in R^2: {x : lo <= x <= hi}, elementwise.
    """

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        if self.lo.shape != (2,) or self.hi.shape != (2,):
            raise ValueError("Interval expects 2-vectors lo, hi")
        if np.any(self.hi < self.lo):
            raise ValueError("hi must be >= lo elementwise")

    @property
    def center(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def half_widths(self):
        return 0.5 * (self.hi - self.lo)

    def __repr__(self):
        return f"Interval(lo={self.lo}, hi={self.hi})"

    def _is_interval(S):
        """True for an Interval, or any duck-typed box exposing .lo and .hi."""
        return isinstance(S, Interval) or (hasattr(S, "lo") and hasattr(S, "hi"))

    def _sample_box(self, n_samples, rng):
        """
        Sample n_samples points (n,2) from Interval (any object exposing `.lo` and `.hi`,
            each length-2 array-like) -- uniform sampling inside the box.
        """
        if hasattr(self, "lo") and hasattr(self, "hi"):
            lo = np.asarray(self.lo, dtype=float)
            hi = np.asarray(self.hi, dtype=float)
            return rng.uniform(lo, hi, size=(n_samples, 2))

class Circle():
    """Circle of radius r centered at ``center``"""

    def __init__(self, r, center=(0.0, 0.0)):
        self.r = float(r)
        self.center = np.asarray(center, dtype=float)

    def support(self, d):
        d = np.asarray(d, dtype=float)
        return float(self.center @ d + self.r * np.linalg.norm(d))

    def support_point(self, d):
        d = np.asarray(d, dtype=float)
        return self.center + self.r * d / np.linalg.norm(d)

    def reflect(self):                     # a disk is centrally symmetric
        return Circle(self.r, -self.center)

    def __repr__(self):
        return f"Circle(r={self.r}, center={self.center.tolist()})"
    
class Ship:
    def __init__(self, pos, vel, R=10.0, shape=None):
        """
        Parameters
        ----------
        pos : array_like, shape (2,)
            Position.
        vel : array_like, shape (2,)
            Constant velocity.
        domain : Shape, optional
            Convex footprint / safety region. Defaults to a radius-10 disc.
        """
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.R = R
        self.shape = shape if shape is not None else Circle(R, center=(0.0, 0.0))

    def position_at(self, t):
        """Position after elapsed time ``t`` under constant velocity (non-mutating)."""
        return self.pos + self.vel * t

    def step(self, t):
        """Advance the position by elapsed time ``t`` (mutating)."""
        self.pos = self.pos + self.vel * t

    def __repr__(self):
        return f"Ship({self.domain}, pos={self.pos.tolist()}, vel={self.vel.tolist()})"


class MinkowskiSum:
    """Minkowski sum of two convex shapes: A (+) B.
    """

    def __init__(self, A, B):
        self.A = A
        self.B = B

    def support(self, d):
        return self.A.support(d) + self.B.support(d)

    def support_point(self, d):
        return self.A.support_point(d) + self.B.support_point(d)

    def reflect(self):                         # -(A(+)B) = (-A)(+)(-B)
        return MinkowskiSum(self.A.reflect(), self.B.reflect())

    def __repr__(self):
        return f"MinkowskiSum(A={self.A!r}, B={self.B!r})"