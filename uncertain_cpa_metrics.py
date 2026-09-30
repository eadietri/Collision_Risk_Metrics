
import numpy as np
from scipy.optimize import minimize, linprog

from utils.set_utils import Interval, Ellipse, Ship, Circle
from utils.metric_utils import TCPA, DCPA


def _solve(objective, P, V):
    """
    Minimize objective(x) where x = [p0, p1, v0, v1]
    subject to p ∈ P, v ∈ V.

    Supports Ellipse (unit-disk parametrisation via Cholesky of rQ)
    and Interval / any lo-hi box (unit-box parametrisation).
    """
    pieces = []
    idx = 0

    for S in (P, V):
        if isinstance(S, Ellipse):
            # x[idx:idx+2] = u with ||u|| <= 1
            # actual point = center + L @ u, where rQ = L L^T
            pieces.append(("ellipse", S, idx, 2))
            idx += 2
        elif Interval._is_interval(S):
            # x[idx:idx+2] = t with 0 <= t <= 1
            # actual point = lo + t * (hi - lo)
            pieces.append(("interval", S, idx, 2))
            idx += 2
        else:
            raise TypeError(f"Unsupported shape: {type(S)}")

    # precompute Cholesky factors once, not on every objective call
    chol = {id(S): np.linalg.cholesky(S.rQ) for _, S, _, _ in pieces
            if isinstance(S, Ellipse)}

    def unpack(x):
        results = []
        for kind, S, off, size in pieces:
            if kind == "ellipse":
                L = chol[id(S)]
                results.append(S.center + L @ x[off:off+2])
            else:  # interval
                lo = np.asarray(S.lo, dtype=float)
                hi = np.asarray(S.hi, dtype=float)
                results.append(lo + x[off:off+2] * (hi - lo))
        return results[0], results[1]

    def wrapped(x):
        p, v = unpack(x)
        return objective(np.concatenate([p, v]))

    constraints = []
    for kind, S, off, size in pieces:
        if kind == "ellipse":
            constraints.append({
                "type": "ineq",
                "fun": lambda x, o=off: 1.0 - x[o]**2 - x[o+1]**2,
            })
        else:  # interval: 0 <= t_i <= 1
            for i in range(size):
                constraints.append({
                    "type": "ineq",
                    "fun": lambda x, o=off, i=i: x[o+i],
                })
                constraints.append({
                    "type": "ineq",
                    "fun": lambda x, o=off, i=i: 1.0 - x[o+i],
                })

    # feasible start: origin for ellipse, box midpoint for interval
    x0 = np.zeros(idx)
    for kind, S, off, size in pieces:
        if kind == "interval":
            x0[off:off+size] = 0.5

    res = minimize(wrapped, x0, method="SLSQP", constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-14})
    return res.fun


def solve_tcpa_interval(P, V, **kw):
    """
    Problem 1: min/max TCPA(p, v) = -(p · v) / (v · v)
               s.t. p ∈ P, v ∈ V   (each Ellipse or Interval)
    """
    lo = _solve(lambda x: TCPA(x[:2], x[2:]), P, V, **kw)
    hi = -_solve(lambda x: -TCPA(x[:2], x[2:]), P, V, **kw)
    return lo, hi


def solve_dcpa_interval(P, V, **kw):
    """
    Problem 2: min/max DCPA(p, v) = (v × p) / ||v||
               s.t. p ∈ P, v ∈ V   (each Ellipse or Interval)
    """
    lo = _solve(lambda x: DCPA(x[:2], x[2:]), P, V, **kw)
    hi = -_solve(lambda x: -DCPA(x[:2], x[2:]), P, V, **kw)
    return lo, hi


def solve_dcpa_interval_unsigned(P, V, **kw):
    lo, hi = solve_dcpa_interval(P, V, **kw)
    if lo <= 0.0 <= hi:
        return 0.0, max(-lo, hi)
    return min(abs(lo), abs(hi)), max(abs(lo), abs(hi))


if __name__ == "__main__":
    # ---- encounter (disc domains so Metrics.TCPA/DCPA work) ---------------
    # (0, 0), (5, 0), (100, 100), (0, -5) Crossing
    own = Ship([0.0, 0.0], [5.0, 0.0], domain=Circle(10.0))
    target = Ship([100.0, 100.0], [0.0, -5.0], domain=Circle(10.0))

    # ---- nominal relative state -------------------------------------------
    p_nom = target.pos - own.pos
    v_nom = target.vel - own.vel

    # ---- uncertainty sets centered on nominal ------------------------------
    P = Ellipse.from_axes(10.0, 5.0, center=p_nom, theta=np.deg2rad(20))
    V = Interval(lo=v_nom - [1.5, 0.8], hi=v_nom + [1.5, 0.8])

    print("Position Ellipse (P): Q =", P.Q.tolist(), " r =", P.r,
          " center =", P.center.tolist())
    print("Velocity Interval (V):", V.lo, V.hi)


    # ---- solve the two separate problems ----------------------------------
    tcpa_lo, tcpa_hi = solve_tcpa_interval(P, V)
    dcpa_lo, dcpa_hi = solve_dcpa_interval(P, V)

    print("Under uncertainty (ellipse x box):")
    print(f"  TCPA ∈ [{tcpa_lo:.4f}, {tcpa_hi:.4f}] s")
    print(f"  DCPA ∈ [{dcpa_lo:.4f}, {dcpa_hi:.4f}] m")