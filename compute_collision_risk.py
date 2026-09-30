import numpy as np

from utils.set_utils import Ellipse, Ship, Circle, Interval, MinkowskiSum
from utils.metric_utils import DCPA, TCPA, point_in_inflated_cone, VelocityObstacle, point_in_cone

def _sample_ellipse(E, n=5000, rng=None):
    """Uniform samples from interior of an Ellipse."""
    if rng is None:
        rng = np.random.default_rng(0)
    z = rng.standard_normal((n, 2))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    r = np.sqrt(rng.uniform(size=(n, 1)))
    return E.center + (r * z) @ E.M.T

def combined_safety_radius(ego, obstacle):
    """Combined disc clearance (own + target radii). Circular domains only."""
    return ego.R + obstacle.R


if __name__ == "__main__":
    scenarios = [
        dict(
            name="crossing_high",
            ego=Ship([0.0, 0.0], [11.0, 0.0], 10.0, shape=Circle(10.0)),
            obstacle=Ship([100.0, 100.0], [0.0, -5.0], 10.0, shape=Circle(10.0)),
            p_axes=(30.0, 25.0),
            v_axes=(1.5, 0.8),
            ),
        dict(
            name="crossing_low",
            ego=Ship([0.0, 0.0], [-1.0, 0.0], 10.0, shape=Circle(10.0)),
            obstacle=Ship([100.0, 100.0], [0.0, -5.0], 10.0, shape=Circle(10.0)),
            p_axes=(30.0, 25.0),
            v_axes=(1.5, 0.8),
        ),
        dict(
            name="crossing",
            ego=Ship([0.0, 0.0], [5.0, 0.0], 10.0, shape=Circle(10.0)),
            obstacle=Ship([100.0, 100.0], [0.0, -5.0], 10.0, shape=Circle(10.0)),
            p_axes=(30.0, 25.0),
            v_axes=(1.5, 0.8),
        ),
        dict(
            name="headon",
            ego=Ship([0.0, 0.0], [0.0, 5.0], 10.0, shape=Circle(10.0)),
            obstacle=Ship([0.0, 200.0], [0.0, -5.0], 10.0, shape=Circle(10.0)),
            p_axes=(30.0, 25.0),
            v_axes=(1.5, 0.8),
        ),
        dict(
            name="overtaking",
            ego=Ship([0.0, 0.0], [0.0, 8.0], 10.0, shape=Circle(10.0)),
            obstacle=Ship([0.0, 120.0], [0.0, 4.0], 10.0, shape=Circle(10.0)),
            p_axes=(30.0, 25.0),
            v_axes=(1.5, 0.8),
        ),
        dict(
            name="collinear",
            ego=Ship([0.0, 0.0], [0.0, -5.0], 10.0, shape=Circle(10.0)),
            obstacle=Ship([0.0, 200.0], [0.0, -5.0], 10.0, shape=Circle(10.0)),
            p_axes=(30.0, 25.0),
            v_axes=(1.5, 0.8),
        ), 
        dict(
            name="parallel",
            ego=Ship([0.0, 0.0], [0.0, 5.0], 10.0, shape=Circle(10.0)),
            obstacle=Ship([150.0, 0.0], [0.0, 5.0], 10.0, shape=Circle(10.0)),
            p_axes=(30.0, 25.0),
            v_axes=(1.5, 0.8),
        ),
    ]

    results = {}
    for sc in scenarios:
        ego, obstacle = sc["ego"], sc["obstacle"]

        prel = obstacle.pos - ego.pos
        vrel = obstacle.vel - ego.vel

        P = Ellipse.from_axes(*sc["p_axes"], center=prel, theta=0.0)
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

        original_VO =  VelocityObstacle.from_ships(ego, obstacle)
        VO_with_position_uncertainty = VelocityObstacle(MinkowskiSum(original_VO.O, P), obstacle.vel)
        VO_with_velocity_uncertainty = VO_with_position_uncertainty.inflate(Kv)

        results[sc["name"]] = dict(
            ego=ego, obstacle=obstacle, prel=prel, vrel=vrel, P=P, vrels=vrels, ves=ves, V=Vrel,
            og_VO=original_VO, pos_VO=VO_with_position_uncertainty, vel_VO=VO_with_velocity_uncertainty
        )

    print("##################################################################")

    n_samples = 10000
    rng = np.random.default_rng()

    for name, r in results.items():
        VO_with_velocity_uncertainty = r["vel_VO"]
        og_VO = r["og_VO"]
        ego, obstacle = r["ego"], r["obstacle"]
        P, vrels, ves = r["P"], r["vrels"], r["ves"] 
        ps = P.sample_ellipse_interior(n_samples, rng)
        vs = vrels

        dcpa_vals = np.empty(n_samples)
        tcpa_vals = np.empty(n_samples)
        for i in range(n_samples):
            p, v = ps[i], vs[i]
            dcpa_vals[i] = DCPA(p, v)
            tcpa_vals[i] = TCPA(p, v)

        R = combined_safety_radius(ego, obstacle)

        dcpa_collision = np.abs(dcpa_vals) <= R
        tcpa_collision = tcpa_vals > 0
        vo_conditions_collision = dcpa_collision & tcpa_collision  
        vo_collision = point_in_cone(og_VO.cone, obstacle.vel, ves).sum()
        uncertainty_vo_collision = point_in_inflated_cone(VO_with_velocity_uncertainty, obstacle.vel, ves).sum()

        dcpa_prob = np.sum(dcpa_collision) / n_samples
        tcpa_prob = np.sum(tcpa_collision) / n_samples
        vo_conditions_prob = np.sum(vo_conditions_collision) / n_samples
        vo_prob = vo_collision / n_samples
        uncertainty_vo_prob = uncertainty_vo_collision / n_samples

        r.update(
            dcpa_vals=dcpa_vals, dcpa_collision=dcpa_collision, dcpa_collision_prob=dcpa_prob,
            tcpa_vals=tcpa_vals, tcpa_collision=tcpa_collision, tcpa_collision_prob=tcpa_prob,
            vo_conditions_collision=vo_conditions_collision, vo_conditions_prob=vo_conditions_prob,
            vo_collision=vo_collision, vo_prob=vo_prob,
            uncertainty_vo_collision=uncertainty_vo_collision, uncertainty_vo_prob=uncertainty_vo_prob
        )

        print(f"{name} encounter:")
        print("  DCPA prob:", dcpa_prob)
        print("  TCPA prob:", tcpa_prob)
        print("  VO conditions prob:", vo_conditions_prob)
        print("  VO collisions:", vo_prob)
        print("  Uncertainty VO collisions:", uncertainty_vo_prob)
        print("##################################################################")