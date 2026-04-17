"""
Robust location with IRLS and variance component estimation.

Demonstrates how adaptive weighting handles outlier picks
and automatically calibrates DAS vs broadband uncertainty balance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dasbb import (
    JointInversion, InversionConfig, generate_synthetic_test,
    AdaptiveWeighting, DASPicks, BroadbandPicks,
)

synth = generate_synthetic_test(n_das_channels=200, n_bb_stations=15)
true_src = synth['true_source']
inv = JointInversion(synth['velocity_model'], InversionConfig(max_iter_location=15))
init = true_src + np.array([1.0, -0.5, 1.5])

# --- Inject outliers ---
das_bad = DASPicks(
    times=synth['das_picks'].times.copy(),
    receiver_xyz=synth['das_picks'].receiver_xyz,
    fiber_coords=synth['das_picks'].fiber_coords,
    phase='P', pick_sigma=synth['das_picks'].pick_sigma,
    clock_sigma=synth['das_picks'].clock_sigma,
    correlation_length=synth['das_picks'].correlation_length,
)
das_bad.times[[10, 50, 100]] += 1.5  # 3 bad DAS picks

bb_bad = BroadbandPicks(
    times=synth['bb_picks'].times.copy(),
    receiver_xyz=synth['bb_picks'].receiver_xyz,
    phases=synth['bb_picks'].phases,
    uncertainties=synth['bb_picks'].uncertainties,
)
bb_bad.times[5] += 3.0  # 1 bad BB pick

# --- Compare naive vs IRLS ---
result_naive = inv.locate_event(das_bad, bb_bad, init)
result_irls = AdaptiveWeighting.irls_locate(
    inv, das_bad, bb_bad, init, synth['true_origin_time'],
    n_irls_iterations=4, method='huber',
)

err_naive = np.linalg.norm(result_naive['source_xyz'] - true_src)
err_irls = np.linalg.norm(result_irls['source_xyz'] - true_src)

print(f"Naive location error (with outliers): {err_naive:.3f} km")
print(f"IRLS location error (with outliers):  {err_irls:.3f} km")
print(f"DAS outliers detected: {result_irls['n_outliers_das']}")
print(f"BB outliers detected:  {result_irls['n_outliers_bb']}")
print(f"Improvement: {(1 - err_irls/err_naive)*100:.0f}%")
