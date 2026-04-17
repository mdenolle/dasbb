"""
Quick start example: locate an event with joint DAS + broadband.

Demonstrates the core workflow:
  1. Generate synthetic data
  2. Run joint location
  3. Compare joint vs single-data-type results
  4. Inspect Fisher information decomposition
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dasbb import (
    JointInversion, InversionConfig, generate_synthetic_test,
    InversionDiagnostics,
)

# --- Generate data ---
synth = generate_synthetic_test(n_das_channels=300, n_bb_stations=20)
vm = synth['velocity_model']
true_src = synth['true_source']

print(f"True source: {true_src}")
print(f"DAS N_eff: {synth['das_picks'].effective_n():.0f} / {synth['das_picks'].n_picks}")

# --- Inversion ---
inv = JointInversion(vm, InversionConfig(max_iter_location=20))
init = true_src + np.array([1.5, -1.0, 2.0])

result_joint = inv.locate_event(synth['das_picks'], synth['bb_picks'], init)
result_das = inv.locate_event(synth['das_picks'], None, init)
result_bb = inv.locate_event(None, synth['bb_picks'], init)

for label, r in [('Joint', result_joint), ('DAS', result_das), ('BB', result_bb)]:
    err = np.linalg.norm(r['source_xyz'] - true_src)
    ell = r.get('location_ellipsoid')
    vol = f"{ell['volume_km3']:.4f}" if ell else "N/A"
    print(f"  {label:6s}: error={err:.3f} km, ellipsoid volume={vol} km³")

# --- Fisher decomposition ---
print("\nInformation decomposition:")
decomp = inv.information_decomposition(result_joint['F_das'], result_joint['F_bb'])
for name, info in decomp.items():
    print(f"  {info['dominant_parameter']:2s}: DAS {info['das_fraction']:5.1%}, BB {info['bb_fraction']:5.1%}")

# --- Plot ---
fig = InversionDiagnostics.plot_location_comparison(
    result_joint, result_das, result_bb, true_src
)
fig.savefig('quick_start_ellipsoids.png', dpi=150, bbox_inches='tight')
print("\nSaved: quick_start_ellipsoids.png")
