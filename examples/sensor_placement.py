"""
Optimal sensor placement example.

Demonstrates D-optimal greedy placement, geometry weight mapping,
and information gain analysis for location vs tomography.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dasbb import (
    JointInversion, InversionConfig, generate_synthetic_test,
    OptimalDesign, InformationGain, AdaptiveWeighting,
    DesignDiagnostics,
)

synth = generate_synthetic_test(n_das_channels=200, n_bb_stations=15)
vm, true_src = synth['velocity_model'], synth['true_source']

# Baseline location
inv = JointInversion(vm, InversionConfig(max_iter_location=15))
result = inv.locate_event(
    synth['das_picks'], synth['bb_picks'],
    true_src + [1, -0.5, 1.5],
)
F_total = result['F_das'] + result['F_bb']

# --- Greedy sensor addition ---
od = OptimalDesign(vm)
rng = np.random.RandomState(42)
candidates = np.column_stack([
    1 + 27 * rng.rand(80), 1 + 27 * rng.rand(80), np.zeros(80)
])

print("Adding 5 optimal sensors (D-optimal)...")
greedy = od.greedy_optimal_placement(
    target_sources=np.array([true_src]),
    candidate_positions=candidates, existing_F=F_total,
    n_to_add=5, sigma=0.05, criterion='D',
)
print(f"  Criterion: {greedy['criterion_history'][0]:.2f} → {greedy['criterion_history'][-1]:.2f}")
print(f"  Marginal gains: {[f'{g:.3f}' for g in greedy['marginal_gains']]}")

# --- Per-station information gain ---
ig = InformationGain(vm)
loc_gains = ig.location_information_gain_per_station(
    true_src, synth['bb_xyz'], ['P'] * len(synth['bb_xyz']),
    synth['bb_picks'].uncertainties, F_base=result['F_das'],
)
print(f"\nPer-station location info gain: max={loc_gains.max():.4f}, min={loc_gains.min():.4f}")

# --- Geometry weight mapping ---
weights = od.compute_geometry_weights(
    true_src, synth['bb_xyz'], 'P', synth['bb_picks'].uncertainties,
)
print(f"Geometry weights: range [{weights.min():.2f}, {weights.max():.2f}]")

# --- Plot ---
fig = DesignDiagnostics.plot_greedy_placement(
    candidates, greedy['selected_indices'],
    greedy['criterion_history'], greedy['marginal_gains'],
    existing_receivers=synth['bb_xyz'], das_xyz=synth['das_xyz'],
    source_xyz=true_src,
)
fig.savefig('sensor_placement.png', dpi=150, bbox_inches='tight')
print("\nSaved: sensor_placement.png")
