# dasbb — Joint DAS + Broadband Seismic Inversion

Bayesian framework for fusing dense, low-aperture DAS fiber-optic arrays with sparse, wide-aperture broadband seismic networks. Solves the earthquake location and velocity tomography problems with proper heteroscedastic, correlated data covariance.

## Background

DAS gives you thousands of picks at sub-millisecond precision along a fiber — but the geometry is quasi-linear, so it constrains one slowness component well and the others poorly. Broadband networks give wide azimuthal coverage with far fewer observations and larger uncertainties.

Naively stacking them lets DAS dominate by count. This framework solves the problem correctly: the DAS covariance matrix (exponential spatial correlation + shared interrogator clock jitter) reduces thousands of channels to ~100–200 effective independent observations. The covariance-weighted objective function then balances the two data types automatically — no ad hoc reweighting required.

## Methodological summary

### Data covariance

DAS arrival-time picks are modelled with a three-component covariance structure:

$$C_{ij}^{\text{DAS}} = \sigma_\text{pick}^2 \exp\!\left(-\frac{|\xi_i - \xi_j|^2}{2\ell^2}\right) + \sigma_\text{clock}^2 + \delta_{ij}\,\sigma_\text{pick}^2$$

where $\xi$ is the along-fiber coordinate, $\ell$ the correlation length (~50 m), and $\sigma_\text{clock}$ the shared interrogator clock jitter. The effective number of independent observations is

$$N_\text{eff} = \frac{[\text{tr}(\mathbf{C})]^2}{\text{tr}(\mathbf{C}^2)}$$

which for a 500-channel deployment with $\ell = 50\,\text{m}$ and 1 m channel spacing typically gives $N_\text{eff} \approx 60$–$200$. Broadband picks are treated as independent with heteroscedastic per-phase uncertainties.

### Forward model

Travel times are computed by solving the eikonal equation

$$|\nabla T(\mathbf{x})|^2 = s(\mathbf{x})^2$$

via the fast-marching method (scikit-fmm) on a 3-D Cartesian grid. Sub-cell source positioning uses a signed-distance level-set initialisation; receivers are interpolated trilinearly. Source Fréchet derivatives are computed by centred finite differences; velocity Fréchet kernels follow the ray-tracing approximation.

### Inversion

The joint objective function is:

$$\Phi(\mathbf{m}) = \frac{1}{2}\mathbf{r}_\text{DAS}^\top \mathbf{C}_\text{DAS}^{-1} \mathbf{r}_\text{DAS} + \frac{1}{2}\mathbf{r}_\text{BB}^\top \mathbf{C}_\text{BB}^{-1} \mathbf{r}_\text{BB} + \frac{\lambda}{2}\|\mathbf{L}\mathbf{m}\|^2$$

minimised iteratively by damped Gauss-Newton. The Fisher information matrix splits naturally into DAS and BB contributions:

$$\mathbf{F} = \mathbf{F}_\text{DAS} + \mathbf{F}_\text{BB} = \mathbf{G}_\text{DAS}^\top \mathbf{C}_\text{DAS}^{-1} \mathbf{G}_\text{DAS} + \mathbf{G}_\text{BB}^\top \mathbf{C}_\text{BB}^{-1} \mathbf{G}_\text{BB}$$

The posterior covariance $\boldsymbol{\Sigma} = \mathbf{F}^{-1}$ gives a 95% error ellipsoid via the eigendecomposition of $\mathbf{F}$. Information decomposition shows which data type controls each parameter direction.

### Robust weighting and adaptive schemes

Three complementary strategies handle real-data imperfections:

- **Helmert VCE** — estimates actual variance scaling factors $(\hat\sigma_\text{DAS}^2,\, \hat\sigma_\text{BB}^2)$ from residuals, correcting for misspecified a priori uncertainties.
- **IRLS** — iteratively down-weights outlier picks using Huber or Tukey influence functions, robust to cycle skips and misassociations.
- **Task-adaptive weights** — stations are weighted differently for location versus tomography. A station that adds nothing for location (redundant azimuth) may be critical for tomography (unique ray path).

### Optimal sensor placement

New sensor positions are selected greedily to maximise one of the D/A/E-optimality criteria over the Fisher matrix. The marginal gain of each candidate is evaluated using the matrix-determinant lemma; location and tomography objectives are separated via task-specific information gain (KL divergence and resolution-matrix diagonal, respectively).

## Repository structure

```
dasbb/
├── dasbb/
│   ├── __init__.py          # Public API
│   ├── data.py              # DASPicks, BroadbandPicks, VelocityModel + covariance builders
│   ├── forward.py           # EikonalSolver, fast marching, Fréchet derivatives
│   ├── inversion.py         # JointInversion, InversionConfig, Fisher decomposition
│   ├── design.py            # OptimalDesign, D/A/E-optimal placement, geometry weights
│   ├── information.py       # InformationGain, KL divergence, resolution gain
│   ├── weighting.py         # AdaptiveWeighting: Helmert VCE, IRLS, GCV
│   ├── diagnostics.py       # Location ellipsoid plots, sensor ranking maps
│   └── synthetic.py         # Test scenario generators (generic, Alaska, ocean island)
├── examples/
│   ├── quick_start.py       # Joint location + Fisher decomposition
│   ├── robust_location.py   # IRLS robust inversion with injected outliers
│   ├── sensor_placement.py  # D-optimal greedy sensor design
│   └── dasbb_demo.ipynb     # Interactive notebook covering all three examples
├── tests/
│   ├── test_data.py         # Covariance structure, effective N
│   ├── test_forward.py      # Travel-time accuracy, Fréchet derivatives
│   ├── test_inversion.py    # Location, posterior, Fisher decomposition
│   ├── test_design.py       # D-optimal placement, geometry weights
│   └── test_weighting.py    # VCE, IRLS, GCV
├── pixi.toml                # Pixi environment (conda-forge + PyPI)
└── pyproject.toml           # Build metadata and optional dependencies
```

## Installation

This project is managed with [pixi](https://prefix.dev/docs/pixi/overview), which handles the conda + pip dependency mix automatically.

### With pixi (recommended)

```bash
# Install pixi if you don't have it
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and install
git clone https://github.com/marinedenolle/dasbb.git
cd dasbb
pixi install          # default environment (core deps + dev tools)
pixi run test         # verify the installation (33 tests)
```

Optional feature environments:

```bash
pixi install -e mcmc  # adds PyMC + emcee for MCMC sampling
pixi install -e io    # adds ObsPy for reading real seismic data
pixi install -e all   # all optional dependencies
```

### With pip

```bash
git clone https://github.com/marinedenolle/dasbb.git
cd dasbb
pip install -e .
# scikit-fmm is not on conda-forge; always install from PyPI:
pip install scikit-fmm
```

Optional extras:

```bash
pip install -e ".[mcmc]"   # PyMC + emcee
pip install -e ".[io]"     # ObsPy
pip install -e ".[all]"    # everything
```

### Dependencies

| Package | Source | Purpose |
|---------|--------|---------|
| numpy, scipy, matplotlib, h5py | conda-forge | numerics, I/O, plotting |
| scikit-fmm | **PyPI only** | fast-marching eikonal solver |
| pymc, emcee | conda-forge | MCMC sampling (optional) |
| obspy | conda-forge | seismic data I/O (optional) |

## Quick start

```python
import numpy as np
from dasbb import JointInversion, InversionConfig, generate_synthetic_test

synth = generate_synthetic_test(n_das_channels=300, n_bb_stations=20)

inv    = JointInversion(synth['velocity_model'], InversionConfig(max_iter_location=20))
result = inv.locate_event(
    synth['das_picks'], synth['bb_picks'],
    initial_source=synth['true_source'] + [1, -0.5, 1.5],
)

print(f"Location error: {np.linalg.norm(result['source_xyz'] - synth['true_source']):.3f} km")
print(f"95% ellipsoid volume: {result['location_ellipsoid']['volume_km3']:.4f} km³")

# Which data type constrains which direction?
decomp = inv.information_decomposition(result['F_das'], result['F_bb'])
for direction, info in decomp.items():
    print(f"  {direction}: DAS {info['das_fraction']:.0%}  BB {info['bb_fraction']:.0%}")
```

See `examples/dasbb_demo.ipynb` for an interactive walkthrough of all three workflows.

## Contributing

Contributions are welcome. Please follow these steps:

1. **Fork** the repository and create a feature branch (`git checkout -b feature/my-change`).
2. **Install** the dev environment: `pixi install` (includes pytest and pytest-cov).
3. **Write tests** for any new functionality in `tests/`. All 33 existing tests must pass.
4. **Run the test suite** before opening a PR: `pixi run test`.
5. **Keep scope narrow** — one feature or fix per PR.
6. **Open a pull request** with a clear description of what changed and why.

### Code style

- Follow PEP 8; keep line length ≤ 100 characters.
- New public functions and classes need a NumPy-style docstring.
- Avoid adding hard dependencies without discussion; prefer conda-forge packages. If a package is PyPI-only (like `scikit-fmm`), add it to `[pypi-dependencies]` in `pixi.toml`.

### Reporting issues

Please include: Python version, pixi/pip environment, a minimal reproducible example, and the full traceback.

## License

MIT

## Citation

If you use this framework in published research, please cite:
```
@software{dasbb,
  title={dasbb: Joint DAS + Broadband Bayesian Seismic Inversion},
  url={https://github.com/your-org/dasbb},
}
```
