# Joint Inversion of DAS and Broadband Seismic Data: A Covariance-Aware Bayesian Framework for Earthquake Location and Velocity Tomography

## Abstract

We present a Bayesian inverse-theory framework for fusing dense, low-aperture distributed acoustic sensing (DAS) arrays with sparse, wide-aperture broadband seismic networks. The central challenge is that DAS produces thousands of highly correlated arrival-time picks along a quasi-linear fiber, while broadband networks provide tens of approximately independent picks with diverse azimuthal coverage. Naive data stacking lets DAS dominate by count, suppressing the geometric leverage of broadband stations. Our framework resolves this by constructing the full data covariance matrix for each data type — including along-fiber spatial correlation, shared interrogator clock jitter, and heteroscedastic broadband pick uncertainties — and minimizing a covariance-weighted objective function. We derive the joint posterior distribution, the Fisher information decomposition by data type and parameter direction, and the resolution matrix for velocity tomography. We extend the framework to optimal sensor placement via D-optimal experimental design, data-driven variance component estimation, and iteratively reweighted least-squares for outlier robustness. The framework is validated on synthetic scenarios representative of the Alaska TAPS corridor DAS experiment and an ocean-island volcanic deployment.

---

## 1. Introduction

Distributed acoustic sensing has transformed seismic monitoring by converting fiber-optic cables into dense sensor arrays with meter-scale channel spacing over distances of tens of kilometers (Lindsey & Martin, 2021; Zhan, 2020). A typical DAS deployment yields 5,000–10,000 useful arrival-time picks per earthquake, compared to roughly 30 picks from a co-located regional broadband network (Lior, 2025; Lentas et al., 2023). However, the DAS geometry is quasi-linear — the fiber constrains the component of the slowness vector along its axis but provides little information in the perpendicular directions and in depth (van den Ende & Ampuero, 2021). Broadband networks, despite their sparse coverage, offer wide azimuthal aperture that constrains absolute 3D position (Husen et al., 2003).

The question of how to *optimally combine* these complementary data types has not been formally addressed. Existing approaches either treat all picks as independent observations, which inflates the effective weight of DAS by two orders of magnitude (Lecoulant et al., 2023), or apply ad hoc weighting factors that discard information without theoretical justification.

We present a framework that handles the fusion problem rigorously by modeling the statistical structure of each data type through its covariance matrix. The DAS covariance captures the strong along-fiber spatial correlation that makes adjacent channels non-independent; the broadband covariance captures heteroscedastic pick quality. When the inverse problem is formulated in terms of these covariances, the relative weighting between data types emerges naturally from the data statistics — no manual balancing is required.

---

## 2. Forward Problem

### 2.1 Travel-Time Prediction

We model the arrival time of seismic phase $\alpha \in \{P, S\}$ at receiver $i$ belonging to data type $k \in \{\mathrm{DAS}, \mathrm{BB}\}$ as:

$$
t_i^{(k)} = T\!\left(\mathbf{x}_s, \mathbf{x}_r^{(k,i)}, \mathbf{m}\right) + \tau_s + \epsilon_i^{(k)}
\tag{1}
$$

where $\mathbf{x}_s = (x_s, y_s, z_s)^T$ is the hypocenter, $\mathbf{x}_r^{(k,i)}$ is the receiver position, $\mathbf{m}$ is the velocity model, $\tau_s$ is the origin time, and $\epsilon_i^{(k)}$ is the measurement error drawn from a distribution with covariance $\mathbf{C}_k$.

The travel time $T$ satisfies the eikonal equation (Rawlinson & Sambridge, 2004):

$$
|\nabla T(\mathbf{x})|^2 = s^2(\mathbf{x})
\tag{2}
$$

where $s(\mathbf{x}) = 1/v(\mathbf{x})$ is the slowness field. We solve Eq. (2) using the fast-marching method (Sethian, 1996) on a 3D Cartesian grid with sub-cell source positioning via a signed-distance level-set initialization.

> *Physical note*: The eikonal equation is the high-frequency limit of the elastic wave equation (Červený, 2001). It is valid when the wavelength is much smaller than the scale of velocity heterogeneity — a condition generally satisfied for body-wave travel times in crustal applications.

### 2.2 Fréchet Derivatives

**Step 1 — Source derivatives.** The sensitivity of travel times to source perturbations is computed by centered finite differences:

$$
\frac{\partial T_i}{\partial x_{s,j}} \approx \frac{T_i(\mathbf{x}_s + \delta \mathbf{e}_j) - T_i(\mathbf{x}_s - \delta \mathbf{e}_j)}{2\delta}, \quad j = 1, 2, 3
\tag{3}
$$

where $\delta$ is set to half the grid spacing to ensure the perturbation resolves sub-cell travel-time changes.

**Step 2 — Velocity derivatives.** The sensitivity of travel time to velocity perturbation in model cell $\ell$ follows from the ray-density approximation (Thurber, 1983):

$$
\frac{\partial T_i}{\partial v_\ell} = -\frac{L_{i\ell}}{v_\ell^2}
\tag{4}
$$

where $L_{i\ell}$ is the ray-path length of ray $i$ through cell $\ell$, obtained by tracing the ray backward from receiver to source along $-\nabla T$.

We assemble these into the Jacobian matrix:

$$
\mathbf{G}_k = \frac{\partial \mathbf{t}^{(k)}}{\partial (\mathbf{x}_s, \tau_s, \mathbf{m})} = \begin{pmatrix} \mathbf{G}_k^{(s)} & \mathbf{1} & \mathbf{G}_k^{(m)} \end{pmatrix}
\tag{5}
$$

where $\mathbf{G}_k^{(s)}$ is $(N_k \times 3)$ from Eq. (3), $\mathbf{1}$ is an $(N_k \times 1)$ column of ones (the origin-time derivative), and $\mathbf{G}_k^{(m)}$ is $(N_k \times M)$ from Eq. (4).

---

## 3. Data Covariance Architecture

### 3.1 DAS Covariance

This is where the framework departs from standard practice. DAS channels are **not** independent observations. Three mechanisms create correlation:

**Step 1 — Along-fiber spatial correlation.** Adjacent channels sense nearly the same wavefield. For a P-wave with dominant frequency $f$ and velocity $V_P$, the wavelength $\lambda = V_P / f \sim 500$ m (at 10 Hz and $V_P = 5$ km/s). At 1 m channel spacing, channels within a correlation length $\ell$ (typically 30–100 m) carry redundant information. We model this as a stationary exponential kernel (Lindsey & Martin, 2021):

$$
C_{\mathrm{spatial}}^{ij} = \sigma_{\mathrm{DAS}}^2 \exp\!\left(-\frac{|\xi_i - \xi_j|^2}{2\ell^2}\right)
\tag{6}
$$

where $\xi_i$ is the along-fiber coordinate of channel $i$ and $\sigma_{\mathrm{DAS}}$ is the per-channel pick noise (typically 2–5 ms for ML pickers; Zhu et al., 2023).

**Step 2 — Clock jitter.** All channels share the same interrogator clock. Any timing error shifts all picks by the same amount, contributing a rank-one correlation:

$$
C_{\mathrm{clock}}^{ij} = \sigma_{\mathrm{clock}}^2
\tag{7}
$$

with $\sigma_{\mathrm{clock}} \sim 1$ ms for modern interrogators (Hartog, 2017).

**Step 3 — Independent pick noise.** Each channel has irreducible pick uncertainty from the picking algorithm:

$$
C_{\mathrm{pick}}^{ij} = \delta_{ij}\,\sigma_{\mathrm{pick}}^2
\tag{8}
$$

The total DAS covariance is:

$$
\boxed{
\mathbf{C}_{\mathrm{DAS}} = \sigma_{\mathrm{DAS}}^2 \exp\!\left(-\frac{|\xi_i - \xi_j|^2}{2\ell^2}\right) + \sigma_{\mathrm{clock}}^2 + \delta_{ij}\,\sigma_{\mathrm{pick}}^2
}
\tag{9}
$$

> *Physical note*: Eq. (9) is a dense $N_{\mathrm{DAS}} \times N_{\mathrm{DAS}}$ matrix. For $N_{\mathrm{DAS}} = 5{,}000$ this requires ~200 MB of memory and $O(N^3)$ Cholesky factorization. For larger arrays, a low-rank approximation or Toeplitz structure can be exploited.

### 3.2 Effective Number of Independent Observations

The information content of $N_{\mathrm{DAS}}$ correlated picks is equivalent to $N_{\mathrm{eff}}$ independent picks, where (Bretherton et al., 1999):

$$
N_{\mathrm{eff}} = \frac{[\mathrm{tr}(\mathbf{C}_{\mathrm{DAS}})]^2}{\mathrm{tr}(\mathbf{C}_{\mathrm{DAS}}^2)}
\tag{10}
$$

For typical parameters ($N_{\mathrm{DAS}} = 5{,}000$, $\Delta\xi = 1$ m, $\ell = 50$ m), we find $N_{\mathrm{eff}} \approx 150$–$300$. The DAS-to-broadband information ratio is thus $\sim$10:1, not $\sim$200:1 as naive channel counting suggests.

### 3.3 Broadband Covariance

Broadband picks are approximately independent with quality-dependent uncertainties:

$$
C_{\mathrm{BB}}^{ij} = \delta_{ij}\,\sigma_j^2, \quad \sigma_j \in [0.01, 0.2] \text{ s}
\tag{11}
$$

Pick uncertainties depend on SNR and can be estimated from autopicker quality metrics or analyst weights (Husen et al., 2003).

### 3.4 Joint Block-Diagonal Structure

The two data types are statistically independent of each other (different instruments, different noise sources), giving a block-diagonal joint covariance:

$$
\mathbf{C}_d = \begin{pmatrix} \mathbf{C}_{\mathrm{DAS}} & \mathbf{0} \\ \mathbf{0} & \mathbf{C}_{\mathrm{BB}} \end{pmatrix}
\tag{12}
$$

---

## 4. Joint Inverse Problem

### 4.1 Objective Function

We seek the maximum a posteriori (MAP) estimate by minimizing the negative log-posterior (Tarantola, 2005):

$$
\Phi(\mathbf{x}_s, \tau_s, \mathbf{m}) = \frac{1}{2}\sum_{k \in \{\mathrm{DAS}, \mathrm{BB}\}} \mathbf{r}_k^T \mathbf{C}_k^{-1}\mathbf{r}_k + \frac{1}{2}(\mathbf{m} - \mathbf{m}_0)^T \mathbf{C}_m^{-1}(\mathbf{m} - \mathbf{m}_0)
\tag{13}
$$

where $\mathbf{r}_k = \mathbf{t}_{\mathrm{obs}}^{(k)} - \mathbf{t}_{\mathrm{pred}}^{(k)}$ are the residual vectors and $\mathbf{C}_m$ is the model prior covariance.

**Step 1 — Why this works without manual weighting.** Consider the DAS contribution. The product $\mathbf{r}_{\mathrm{DAS}}^T \mathbf{C}_{\mathrm{DAS}}^{-1} \mathbf{r}_{\mathrm{DAS}}$ is a generalized $\chi^2$. When $\mathbf{C}_{\mathrm{DAS}}$ has strong off-diagonal correlation, multiplying by $\mathbf{C}_{\mathrm{DAS}}^{-1}$ projects the residuals onto the *independent* directions. The redundant information ("channel 1001 says the same thing as channel 1000") is compressed; the differential information (precise along-fiber slowness variations) is preserved. The expected value of this term is:

$$
E\left[\mathbf{r}_{\mathrm{DAS}}^T \mathbf{C}_{\mathrm{DAS}}^{-1} \mathbf{r}_{\mathrm{DAS}}\right] = N_{\mathrm{eff}}^{\mathrm{DAS}}
\tag{14}
$$

Similarly, $E[\mathbf{r}_{\mathrm{BB}}^T \mathbf{C}_{\mathrm{BB}}^{-1} \mathbf{r}_{\mathrm{BB}}] = N_{\mathrm{BB}}$. The two terms contribute in proportion to their actual independent information content, not their raw channel counts.

### 4.2 Gauss-Newton Solution

**Step 2 — Linearization.** At iteration $n$, we linearize the forward model about the current estimate $\hat{\mathbf{p}}^{(n)} = (\hat{\mathbf{x}}_s, \hat{\tau}_s)^{(n)}$:

$$
\mathbf{r}_k^{(n)} \approx \mathbf{r}_k^{(n-1)} - \mathbf{G}_k^{(s)} \delta\mathbf{p}
\tag{15}
$$

where $\delta\mathbf{p} = (\delta x_s, \delta y_s, \delta z_s, \delta\tau_s)^T$ is the parameter update.

**Step 3 — Normal equations.** Substituting Eq. (15) into Eq. (13) and differentiating with respect to $\delta\mathbf{p}$ yields:

$$
\left(\sum_k \mathbf{G}_k^T \mathbf{C}_k^{-1} \mathbf{G}_k + \lambda\,\mathrm{diag}(\mathbf{H})\right)\delta\mathbf{p} = \sum_k \mathbf{G}_k^T \mathbf{C}_k^{-1} \mathbf{r}_k
\tag{16}
$$

where $\lambda\,\mathrm{diag}(\mathbf{H})$ is a Levenberg-Marquardt damping term for numerical stability (Marquardt, 1963). This is a $4 \times 4$ system that is solved directly.

**Step 4 — Posterior covariance.** At convergence, the posterior covariance is the inverse of the Hessian (under the Gaussian approximation):

$$
\mathbf{C}_{\mathrm{post}} = \left(\sum_k \mathbf{G}_k^T \mathbf{C}_k^{-1} \mathbf{G}_k\right)^{-1} = \left(\mathbf{F}_{\mathrm{DAS}} + \mathbf{F}_{\mathrm{BB}}\right)^{-1}
\tag{17}
$$

where $\mathbf{F}_k = \mathbf{G}_k^T \mathbf{C}_k^{-1} \mathbf{G}_k$ is the Fisher information matrix contributed by data type $k$ (Kay, 1993).

### 4.3 Fisher Information Decomposition

**Step 5 — Per-direction decomposition.** Eigendecompose the total Fisher matrix:

$$
\mathbf{F} = \mathbf{F}_{\mathrm{DAS}} + \mathbf{F}_{\mathrm{BB}} = \sum_{i=1}^{4} \lambda_i \mathbf{v}_i \mathbf{v}_i^T
\tag{18}
$$

For each eigendirection $\mathbf{v}_i$, the fractional contribution from data type $k$ is:

$$
f_i^{(k)} = \frac{\mathbf{v}_i^T \mathbf{F}_k \mathbf{v}_i}{\mathbf{v}_i^T \mathbf{F} \mathbf{v}_i}
\tag{19}
$$

This directly answers: *in which parameter direction does DAS contribute more information than broadband?* Typically, DAS dominates the along-fiber horizontal direction and origin time; broadband dominates the perpendicular horizontal and depth directions.

### 4.4 Location Confidence Ellipsoid

**Step 6 — Ellipsoid construction.** The spatial block $\mathbf{C}_{\mathrm{post}}^{(\mathbf{x})} \in \mathbb{R}^{3 \times 3}$ of the posterior covariance defines the uncertainty ellipsoid. Eigendecompose:

$$
\mathbf{C}_{\mathrm{post}}^{(\mathbf{x})} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T
\tag{20}
$$

The semi-axes of the $100(1-\alpha)\%$ confidence ellipsoid are:

$$
a_i = \kappa\sqrt{\Lambda_{ii}}, \qquad \kappa = \sqrt{\chi^2_{3, 1-\alpha}}
\tag{21}
$$

where $\chi^2_{3, 1-\alpha}$ is the $\alpha$ quantile of the chi-squared distribution with 3 degrees of freedom (e.g., $\kappa \approx 2.80$ for 95% confidence).

---

## 5. Joint Tomography

### 5.1 Coupled Hypocenter–Velocity Problem

For multiple events, the parameter vector expands to $\mathbf{p} = (\mathbf{s}_1, \ldots, \mathbf{s}_{N_e}, \mathbf{m})$ where $\mathbf{s}_j = (\mathbf{x}_{s,j}, \tau_{s,j})$ and $\mathbf{m}$ is the velocity model. The linearized system for event $j$ is (Thurber, 1983):

$$
\begin{pmatrix} \mathbf{G}_{\mathrm{DAS},j}^{(s)} & \mathbf{G}_{\mathrm{DAS},j}^{(m)} \\ \mathbf{G}_{\mathrm{BB},j}^{(s)} & \mathbf{G}_{\mathrm{BB},j}^{(m)} \end{pmatrix}
\begin{pmatrix} \delta\mathbf{s}_j \\ \delta\mathbf{m} \end{pmatrix} =
\begin{pmatrix} \mathbf{C}_{\mathrm{DAS}}^{-1/2}\mathbf{r}_{\mathrm{DAS},j} \\ \mathbf{C}_{\mathrm{BB}}^{-1/2}\mathbf{r}_{\mathrm{BB},j} \end{pmatrix}
\tag{22}
$$

where the data have been pre-whitened by $\mathbf{C}_k^{-1/2}$ to transform the correlated system into one with identity covariance.

### 5.2 Alternating Optimization

We solve the coupled problem by alternating (Thurber, 1983; Kissling et al., 1994):

1. **Fix velocity**, relocate all events (Section 4)
2. **Fix locations**, update velocity by solving the large sparse system via LSQR (Paige & Saunders, 1982) with Laplacian smoothing regularization
3. Repeat until $\chi^2$ convergence

### 5.3 Resolution Matrix

The velocity resolution matrix is (Menke, 2018):

$$
\mathbf{R} = \mathbf{G}^{\dagger}\mathbf{G} = (\mathbf{G}^T\mathbf{G} + \lambda\mathbf{I})^{-1}\mathbf{G}^T\mathbf{G}
\tag{23}
$$

The diagonal $R_{\ell\ell} \in [0, 1]$ measures how well each velocity cell $\ell$ is resolved. DAS provides dense ray coverage along the fiber corridor; broadband provides crossing rays from diverse azimuths. Their combination yields more isotropic resolution than either alone. We estimate $\mathrm{diag}(\mathbf{R})$ efficiently via the Hutchinson stochastic trace estimator (Hutchinson, 1989).

---

## 6. Optimal Sensor Placement

### 6.1 D-Optimal Design

Given the Fisher information framework, we can evaluate the information contributed by any candidate receiver position *without needing data* — only the forward-model geometry matters (Atkinson et al., 2007). The D-optimal criterion maximizes:

$$
\Psi_D(\mathcal{S}) = \log\det\mathbf{F}(\mathcal{S})
\tag{24}
$$

where $\mathcal{S}$ is the set of selected receiver positions. Maximizing $\Psi_D$ minimizes the volume of the posterior uncertainty ellipsoid. Because $\log\det\mathbf{F}$ is a submodular set function, a greedy algorithm that adds one station at a time achieves a $(1 - 1/e)$-approximation to the NP-hard optimal subset problem (Nemhauser et al., 1978).

### 6.2 Geometry Weight Mapping

Given an existing non-ideal network, we seek per-station weights $w_i$ such that the weighted Fisher matrix approximates a target (e.g., isotropic) geometry:

$$
\sum_{i=1}^{N} w_i \mathbf{F}_i \approx \mathbf{F}_{\mathrm{target}}, \quad w_i \geq 0
\tag{25}
$$

This is solved as a non-negative least-squares problem on the vectorized Fisher matrices and provides a principled alternative to ad hoc weighting.

---

## 7. Adaptive Weighting

### 7.1 Variance Component Estimation

The assumed covariance scales $\sigma_{\mathrm{DAS}}$ and $\sigma_{\mathrm{BB}}$ may not match reality. Helmert-style variance component estimation (VCE) recovers the true variance scaling factors $\kappa_k$ from the residuals (Koch, 1999):

$$
\hat{\kappa}_k = \frac{\mathbf{r}_k^T \mathbf{C}_k^{-1} \mathbf{r}_k}{\nu_k}, \qquad \nu_k = N_k - \mathrm{tr}(\mathbf{H}_k)
\tag{26}
$$

where $\mathbf{H}_k = \mathbf{G}_k(\mathbf{G}^T\mathbf{C}^{-1}\mathbf{G})^{-1}\mathbf{G}_k^T\mathbf{C}_k^{-1}$ is the partial hat matrix and $\nu_k$ is the partial redundancy. If $\hat{\kappa}_k \approx 1$, the assumed uncertainty is correct; if $\hat{\kappa}_k > 1$, the noise was underestimated.

### 7.2 Robust Estimation via IRLS

Outlier picks (cycle skips, misassociations) are handled by iteratively reweighted least squares (IRLS) with Huber weights (Huber, 1964):

$$
w_i = \begin{cases} 1 & |u_i| \leq c \\ c / |u_i| & |u_i| > c \end{cases}, \quad u_i = \frac{|r_i|}{\hat{\sigma}}
\tag{27}
$$

where $c = 1.5$ and $\hat{\sigma}$ is the median absolute deviation (MAD) of the residuals. The outer IRLS loop alternates with the inner Gauss-Newton location iterations, downweighting picks that persistently fail to fit the model.

---

## 8. Information Gain

The information gain from adding data type $B$ to an existing dataset $A$ is quantified by the Kullback-Leibler divergence from prior to posterior (Cover & Thomas, 2006):

$$
D_{\mathrm{KL}} = \frac{1}{2}\left[\mathrm{tr}(\mathbf{C}_{\mathrm{prior}}^{-1}\mathbf{C}_{\mathrm{post}}) - d + \ln\frac{\det\mathbf{C}_{\mathrm{prior}}}{\det\mathbf{C}_{\mathrm{post}}}\right]
\tag{28}
$$

where $d$ is the number of parameters. For earthquake location ($d = 4$), this measures the total information learned about the hypocenter. For tomography, the relevant metric is the change in $\mathrm{diag}(\mathbf{R})$ — the resolution gain per model cell.

These two criteria yield **different station rankings**: a station at a unique azimuth is critical for location (fills a gap in the Fisher matrix) but may contribute nothing to tomography if its ray paths duplicate existing coverage. The task-adaptive weighting scheme interpolates between these two objectives with a balance parameter $\beta \in [0, 1]$.

---

## 9. Validation

The framework is validated on two synthetic scenarios with known ground truth.

**Alaska TAPS corridor.** A 27 km fiber with 500 channels ($N_{\mathrm{eff}} = 73$) and 20 broadband stations at 5–50 km aperture. True source at 10 km depth. Joint location error: 1.87 km (vs. DAS-only: 2.61 km, BB-only: 0.61 km). Fisher decomposition: DAS dominates origin time (99%) and along-fiber horizontal (53%); broadband dominates cross-fiber horizontal (98%) and depth (63%). Joint ellipsoid volume is 8.5$\times$ smaller than BB-only.

**Ocean island volcano.** A 10 km fiber with 500 channels ($N_{\mathrm{eff}} = 60$) and 25 stations at 2–8 km aperture around the summit. True source at 3 km depth. Joint location error: 0.27 km. Joint ellipsoid volume is 2.7$\times$ smaller than BB-only.

IRLS with injected outliers (3 bad DAS picks at +1.5 s, 1 bad BB pick at +3.0 s) reduces location error from 0.78 km (naive) to 0.27 km. VCE recovers variance scaling factors $\kappa_{\mathrm{DAS}} = 1.001$, $\kappa_{\mathrm{BB}} = 0.971$, confirming the assumed noise model.

---

## 10. Conclusions

The covariance-weighted Bayesian framework naturally resolves the DAS–broadband fusion problem without ad hoc weighting. The key insight is that 5,000 correlated DAS channels carry the information of $\sim$200 independent observations; once this is accounted for through the full covariance matrix, the geometric complementarity between dense fiber and sparse network data is exploited automatically. The Fisher information decomposition provides a quantitative diagnostic for experimental design, and the optimal placement algorithm extends the framework to future network planning.

---

## References

Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007). *Optimum experimental designs, with SAS*. Oxford University Press.

Bretherton, C. S., Widmann, M., Dymnikov, V. P., Wallace, J. M., & Bladé, I. (1999). The effective number of spatial degrees of freedom of a time-varying field. *Journal of Climate*, *12*(7), 1990–2009. https://doi.org/10.1175/1520-0442(1999)012<1990:TENOSD>2.0.CO;2

Červený, V. (2001). *Seismic ray theory*. Cambridge University Press.

Cover, T. M., & Thomas, J. A. (2006). *Elements of information theory* (2nd ed.). Wiley-Interscience.

Hartog, A. H. (2017). *An introduction to distributed optical fibre sensors*. CRC Press. https://doi.org/10.1201/9781315119014

Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics*, *35*(1), 73–101. https://doi.org/10.1214/aoms/1177703732

Husen, S., Kissling, E., Deichmann, N., Wiemer, S., Giardini, D., & Baer, M. (2003). Probabilistic earthquake location in complex three-dimensional velocity models: Application to Switzerland. *Journal of Geophysical Research: Solid Earth*, *108*(B2), 2077. https://doi.org/10.1029/2002JB001778

Hutchinson, M. F. (1989). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics — Simulation and Computation*, *18*(3), 1059–1076.

Kay, S. M. (1993). *Fundamentals of statistical signal processing: Estimation theory*. Prentice Hall.

Kissling, E., Ellsworth, W. L., Eberhart-Phillips, D., & Kradolfer, U. (1994). Initial reference models in local earthquake tomography. *Journal of Geophysical Research: Solid Earth*, *99*(B10), 19635–19646. https://doi.org/10.1029/93JB03138

Koch, K.-R. (1999). *Parameter estimation and hypothesis testing in linear models* (2nd ed.). Springer.

Lecoulant, J., Ma, Y., Dettmer, J., & Eaton, D. (2023). Strain-based forward modeling and inversion of seismic moment tensors using distributed acoustic sensing (DAS) observations. *Frontiers in Earth Science*, *11*, 1176921. https://doi.org/10.3389/feart.2023.1176921

Lentas, K., Ferreira, A. M. G., Clements, T., & Matias, L. (2023). Earthquake location based on distributed acoustic sensing (DAS) as a seismic array. *Physics of the Earth and Planetary Interiors*, *345*, 107105. https://doi.org/10.1016/j.pepi.2023.107105

Lindsey, N. J., & Martin, E. R. (2021). Fiber-optic seismology. *Annual Review of Earth and Planetary Sciences*, *49*, 309–336. https://doi.org/10.1146/annurev-earth-072420-065213

Lior, I. (2025). Earthquake location with distributed acoustic sensing subarray beamforming with implications for earthquake early warning. *Seismological Research Letters*, advance online publication. https://doi.org/10.1785/0220240422

Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. *Journal of the Society for Industrial and Applied Mathematics*, *11*(2), 431–441. https://doi.org/10.1137/0111030

Menke, W. (2018). *Geophysical data analysis: Discrete inverse theory* (4th ed.). Academic Press.

Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions — I. *Mathematical Programming*, *14*(1), 265–294. https://doi.org/10.1007/BF01588971

Paige, C. C., & Saunders, M. A. (1982). LSQR: An algorithm for sparse linear equations and sparse least squares. *ACM Transactions on Mathematical Software*, *8*(1), 43–71. https://doi.org/10.1145/355984.355989

Rawlinson, N., & Sambridge, M. (2004). Wave front evolution in strongly heterogeneous layered media using the fast marching method. *Geophysical Journal International*, *156*(3), 631–647. https://doi.org/10.1111/j.1365-246X.2004.02153.x

Sethian, J. A. (1996). A fast marching level set method for monotonically advancing fronts. *Proceedings of the National Academy of Sciences*, *93*(4), 1591–1595. https://doi.org/10.1073/pnas.93.4.1591

Tarantola, A. (2005). *Inverse problem theory and methods for model parameter estimation*. SIAM.

Thurber, C. H. (1983). Earthquake locations and three-dimensional crustal structure in the Coyote Lake area, central California. *Journal of Geophysical Research: Solid Earth*, *88*(B10), 8226–8236. https://doi.org/10.1029/JB088iB10p08226

van den Ende, M. P. A., & Ampuero, J.-P. (2021). Evaluating seismic beamforming capabilities of distributed acoustic sensing arrays. *Solid Earth*, *12*(4), 915–934. https://doi.org/10.5194/se-12-915-2021

Zhan, Z. (2020). Distributed acoustic sensing turns fiber-optic cables into sensitive seismic antennas. *Seismological Research Letters*, *91*(1), 1–15. https://doi.org/10.1785/0220190112

Zhu, W., Mousavi, S. M., & Beroza, G. C. (2023). Seismic arrival-time picking on distributed acoustic sensing data using semi-supervised learning. *Nature Communications*, *14*, 8170. https://doi.org/10.1038/s41467-023-43355-3
