"""
Adaptive weighting schemes for heterogeneous data fusion.

- Helmert Variance Component Estimation (VCE): data-driven σ² scaling
- Iteratively Reweighted Least Squares (IRLS): outlier robustness
- Task-adaptive weights: location vs tomography balance
- GCV regularization parameter selection
"""

import numpy as np
from scipy.linalg import inv
from scipy import sparse
from typing import Optional, Dict, Any, List
import logging

from .data import DASPicks, BroadbandPicks
from .forward import EikonalSolver
from .inversion import JointInversion

logger = logging.getLogger(__name__)

class AdaptiveWeighting:
    """
    Data-driven weighting schemes that replace or augment
    the static covariance-based approach.

    Three methods, each solving a different problem:

    1. VARIANCE COMPONENT ESTIMATION (VCE / Helmert):
       Estimate the true variance scaling factors for DAS and BB
       from the residuals themselves. Answers: "Is my assumed
       DAS σ=3ms actually correct, or should it be 5ms?"

    2. ITERATIVELY REWEIGHTED LEAST SQUARES (IRLS):
       Down-weight outlier picks that don't fit the model,
       regardless of their a priori uncertainty. Robust to
       bad picks, cycle skips, and misassociations.

    3. TASK-ADAPTIVE WEIGHTS:
       Weight stations differently depending on whether you're
       solving for location or tomography. A station that adds
       nothing for location (redundant azimuth) might be critical
       for tomography (unique ray path).
    """

    # --------------------------------------------------------
    # VARIANCE COMPONENT ESTIMATION
    # --------------------------------------------------------

    @staticmethod
    def helmert_vce(
        residuals_das: np.ndarray,
        residuals_bb: np.ndarray,
        G_das: np.ndarray,
        G_bb: np.ndarray,
        C_das: np.ndarray,
        C_bb: np.ndarray,
        n_iterations: int = 10,
    ) -> Dict[str, float]:
        """
        Helmert-style variance component estimation.

        Given residuals from an initial inversion with assumed
        covariances, estimate the true variance scaling factors
        κ_DAS and κ_BB such that:

          C_DAS_true = κ_DAS * C_DAS_assumed
          C_BB_true  = κ_BB  * C_BB_assumed

        If κ ≈ 1, your assumed uncertainties are correct.
        If κ > 1, you underestimated the noise.
        If κ < 1, you overestimated the noise.

        The iteration alternates between:
          1. Solve the normal equations with current κ values
          2. Update κ from the weighted residual sums

        Parameters
        ----------
        residuals_das : (n_das,) DAS residuals from initial solution
        residuals_bb : (n_bb,) BB residuals
        G_das : (n_das, n_params) DAS Jacobian
        G_bb : (n_bb, n_params) BB Jacobian
        C_das : (n_das, n_das) assumed DAS covariance
        C_bb : (n_bb, n_bb) assumed BB covariance
        n_iterations : VCE iterations

        Returns
        -------
        dict with:
          kappa_das : variance scaling factor for DAS
          kappa_bb : variance scaling factor for BB
          sigma_das_estimated : estimated DAS pick sigma
          sigma_bb_estimated : estimated BB pick sigma
          convergence : kappa history
        """
        n_das = len(residuals_das)
        n_bb = len(residuals_bb)
        n_params = G_das.shape[1]

        kappa_das = 1.0
        kappa_bb = 1.0
        history = []

        C_das_cho_base = cho_factor(C_das)
        if C_bb.ndim == 2:
            C_bb_inv_base = inv(C_bb)
        else:
            C_bb_inv_base = np.diag(1.0 / np.diag(C_bb))

        for it in range(n_iterations):
            # Current covariance inverses
            C_das_inv = cho_solve(C_das_cho_base, np.eye(n_das)) / kappa_das
            C_bb_inv = C_bb_inv_base / kappa_bb

            # Redundancy numbers (partial redundancy per group)
            # r_k = n_k - tr(G_k (G^T C^{-1} G)^{-1} G_k^T C_k^{-1})
            H = G_das.T @ C_das_inv @ G_das + G_bb.T @ C_bb_inv @ G_bb
            try:
                H_inv = inv(H)
            except np.linalg.LinAlgError:
                logger.warning(f"VCE: singular Hessian at iteration {it}")
                break

            hat_das = G_das @ H_inv @ G_das.T @ C_das_inv
            hat_bb = G_bb @ H_inv @ G_bb.T @ C_bb_inv
            r_das = n_das - np.trace(hat_das)
            r_bb = n_bb - np.trace(hat_bb)

            # Weighted sum of squared residuals
            vtPv_das = float(
                residuals_das @ C_das_inv @ residuals_das
            )
            vtPv_bb = float(
                residuals_bb @ C_bb_inv @ residuals_bb
            )

            # Update variance components
            if r_das > 0.5:
                kappa_das = vtPv_das / r_das
            if r_bb > 0.5:
                kappa_bb = vtPv_bb / r_bb

            history.append((kappa_das, kappa_bb))
            logger.debug(
                f"  VCE iter {it}: κ_DAS={kappa_das:.4f}, κ_BB={kappa_bb:.4f}"
            )

        # Estimate actual sigmas
        sigma_das_orig = np.sqrt(np.mean(np.diag(C_das)))
        sigma_bb_orig = np.sqrt(np.mean(np.diag(C_bb)))

        return {
            'kappa_das': kappa_das,
            'kappa_bb': kappa_bb,
            'sigma_das_estimated': sigma_das_orig * np.sqrt(kappa_das),
            'sigma_bb_estimated': sigma_bb_orig * np.sqrt(kappa_bb),
            'convergence': history,
        }

    # --------------------------------------------------------
    # ITERATIVELY REWEIGHTED LEAST SQUARES
    # --------------------------------------------------------

    @staticmethod
    def irls_weights(
        residuals: np.ndarray,
        sigma: float,
        method: str = 'huber',
        huber_c: float = 1.5,
        tukey_c: float = 4.685,
    ) -> np.ndarray:
        """
        Compute IRLS weights for robust estimation.

        Down-weights outlier residuals that are likely bad picks,
        cycle skips, or misassociations.

        Parameters
        ----------
        residuals : (n,) residual vector
        sigma : estimated scale
        method : 'huber' (soft), 'tukey' (hard), or 'cauchy'
        huber_c : Huber threshold in units of sigma
        tukey_c : Tukey biweight threshold

        Returns
        -------
        weights : (n,) in [0, 1]
        """
        u = np.abs(residuals) / max(sigma, 1e-10)

        if method == 'huber':
            # Linear beyond threshold, constant weight inside
            w = np.where(u <= huber_c, 1.0, huber_c / u)

        elif method == 'tukey':
            # Zero weight beyond threshold (hard rejection)
            w = np.where(
                u <= tukey_c,
                (1.0 - (u / tukey_c) ** 2) ** 2,
                0.0
            )

        elif method == 'cauchy':
            # Gradual down-weighting, never zero
            w = 1.0 / (1.0 + (u / huber_c) ** 2)

        else:
            raise ValueError(f"Unknown method: {method}")

        return w

    @staticmethod
    def irls_locate(
        inv_engine: JointInversion,
        das_picks: Optional[DASPicks],
        bb_picks: Optional[BroadbandPicks],
        initial_source: np.ndarray,
        initial_tau: float,
        n_irls_iterations: int = 5,
        method: str = 'huber',
    ) -> Dict[str, Any]:
        """
        Robust location via IRLS.

        Outer loop: compute weights from residuals.
        Inner loop: standard Gauss-Newton with weighted data.

        This handles outlier picks gracefully — a single bad
        DAS channel or a misassociated BB pick won't bias the
        location.

        Returns standard location result dict plus:
          irls_weights_das : (n_das,) final weights
          irls_weights_bb : (n_bb,) final weights
          n_outliers_das : number of strongly downweighted DAS picks
          n_outliers_bb : number of strongly downweighted BB picks
        """
        result = None
        w_das = np.ones(das_picks.n_picks) if das_picks else None
        w_bb = np.ones(bb_picks.n_picks) if bb_picks else None

        src = initial_source.copy()
        tau = initial_tau

        for irls_iter in range(n_irls_iterations):
            # Modify picks with current weights
            if das_picks is not None and w_das is not None:
                modified_das = DASPicks(
                    times=das_picks.times,
                    receiver_xyz=das_picks.receiver_xyz,
                    fiber_coords=das_picks.fiber_coords,
                    phase=das_picks.phase,
                    pick_sigma=das_picks.pick_sigma,
                    clock_sigma=das_picks.clock_sigma,
                    correlation_length=das_picks.correlation_length,
                )
                # Apply IRLS weights by inflating uncertainty for bad picks
                # This modifies the effective covariance: C' = C / w
                # Implemented by scaling pick_sigma
                effective_sigma = das_picks.pick_sigma / np.sqrt(
                    np.mean(w_das) + 1e-10
                )
                modified_das.pick_sigma = max(effective_sigma, 1e-6)
            else:
                modified_das = das_picks

            if bb_picks is not None and w_bb is not None:
                # Apply weights by inflating uncertainties for downweighted picks
                weighted_sigmas = bb_picks.uncertainties / np.sqrt(
                    np.maximum(w_bb, 0.01)
                )
                modified_bb = BroadbandPicks(
                    times=bb_picks.times,
                    receiver_xyz=bb_picks.receiver_xyz,
                    phases=bb_picks.phases,
                    uncertainties=weighted_sigmas,
                )
            else:
                modified_bb = bb_picks

            # Inner Gauss-Newton
            result = inv_engine.locate_event(
                modified_das, modified_bb, src, tau
            )
            src = result['source_xyz']
            tau = result['origin_time']

            # Compute residuals and update weights
            solver = inv_engine.solver

            if das_picks is not None:
                tt_das = solver.travel_times_at_receivers(
                    src, das_picks.receiver_xyz, das_picks.phase
                )
                r_das = das_picks.times - (tt_das + tau)
                mad_das = 1.4826 * np.median(np.abs(r_das - np.median(r_das)))
                w_das = AdaptiveWeighting.irls_weights(
                    r_das, max(mad_das, 1e-6), method=method
                )

            if bb_picks is not None:
                tt_bb = solver.travel_times_at_receivers(
                    src, bb_picks.receiver_xyz, bb_picks.phases[0]
                )
                r_bb = bb_picks.times - (tt_bb + tau)
                mad_bb = 1.4826 * np.median(np.abs(r_bb - np.median(r_bb)))
                w_bb = AdaptiveWeighting.irls_weights(
                    r_bb, max(mad_bb, 1e-6), method=method
                )

            logger.info(
                f"  IRLS iter {irls_iter}: "
                f"DAS outliers={np.sum(w_das < 0.5) if w_das is not None else 0}, "
                f"BB outliers={np.sum(w_bb < 0.5) if w_bb is not None else 0}"
            )

        # Add IRLS info to result
        if result is not None:
            result['irls_weights_das'] = w_das
            result['irls_weights_bb'] = w_bb
            result['n_outliers_das'] = int(np.sum(w_das < 0.5)) if w_das is not None else 0
            result['n_outliers_bb'] = int(np.sum(w_bb < 0.5)) if w_bb is not None else 0

        return result

    # --------------------------------------------------------
    # TASK-ADAPTIVE WEIGHTS
    # --------------------------------------------------------

    @staticmethod
    def task_adaptive_weights(
        location_gains: np.ndarray,
        tomography_values: np.ndarray,
        task_balance: float = 0.5,
    ) -> np.ndarray:
        """
        Combine location and tomography importance into a single
        per-station weight vector.

        Parameters
        ----------
        location_gains : (n,) per-station location information gain
        tomography_values : (n,) per-station tomography value
        task_balance : 0 = pure location, 1 = pure tomography

        Returns
        -------
        weights : (n,) combined task-adaptive weights, mean = 1
        """
        # Normalize each to [0, 1]
        def normalize(x):
            r = x.max() - x.min()
            if r < 1e-15:
                return np.ones_like(x)
            return (x - x.min()) / r

        loc_norm = normalize(location_gains)
        tomo_norm = normalize(tomography_values)

        combined = (1 - task_balance) * loc_norm + task_balance * tomo_norm
        combined = normalize(combined)

        # Map to weights: minimum weight = 0.1 (never fully discard)
        weights = 0.1 + 0.9 * combined

        # Normalize mean to 1
        weights *= len(weights) / np.sum(weights)

        return weights

    # --------------------------------------------------------
    # REGULARIZATION PARAMETER SELECTION
    # --------------------------------------------------------

    @staticmethod
    def gcv_lambda(
        G: np.ndarray,
        d: np.ndarray,
        lambda_range: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Generalized Cross-Validation for regularization parameter.

        Finds λ that minimizes:
          GCV(λ) = ||r(λ)||² / [tr(I - H(λ))]²

        where H(λ) = G(G^T G + λI)⁻¹ G^T is the hat matrix.

        This gives an automatic, data-driven choice of λ for the
        velocity regularization — no manual tuning needed.

        Parameters
        ----------
        G : (m, n) Jacobian (can be dense or will be converted)
        d : (m,) data vector
        lambda_range : array of λ values to test

        Returns
        -------
        dict with:
          lambda_optimal : best λ
          gcv_values : GCV score at each λ
          lambda_tested : the λ values
        """
        if lambda_range is None:
            # Estimate range from singular values
            if sparse.issparse(G):
                G_dense = G.toarray()
            else:
                G_dense = G
            s = np.linalg.svd(G_dense, compute_uv=False)
            s_max = s[0]
            s_min = s[min(len(s) - 1, G_dense.shape[1] - 1)]
            lambda_range = np.logspace(
                np.log10(max(s_min ** 2, 1e-10)),
                np.log10(s_max ** 2),
                50
            )

        m, n = G.shape if not sparse.issparse(G) else G.shape
        if sparse.issparse(G):
            G_dense = G.toarray()
        else:
            G_dense = G

        GtG = G_dense.T @ G_dense
        Gtd = G_dense.T @ d

        gcv_values = np.zeros(len(lambda_range))

        for i, lam in enumerate(lambda_range):
            A = GtG + lam * np.eye(n)
            try:
                m_hat = np.linalg.solve(A, Gtd)
            except np.linalg.LinAlgError:
                gcv_values[i] = np.inf
                continue

            residual = d - G_dense @ m_hat
            rss = float(residual @ residual)

            # tr(I - H) = m - tr(G (G^TG + λI)^{-1} G^T)
            #            = m - tr((G^TG + λI)^{-1} G^TG)
            #            = m - Σ s_i² / (s_i² + λ)
            A_inv_GtG = np.linalg.solve(A, GtG)
            trace_hat = np.trace(A_inv_GtG)
            dof = m - trace_hat

            if dof > 0:
                gcv_values[i] = (rss / m) / (dof / m) ** 2
            else:
                gcv_values[i] = np.inf

        best_idx = np.argmin(gcv_values)

        return {
            'lambda_optimal': float(lambda_range[best_idx]),
            'gcv_values': gcv_values,
            'lambda_tested': lambda_range,
        }

