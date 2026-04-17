"""
Visualization and diagnostic tools for all inversion products.

Location ellipsoids, Fisher decomposition, resolution slices,
sensor ranking maps, VCE convergence, and greedy placement results.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.stats import chi2 as chi2_dist
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DesignDiagnostics:
    """Visualization tools for optimal design and adaptive weighting."""

    @staticmethod
    def plot_sensor_ranking(
        receiver_xyz: np.ndarray,
        location_gains: np.ndarray,
        tomo_values: np.ndarray,
        source_xyz: np.ndarray,
        das_xyz: Optional[np.ndarray] = None,
    ):
        """
        Side-by-side station ranking for location vs tomography.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Location gain map
        ax = axes[0]
        if das_xyz is not None:
            ax.plot(das_xyz[:, 0], das_xyz[:, 1], '.', color='#ccc',
                    ms=0.5, alpha=0.3)
        sc = ax.scatter(
            receiver_xyz[:, 0], receiver_xyz[:, 1],
            c=location_gains, cmap='YlOrRd', s=60, edgecolor='k',
            linewidth=0.5, zorder=5
        )
        ax.plot(source_xyz[0], source_xyz[1], 'k*', ms=15, zorder=10)
        plt.colorbar(sc, ax=ax, label='Location info gain (nats)')
        ax.set_title('Station Value: Location')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_aspect('equal')

        # Tomography value map
        ax = axes[1]
        if das_xyz is not None:
            ax.plot(das_xyz[:, 0], das_xyz[:, 1], '.', color='#ccc',
                    ms=0.5, alpha=0.3)
        sc = ax.scatter(
            receiver_xyz[:, 0], receiver_xyz[:, 1],
            c=tomo_values, cmap='YlGnBu', s=60, edgecolor='k',
            linewidth=0.5, zorder=5
        )
        ax.plot(source_xyz[0], source_xyz[1], 'k*', ms=15, zorder=10)
        plt.colorbar(sc, ax=ax, label='Tomography resolution value')
        ax.set_title('Station Value: Tomography')
        ax.set_xlabel('X (km)')
        ax.set_aspect('equal')

        # Comparison scatter
        ax = axes[2]
        ax.scatter(location_gains, tomo_values, c='#4060c0', s=40,
                   alpha=0.7, edgecolor='k', linewidth=0.3)
        ax.set_xlabel('Location Information Gain')
        ax.set_ylabel('Tomography Resolution Value')
        ax.set_title('Location vs Tomography Trade-off')
        ax.grid(True, alpha=0.3)

        # Label outliers
        for i in range(len(location_gains)):
            if (location_gains[i] > np.percentile(location_gains, 90) or
                    tomo_values[i] > np.percentile(tomo_values, 90)):
                ax.annotate(
                    f'{i}', (location_gains[i], tomo_values[i]),
                    fontsize=7, ha='center'
                )

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_greedy_placement(
        candidate_positions: np.ndarray,
        selected_indices: List[int],
        criterion_history: List[float],
        marginal_gains: List[float],
        existing_receivers: Optional[np.ndarray] = None,
        das_xyz: Optional[np.ndarray] = None,
        source_xyz: Optional[np.ndarray] = None,
    ):
        """
        Visualize greedy sensor placement results.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Map of selected positions
        ax = axes[0]
        if das_xyz is not None:
            ax.plot(das_xyz[:, 0], das_xyz[:, 1], '.', color='#ddd',
                    ms=0.5, alpha=0.3, label='DAS')
        if existing_receivers is not None:
            ax.plot(existing_receivers[:, 0], existing_receivers[:, 1],
                    '^', color='#4060c0', ms=7, label='Existing BB')
        ax.plot(candidate_positions[:, 0], candidate_positions[:, 1],
                '.', color='#ccc', ms=3, alpha=0.5, label='Candidates')

        sel = candidate_positions[selected_indices]
        colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, len(selected_indices)))
        for i, (pos, c) in enumerate(zip(sel, colors)):
            ax.plot(pos[0], pos[1], 's', color=c, ms=10,
                    markeredgecolor='k', markeredgewidth=1, zorder=10)
            ax.annotate(f'{i + 1}', (pos[0], pos[1]), fontsize=7,
                        ha='center', va='bottom', fontweight='bold')

        if source_xyz is not None:
            ax.plot(source_xyz[0], source_xyz[1], 'k*', ms=15, zorder=10)
        ax.set_title('Optimal Sensor Placement')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_aspect('equal')
        ax.legend(fontsize=7)

        # Criterion history
        ax = axes[1]
        ax.plot(range(len(criterion_history)), criterion_history,
                'o-', color='#2d7a4f', lw=2, ms=6)
        ax.set_xlabel('Number of added sensors')
        ax.set_ylabel('D-optimal criterion (log det F)')
        ax.set_title('Information Growth')
        ax.grid(True, alpha=0.3)

        # Marginal gains
        ax = axes[2]
        ax.bar(range(1, len(marginal_gains) + 1), marginal_gains,
               color='#d06040', alpha=0.8, edgecolor='k', linewidth=0.5)
        ax.set_xlabel('Sensor addition order')
        ax.set_ylabel('Marginal information gain')
        ax.set_title('Diminishing Returns')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_vce_results(vce_result: Dict):
        """Visualize variance component estimation convergence."""
        import matplotlib.pyplot as plt

        history = vce_result['convergence']
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        iters = range(1, len(history) + 1)
        kd = [h[0] for h in history]
        kb = [h[1] for h in history]

        ax = axes[0]
        ax.plot(iters, kd, 'o-', color='#d06040', lw=2, label='κ_DAS')
        ax.plot(iters, kb, 's-', color='#4060c0', lw=2, label='κ_BB')
        ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='κ=1 (ideal)')
        ax.set_xlabel('VCE Iteration')
        ax.set_ylabel('Variance Scale Factor κ')
        ax.set_title('Variance Component Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        labels = ['DAS', 'Broadband']
        assumed = [
            vce_result['sigma_das_estimated'] / np.sqrt(vce_result['kappa_das']),
            vce_result['sigma_bb_estimated'] / np.sqrt(vce_result['kappa_bb']),
        ]
        estimated = [
            vce_result['sigma_das_estimated'],
            vce_result['sigma_bb_estimated'],
        ]
        x = np.arange(2)
        ax.bar(x - 0.15, [a * 1000 for a in assumed], 0.3,
               color='#aaa', label='Assumed σ')
        ax.bar(x + 0.15, [e * 1000 for e in estimated], 0.3,
               color=['#d06040', '#4060c0'], label='Estimated σ')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Pick Uncertainty (ms)')
        ax.set_title('Assumed vs Estimated σ')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        return fig


class InversionDiagnostics:
    """Core inversion plotting tools."""

    @staticmethod
    def plot_location_comparison(
        result_joint, result_das_only=None, result_bb_only=None,
        true_source=None
    ):
        """Compare location ellipsoids: joint vs DAS-only vs BB-only."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        configs = [('XY (map view)', 0, 1), ('XZ (depth)', 0, 2), ('YZ (depth)', 1, 2)]

        for ax, (title, ix, iy) in zip(axes, configs):
            for result, color, label in [
                (result_joint, '#2d7a4f', 'Joint'),
                (result_das_only, '#d06040', 'DAS only'),
                (result_bb_only, '#4060c0', 'BB only'),
            ]:
                if result is None or result.get('posterior_cov') is None:
                    continue
                C_2d = result['posterior_cov'][np.ix_([ix, iy], [ix, iy])]
                eigvals, eigvecs = eigh(C_2d)
                kappa = np.sqrt(chi2_dist.ppf(0.95, df=2))
                angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
                w = 2 * kappa * np.sqrt(max(eigvals[1], 1e-15))
                h = 2 * kappa * np.sqrt(max(eigvals[0], 1e-15))
                s = result['source_xyz']
                ell = Ellipse(xy=(s[ix], s[iy]), width=w, height=h, angle=angle,
                              fill=False, edgecolor=color, lw=2, label=label,
                              linestyle='-' if label == 'Joint' else '--')
                ax.add_patch(ell)
                ax.plot(s[ix], s[iy], 'o', color=color, ms=4)

            if true_source is not None:
                ax.plot(true_source[ix], true_source[iy], 'k*', ms=12, label='True', zorder=10)

            dim_labels = ['X (km)', 'Y (km)', 'Z (km)']
            ax.set_xlabel(dim_labels[ix])
            ax.set_ylabel(dim_labels[iy])
            ax.set_title(f'95% Ellipse — {title}')
            ax.legend(fontsize=8)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.autoscale()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_information_decomposition(F_das, F_bb):
        """Stacked bar chart of per-direction Fisher information."""
        import matplotlib.pyplot as plt

        F_total = F_das + F_bb
        eigenvalues, eigenvectors = eigh(F_total)
        param_names = ['X', 'Y', 'Z', 'τ']
        labels, das_fracs, bb_fracs = [], [], []

        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            labels.append(param_names[np.argmax(np.abs(v))])
            info_d = float(v @ F_das @ v)
            info_b = float(v @ F_bb @ v)
            total = info_d + info_b
            das_fracs.append(info_d / total if total > 1e-15 else 0)
            bb_fracs.append(info_b / total if total > 1e-15 else 0)

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))
        ax.bar(x, das_fracs, label='DAS', color='#d06040')
        ax.bar(x, bb_fracs, bottom=das_fracs, label='Broadband', color='#4060c0')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Information Fraction')
        ax.set_title('Fisher Information Decomposition')
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        return fig
