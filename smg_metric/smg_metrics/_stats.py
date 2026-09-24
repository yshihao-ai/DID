"""Shared statistical / distribution helpers for smg_metrics.

Centralises overlap area, KL divergence, and normal-distribution overlap
computations used by multiple metric modules.

References:
    - Overlap area: Ren et al., "SongMASS," ACM-MM 2020.
    - KL divergence: Hu et al., "Rule Guided Music Generation," ICML 2024.
    - Normal overlap: yjhuangcd/rule-guided-music, figaro/evaluate.py.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "histogram_overlap",
    "kl_divergence",
    "overlap_normal",
]


def histogram_overlap(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Compute overlap area between two (un-)normalised histograms.

    .. math::

        \\text{overlap} = \\sum_i \\min(\\hat{a}_i, \\hat{b}_i)

    where :math:`\\hat{a}` and :math:`\\hat{b}` are L1-normalised versions
    of *hist_a* and *hist_b*.

    Reference:
        Ren et al., "SongMASS," ACM-MM 2020.
        microsoft/muzic, telemelody/evaluation/cal_similarity.py.

    Args:
        hist_a: First histogram (numpy array).
        hist_b: Second histogram (numpy array, same shape).

    Returns:
        Overlap area in [0, 1].  Returns 0.0 if either histogram sums to 0.
    """
    sum_a = np.sum(hist_a)
    sum_b = np.sum(hist_b)
    if sum_a == 0 or sum_b == 0:
        return 0.0
    norm_a = hist_a.astype(np.float64) / sum_a
    norm_b = hist_b.astype(np.float64) / sum_b
    return float(np.sum(np.minimum(norm_a, norm_b)))


def kl_divergence(
    samples_a: list[float],
    samples_b: list[float],
    n_bins: int = 100,
    eps: float = 1e-10,
) -> float:
    """Estimate :math:`\\mathrm{KL}(P_a \\| P_b)` via histogram density.

    Builds histograms on a shared grid, normalises to probability
    distributions, and computes the KL divergence with Laplace smoothing.

    Reference:
        yjhuangcd/rule-guided-music, music_evaluation/mgeval/utils.py.
        Hu et al., "Rule Guided Music Generation," ICML 2024.

    Args:
        samples_a: Samples from distribution P_a.
        samples_b: Samples from distribution P_b.
        n_bins:    Number of histogram bins (default 100).
        eps:       Smoothing constant (default 1e-10).

    Returns:
        KL divergence in [0, inf), or NaN if either sample list is empty.
    """
    if not samples_a or not samples_b:
        return float("nan")

    all_vals = samples_a + samples_b
    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        return 0.0

    margin = (hi - lo) * 0.01
    lo -= margin
    hi += margin

    hist_a, _ = np.histogram(samples_a, bins=n_bins, range=(lo, hi), density=True)
    hist_b, _ = np.histogram(samples_b, bins=n_bins, range=(lo, hi), density=True)

    sum_a = hist_a.sum()
    sum_b = hist_b.sum()
    if sum_a < eps or sum_b < eps:
        return float("nan")

    p = hist_a / sum_a + eps
    q = hist_b / sum_b + eps
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def overlap_normal(
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    eps: float = 0.01,
) -> float:
    """Compute overlap area between two normal distributions.

    Uses numerical integration of :math:`\\min(\\mathcal{N}_1, \\mathcal{N}_2)`
    on a dense grid.

    Reference:
        Hu et al., "Rule Guided Music Generation," ICML 2024.
        yjhuangcd/rule-guided-music, figaro/evaluate.py.

    Args:
        mu1:    Mean of the first distribution.
        sigma1: Std of the first distribution (clamped to *eps*).
        mu2:    Mean of the second distribution.
        sigma2: Std of the second distribution (clamped to *eps*).
        eps:    Minimum std to avoid degeneracy (default 0.01).

    Returns:
        Overlap area in [0, 1].
    """
    from scipy.stats import norm

    sigma1 = max(eps, sigma1)
    sigma2 = max(eps, sigma2)

    lo = min(mu1 - 4 * sigma1, mu2 - 4 * sigma2)
    hi = max(mu1 + 4 * sigma1, mu2 + 4 * sigma2)
    x = np.linspace(lo, hi, 1000)
    pdf1 = norm.pdf(x, mu1, sigma1)
    pdf2 = norm.pdf(x, mu2, sigma2)
    dx = x[1] - x[0]
    return float(np.sum(np.minimum(pdf1, pdf2)) * dx)
