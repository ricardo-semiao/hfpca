
import numpy as np

from numpy.typing import NDArray

from ._validation import pca_high_freq_validate
from ._defaults import default_k, default_u
from ._core import pca_high_freq_core, PCAHighFreqResult


def pca_high_freq(
    X: NDArray[np.number], # Shape: (T, d)
    k: int | None = None,
    u: NDArray[np.number] | None = None, # Shape: (d,)
    evectors_start: NDArray[np.number] | None = None, # Shape: (d, d)
    quiet: bool = False
) -> PCAHighFreqResult:
    """Run high-frequency PCA on a data matrix.

    This function implements the algorithm proposed by [Aït-Sahalia and Xiu
    (2019)](https://www.nber.org/system/files/working_papers/w21584/w21584.pdf),
    returning the time series of principal components, eigenvalues, and
    eigenvectors.

    A summary of the algorithm's hypothesis and parametrization can be found in
    this package's documentation.

    Args:
        X: Input data array with shape `(T, d)`, where `T` is the number of
            observations and `d` is the number of variables. Must not contain
            `npnan` or `np.inf` values.
        k: Block size used for the high-frequency PCA routine. If `None`,
            it is chosen by a rule-of-thumb. Must be smaller than `d`
        u: Jump truncation cutoffs for each variable, with shape `(d,)`. If
            `None`, it is chosen by a rule-of-thumb. Must not contain
            `npnan`, `np.inf`, or nonpositive values.
        evectors_start: Optional initial eigenvector matrix with shape `(d, d)`
            used to warm-start the iteration. If not supplied, the first block
            of the results will be `np.nan`.
        quiet: If ``True``, suppress progress prints during computation.

    Returns:
        A [`PCAHighFreqResult`][hfpca.types.PCAHighFreqResult] object containing
        the the PCs (`.pcs`), eigenvalue estimates (`.evalues`), eigenvector
        estimates (`.evectors`), which can be unpacked into variables. Also
        contains: the values of `k` (`.k`), `u` (`.u`), and `d` (`.d`); and
        methods for extracting PCs, eigenvalues, and eigenvectors as polars
        dataframes (`.pcs_as_polars()`). See the class documentation for more
        details.

    Raises:
        TypeError: If `X`, `u`, or `evectors_start` are not numpy arrays; if
            they do not have a numeric dtype; or if `k` is not an integer.
        ValueError: If `X` is not 2D, or if `u` or `evectors_start` do not have
            the correct dimensions; if any of the inputs contain NaN or Inf
            values; if `u` contains nonpositive values; or if `k` is smaller
            than `d` or larger than `T`.
    """
    # Setup:
    pca_high_freq_validate(X, k, u, evectors_start)
    X.flags.writeable = False

    T: int = X.shape[0] # Number of observations (rows)
    d: int = X.shape[1] # Number of variables (columns)
    Delta: float = 1 / T

    if k is None:
        k = default_k(d, T, Delta)
        if not quiet: print(f"Calculated `k` as {k}.")
    if u is None:
        u = default_u(X, T, Delta)
        if not quiet: print(f"Calculated `u` as {u}.")

    # Warm up:
    N: int = T // k # Number of blocks - 1
    if evectors_start is None:
        evectors_prev = np.full((d, d), np.nan)
    else:
        evectors_prev = evectors_start # Shape: (d, d)

    PC = np.full((T, d), np.nan) # Shape: (T, d)
    E_vals = np.full((N, d), np.nan) # Shape: (N, d)
    E_vecs = np.full((N, d, d), np.nan) # Shape: (N, d, d)

    # Main loop:
    if not quiet: print(f"Starting PCA with {N} blocks of size {k}.")
    PC, E_vals, E_vecs = pca_high_freq_core(
        N = N, k = k, T = T, Delta = Delta, u = u,
        X = X, evectors_prev = evectors_prev,
        PC = PC, E_vals = E_vals, E_vecs = E_vecs,
        quiet = quiet
    )

    return PCAHighFreqResult(PC, E_vals, E_vecs, k, u, d)
