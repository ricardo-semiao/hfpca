
import numpy as np
from numpy.typing import NDArray

from ._validation import pca_high_freq_validate
from ._core import pca_truncated


def default_k(d: int, T: int, Delta: float) -> int:
    theta = 0.5

    k_target = int(np.round(theta * (Delta**-0.5) * np.sqrt(np.log(d))))
    divisors = np.array([i for i in range(1, T + 1) if T % i == 0])
    closest_idx = np.argmin(np.abs(divisors - k_target))

    return int(divisors[closest_idx])


def default_u(X: NDArray[np.number], T: int, Delta: float) -> NDArray[np.number]:
    X_abs = np.abs(X)
    bpv = (np.pi / 2) * np.sum(X_abs[1:] * X_abs[:-1], axis = 0)
    u: NDArray[np.number] = 3 * np.sqrt(bpv / 1) * (Delta ** 0.47) # bpv/1: $t$ normalized to 1
    return u


def pca_start_evectors(
    X: NDArray[np.number], k: int | None = None, u: NDArray[np.number] | None = None,
    quiet: bool = True
) -> NDArray[np.number]:
    """Calculate the starting eigenvectors for the high-frequency PCA routine.

    Function is being questioned. Will probably be moved inside `pca_high_freq`.
    """
    pca_high_freq_validate(X, k, u, None)

    T: int = X.shape[0]
    d: int = X.shape[1]
    Delta: float = 1 / T

    if k is None:
        k = default_k(d, T, Delta)
        if not quiet: print(f"Calculated `k` as {k}.")
    if u is None:
        u = default_u(X, T, Delta)
        if not quiet: print(f"Calculated `u` as {u}.")

    _, evectors = pca_truncated(X, k * Delta, u)

    return evectors
