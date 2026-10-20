
import numpy as np
from numpy.typing import NDArray


def pca_high_freq_validate(
    X: NDArray[np.number], k: int | None, u: NDArray[np.number] | None,
    evectors_start: NDArray[np.number] | None = None
) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError("`X` must be a numpy `ndarray`.")
    if X.ndim != 2:
        raise ValueError("`X` must be a 2D array (matrix).")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("`X` must be a `np.number` dtype.")
    if not np.all(np.isfinite(X)):
        raise ValueError("`X` must not contain NaN or Inf values.")

    T, d = X.shape

    if k is not None:
        if not isinstance(k, int):
            raise TypeError("`k` must be an `int`.")
        if k <= d:
            raise ValueError("`k` must be bigger than `X.shape[1]`.")
        if k > T:
            raise ValueError("`k` must be bigger or equal to `X.shape[0]`.")

    if u is not None:
        if not isinstance(u, np.ndarray):
            raise TypeError("`u` must be a numpy `ndarray`.")
        if not np.issubdtype(u.dtype, np.number):
            raise TypeError("`u` must be a `np.number` dtype.")
        if u.ndim != 1 or u.shape[0] != d:
            raise ValueError("`u` must have (1 x `X.shape[1]`) dimensions.")
        if not np.all(np.isfinite(u)):
            raise ValueError("`u` must not contain NaN or Inf values.")
        if not np.all(u > 0):
            raise ValueError("`u` must contain only strictly positive values.")

    if evectors_start is not None:
        if not isinstance(evectors_start, np.ndarray):
            raise TypeError("`evectors_start` must be a numpy `ndarray`.")
        if not np.issubdtype(evectors_start.dtype, np.number):
            raise TypeError("`evectors_start` must be a `np.number` dtype.")
        es = evectors_start.shape
        if evectors_start.ndim != 2 or es != (d, d):
            raise ValueError(
                "`evectors_start` must have (`X.shape[1]` x "
                "`X.shape[1]`) dimensions."
            )

# TODO: Warn if u is too big given X general level
# TODO: Implement k and u bounds from proposition 3:
# varsigma = np.log(k) / np.log(T)
# k_conforms = (gamma / 2) < varsigma < 0.5
# C = 3.0 * np.sqrt((np.pi / 2) * np.sum(np.abs(X[1:] * X[:-1]), axis=0))
# varpi = np.log(np.maximum(u / C, 1e-12)) / np.log(1 / T)
# u_conforms = np.all((varpi >= (1 - varsigma) / (2 - gamma)) & (varpi < 0.5))
