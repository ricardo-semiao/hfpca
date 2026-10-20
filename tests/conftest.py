import pytest
import numpy as np
from numpy.typing import NDArray

def make_benchmark_X(shape: tuple[int, int] = (1647360, 6)) -> NDArray[np.float64]:
    """Generates a synthetic high-frequency matrix with factor structure.

    Default shape: (1,647,360, 6)
    Secondary shape: (329,472, 6)
    """
    rng = np.random.default_rng(seed=42)
    T, d = shape

    # Latent factors
    f1 = rng.normal(0, 1, T)
    f2 = rng.normal(0, 1, T)

    X = np.empty((T, d), dtype=np.float64)

    # Factor 1 loadings + noise
    X[:, 0] = 0.8 * f1 + 0.2 * rng.normal(0, 1, T)
    X[:, 1] = 0.6 * f1 + 0.4 * rng.normal(0, 1, T)
    X[:, 2] = -0.7 * f1 + 0.3 * rng.normal(0, 1, T)

    # Factor 2 loadings + noise
    X[:, 3] = 0.9 * f2 + 0.1 * rng.normal(0, 1, T)
    X[:, 4] = -0.5 * f2 + 0.5 * rng.normal(0, 1, T)

    # Pure noise column
    X[:, 5] = rng.normal(0, 1, T)

    return X


@pytest.fixture(scope="session")
def benchmark_X_large() -> NDArray[np.float64]:
    """Large dataset fixture: shape (1,647,360, 6)."""
    return make_benchmark_X((1647360, 6))


@pytest.fixture(scope="session")
def benchmark_X_small() -> NDArray[np.float64]:
    """Medium dataset fixture: shape (329,472, 6)."""
    return make_benchmark_X((329472, 6))