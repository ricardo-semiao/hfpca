import numpy as np
import pytest

from hfpca._validation import pca_high_freq_validate


@pytest.mark.parametrize(
    ("X", "k", "message"),
    [
        (np.array([1.0, 2.0, 3.0]), None, r"2D array \(matrix\)"),
        (np.zeros((10, 5, 2)), None, r"2D array \(matrix\)"),
        (np.array([[np.nan, 1.0], [2.0, 3.0]]), None, "contain NaN"),
        (np.array([[np.inf, 1.0], [2.0, 3.0]]), None, "contain NaN or Inf"),
        (np.ones((10, 6)), 11, "bigger or equal"),
    ],
)
def test_pca_high_freq_validate_invalid_inputs(
    X: np.ndarray,
    k: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pca_high_freq_validate(X, k, None)
