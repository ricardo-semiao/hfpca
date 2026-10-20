import numpy as np
import pytest

from hfpca._defaults import default_k, default_u


@pytest.mark.parametrize(
    ("T", "d", "expected"),
    [
        (1647360, 6, 858),
        (329472, 6, 384),
    ],
)
def test_default_k(T: int, d: int, expected: int) -> None:
    actual = default_k(d, T, 1 / T)

    assert actual == expected


@pytest.mark.parametrize(
    ("X", "T", "expected"),
    [
        (
            "benchmark_X_large",
            1647360,
            np.array(
                [3.802132, 3.326268, 3.511256,
                 4.173713, 3.254552, 4.608808]
            ),
        ),
        (
            "benchmark_X_small",
            329472,
            np.array(
                [3.622959, 3.169587, 3.345153,
                 3.978647, 3.105421, 4.383059]
            ),
        ),
    ],
)
def test_default_u(
    request: pytest.FixtureRequest,
    X: str,
    T: int,
    expected: np.ndarray,
) -> None:
    matrix = request.getfixturevalue(X)
    actual = default_u(matrix, T, 1 / T)

    np.testing.assert_allclose(actual, expected, rtol=1e-5)
