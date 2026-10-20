import numpy as np
import pytest

from hfpca import pca_high_freq
from hfpca.analisys import plot_loadings, plot_pcs, quantiles_cut


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=10)
def test_plot_loadings_visual(benchmark_X_small: np.ndarray):
    result = pca_high_freq(benchmark_X_small)
    figure = plot_loadings(result)

    return figure


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=10)
def test_plot_pcs_visual(benchmark_X_small: np.ndarray):
    result = pca_high_freq(benchmark_X_small)
    figure = plot_pcs(result)

    return figure


def test_quantiles_cut_known_values(
    benchmark_X_small: np.ndarray,
) -> None:
    result = pca_high_freq(benchmark_X_small)
    actual = quantiles_cut(benchmark_X_small, result)
    expected = np.array([
        1.821096e-05, 6.070319e-06, 9.105478e-06,
        3.035159e-06, 9.105478e-06, 3.035159e-06,
    ])

    np.testing.assert_allclose(actual, expected, rtol=1e-5)
