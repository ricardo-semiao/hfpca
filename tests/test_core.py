import numpy as np

from hfpca import pca_high_freq
from hfpca._defaults import pca_start_evectors


def test_pca_start_evectors_snapshot(
    benchmark_X_small: np.ndarray,
    ndarrays_regression,
) -> None:
    result = pca_start_evectors(benchmark_X_small)

    ndarrays_regression.check(
        {"evectors": result},
        default_tolerance={"rtol": 1e-6},
    )


def test_pca_high_freq_snapshot(
    benchmark_X_small: np.ndarray,
    ndarrays_regression,
) -> None:
    result_default = pca_high_freq(benchmark_X_small)
    custom_u = result_default.u
    result_custom = pca_high_freq(
        benchmark_X_small,
        k=384,
        u=custom_u,
    )

    ndarrays_regression.check(
        {
            "default_pcs": result_default.pcs,
            "default_evalues": result_default.evalues,
            "default_evectors": result_default.evectors,
            "custom_pcs": result_custom.pcs,
            "custom_evalues": result_custom.evalues,
            "custom_evectors": result_custom.evectors,
        },
        default_tolerance={"rtol": 1e-6},
    )
