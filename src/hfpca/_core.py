
import numpy as np
import polars as pl

from numpy.typing import NDArray
from polars._typing import IntoExpr
from typing import Any

from collections.abc import Iterator
from dataclasses import dataclass

from tqdm import tqdm


def pca_truncated(
    X: NDArray[np.number], # Shape: (k, d) a block of the original X
    n: float, # k * Delta
    u: NDArray[np.number] # Shape: (d,)
) -> tuple[NDArray[np.number], NDArray[np.number]]:
    X_trunc = X * (np.abs(X) <= u) # x*0 for big values, else x*1
    cov_spot = (X_trunc.T @ X_trunc) / n # No need to demean following the paper

    evalues, evectors = np.linalg.eigh(cov_spot)
    evalues_order = np.argsort(evalues)[::-1]

    return evalues[evalues_order], evectors[:, evalues_order]

def pca_high_freq_core(
    N: int, k: int, T: int, Delta: float,
    u: NDArray[np.number], # Shape: (d,)
    X: NDArray[np.number], # Shape: (T, d)
    evectors_prev: NDArray[np.number], # Shape: (d, d)
    PC: NDArray[np.number], # Shape: (T, d)
    E_vals: NDArray[np.number], # Shape: (N, d)
    E_vecs: NDArray[np.number], # Shape: (N, d, d)
    quiet: bool = False
) -> tuple[NDArray[np.number], NDArray[np.number], NDArray[np.number]]:
    for i in tqdm(range(N), total = N, disable = quiet): # Last block after loop
        block_start, block_end = i * k, (i + 1) * k

        X_block = X[block_start:block_end, :]

        PC[block_start:block_end, :] = X_block @ evectors_prev

        evalues_prev, evectors_prev = pca_truncated(X_block, k * Delta, u)
        E_vals[i, :] = evalues_prev
        E_vecs[i, :, :] = evectors_prev

    block_start = N * k
    if block_start < T:
        X_block = X[block_start:T, :]
        PC[block_start:T, :] = X_block @ evectors_prev

    return PC, E_vals, E_vecs


@dataclass(frozen = True, slots = True)
class PCAHighFreqResult:
    """PCA high-frequency result.

    Container for principal component scores, eigenvalues, and eigenvectors
    produced by `pca_high_freq`. They can be unpacked into variables with `pcs,
    evalues, evectors = pca_high_freq(...)`. Additional metadata can be acessed
    by name.

    Attributes:
        pcs: Principal component scores for each time index and component, with
          shape `(T, d)`.
        evalues: Eigenvalues for each block and retained component, with shape
          `(T // k + 1, d)`.
        evectors: Eigenvectors for each block and variable/component pair, with
          shape `(T // k + 1, d, d)`.
        k: Block length.
        u: Jump truncation cutoffs for each variable, with shape `(d,)`.
        d: Number of variables in the original data (`X`).
    """

    pcs: NDArray[np.number]
    evalues: NDArray[np.number]
    evectors: NDArray[np.number]
    k: int
    u: NDArray[np.number]
    d: int

    def __iter__(self) -> Iterator[NDArray[np.number]]:
        yield self.pcs
        yield self.evalues
        yield self.evectors

    def __getitem__(self, idx: int | slice) -> Any:
        return (self.pcs, self.evalues, self.evectors)[idx]

    def __len__(self) -> int:
        return 3

    def col_names(self, prefix: str = "var") -> list[str]:
        """Build column names for a result matrix.

        Useful for manipulating the results in dataframes.

        Args:
            prefix: Prefix used for each generated column name.

        Returns:
            Column names in the form `"{prefix}_1"`, `"{prefix}_2"`, ...,
            `"{prefix}_{d}"`.
        """
        return [f"{prefix}_{i + 1}" for i in range(self.d)]

    def pcs_as_polars(self, **cols_to_add: IntoExpr) -> pl.DataFrame:
        """Convert the principal component scores to a Polars dataframe.

        Args:
            **cols_to_add: Additional polars' expressions to append as columns.

        Returns:
            A [DataFrame][polars.DataFrame] object with shape `(T, d + len(cols_to_add))`, with columns:
            - `index`: A `1:T` index column.
            - `block`: An identifier of which block the observation belongs to,
              with values `0:(T // k)`.
            - `pc_1`, `pc_2`, ..., `pc_d`: The principal component scores.
            - Additional columns specified by `cols_to_add`.
        """
        return (
            pl.DataFrame(self.pcs, schema = self.col_names("pc")) # CHECK: consider from_numpy()
            .with_columns(
                index = pl.row_index(), # CHECK: remove and only add in analysis?
                block = pl.row_index() // self.k,
                **cols_to_add
            )
        )

    def evalues_as_polars(self, **cols_to_add: IntoExpr) -> pl.DataFrame:
        """Convert the eigenvalues to a Polars dataframe.

        Args:
            **cols_to_add: Additional polars expressions to append as columns.

        Returns:
            A [DataFrame][polars.DataFrame] object with shape `(n_blocks, d + len(cols_to_add))`, with columns:
            - `block`: An identifier of which block the eigenvalues belong to,
              with values `0:(n_blocks - 1)`.
            - `pc_1`, `pc_2`, ..., `pc_d`: The eigenvalues for each retained component.
            - Additional columns specified by `cols_to_add`.
        """
        return (
            pl.DataFrame(self.evalues, schema = self.col_names("pc"))
            .with_columns(
                block = pl.row_index(),
                **cols_to_add
            )
        )

    def evectors_as_polars(self, **cols_to_add: IntoExpr) -> pl.DataFrame:
        """Convert the block-wise eigenvectors to a Polars dataframe.

        Args:
            **cols_to_add: Additional polars expressions to append as columns.

        Returns:
            A [DataFrame][polars.DataFrame] object with shape `(n_blocks * d, d + 2 + len(cols_to_add))`, with columns:
            - `pc_1`, `pc_2`, ..., `pc_d`: The eigenvector coefficients for each component.
            - `variable`: A variable identifier with values `0:(d - 1)` for each block.
            - `block`: An identifier of which block the coefficients belong to,
              with values `0:(n_blocks - 1)` repeated by variable.
            - Additional columns specified by `cols_to_add`.
        """
        n_blocks = self.evalues.shape[0]
        return (
            pl.DataFrame(
                self.evectors.transpose(0, 1, 2).reshape((-1, self.d)),
                schema = self.col_names("pc")
            )
            .with_columns(
                variable = pl.repeat(self.col_names("var"), n_blocks).explode(),
                block = pl.arange(0, n_blocks).repeat_by(self.d).list.explode(),
                **cols_to_add
            )
        )
