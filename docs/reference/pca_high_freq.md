## pca_high_freq()


Run high-frequency PCA on a data matrix.


Usage

``` python
pca_high_freq(
    X,
    k=None,
    u=None,
    evectors_start=None,
    quiet=False,
)
```


This function implements the algorithm proposed by [Aït-Sahalia and Xiu (2019)](https://www.nber.org/system/files/working_papers/w21584/w21584.pdf), returning the time series of principal components, eigenvalues, and eigenvectors.

A summary of the algorithm's hypothesis and parametrization can be found in this package's documentation.


## Parameters


`X: NDArray[np.number]`  
Input data array with shape `(T, d)`, where `T` is the number of observations and `d` is the number of variables. Must not contain `npnan` or `np.inf` values.

`k: int | None = None`  
Block size used for the high-frequency PCA routine. If `None`, it is chosen by a rule-of-thumb. Must be smaller than `d`

`u: NDArray[np.number] | None = None`  
Jump truncation cutoffs for each variable, with shape `(d,)`. If `None`, it is chosen by a rule-of-thumb. Must not contain `npnan`, `np.inf`, or nonpositive values.

`evectors_start: NDArray[np.number] | None = None`  
Optional initial eigenvector matrix with shape `(d, d)` used to warm-start the iteration. If not supplied, the first block of the results will be `np.nan`.

`quiet: bool = ``False`  
If `True`, suppress progress prints during computation.


## Returns


`PCAHighFreqResult`  
A \[`PCAHighFreqResult`\]\[hfpca.types.PCAHighFreqResult\] object containing

the the PCs (`.pcs`), eigenvalue estimates (`.evalues`), eigenvector

estimates (`.evectors`), which can be unpacked into variables. Also

`contains: PCAHighFreqResult`  
the values of `k` (`.k`), `u` (`.u`), and `d` (`.d`); and

`PCAHighFreqResult`  
methods for extracting PCs, eigenvalues, and eigenvectors as polars

dataframes (`.pcs_as_polars()`). See the class documentation for more

details.


## Raises


`TypeError`  
If `X`, `u`, or `evectors_start` are not numpy arrays; if they do not have a numeric dtype; or if `k` is not an integer.

`ValueError`  
If `X` is not 2D, or if `u` or `evectors_start` do not have the correct dimensions; if any of the inputs contain NaN or Inf values; if `u` contains nonpositive values; or if `k` is smaller than `d` or larger than `T`.
