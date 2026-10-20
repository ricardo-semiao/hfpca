## quantiles_cut()


Compute the fraction of values above the PCA cutoff per feature.


Usage

``` python
quantiles_cut(
    X,
    pca,
)
```


## Parameters


`X: NDArray[np.number]`  
Input data array with shape `(T, d)`, as passed to [pca_high_freq](pca_high_freq.md#hfpca.pca_high_freq).

`pca: PCAHighFreqResult`  
the result of [pca_high_freq](pca_high_freq.md#hfpca.pca_high_freq).


## Returns


`NDArray[np.number]`  
The fraction of values above the cutoff with shape `(d,)`.
