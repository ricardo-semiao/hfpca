# Principal component analysis (PCA) for high-frequency time-series data

An implementation of the algorithm proposed by [Aït-Sahalia and Xiu (2019)](https://www.nber.org/system/files/working_papers/w21584/w21584.pdf), in Python.

Standard PCA suffers from:

- **Curse of dimensionality:** Estimating covariance matrices for large cross-sections requires long time horizons to avoid singularity.
- **Non-stationarity:** The hypothesis of a constant covariance matrix across time can be too strong in some scenarios.
- **Non-linearities:** a single single covariance matrix, a linear measure of relationships, cannot properly capture non-linear relationships.

Aït-Sahalia and Xiu (2019) develops an algorithm that uses high-frequency data to break a process into blocks, each with its own spot covariance matrices and principal components. With that, one can create a dynamic series of principal components, relaxing  stationarity assumptions and approximating non-linear relationships, given Itô's Lemma.


## Installation

One can install the package via pip or uv, from GitHub:

```bash
pip install git+https://github.com/ricardo-semiao/hfpca.git
uv pip install git+https://github.com/ricardo-semiao/hfpca.git
```


## Methodology

The following math and commentary are largely taken as-is from the paper. First, I present the theory, then the implementation in Python, and finally the results.


### Theoretical setup

Consider the data as a $d$-dimensional process $X$ of the factors is a general Itô-semimartingale, defined on a filtered probability space:

$$
    X_t = X_0 + \int_0^t b_s ds + \int_0^t \sigma_s dW_s + J_t
$$

- $X_t$ is the $d$-dimensional log-price (or cumulative factor) process.
- $b_s$ is the $d$-dimensional locally bounded drift vector.
- $\sigma_s$ is the spot volatility matrix, linking to the spot covariance via $c_s = \sigma_s \sigma_s^T$.
- $W_s$ is a $d$-dimensional standard Brownian motion.
- $J_t$ is a pure jump process accounting for discrete anomalous shocks. (More information in the paper).


### Parameters of interest

The PCA goal is to find instantaneous linear combinations of $X$, orthogonal to each other, that maximize some measure of variation. In the paper's case, the continuous part of the quadratic variation.

Let $\mathcal{M}^+_d$ denote the set of $d \times d$ positive semi-definite symmetric matrices. For any $A \in \mathcal{M}^+_d$, we define:

- The $d$-dimensional vector of its eigenvalues sorted in non-increasing order: $\lambda(A) = \left(\lambda_1(A), \lambda_2(A), \dots, \lambda_d(A)\right)^T$.
- The eigenvector corresponding to the $g$-th eigenvalue $\lambda_g(A)$ as $\gamma_g(A)$, which satisfies: $A\gamma_g = \lambda_g \gamma_g$ and $\gamma_g^T \gamma_g = 1$.

Our interest is calculating the $T \times d$ matrix of principal components, where $T$ is the total number of (high-frequency) observations. This is related to the paper's derivation of the integrated/realized principal component of the $g$-th eigenvalue $\lambda_g$, $\int_0^t \gamma_{g,s-}^T dX_s$.

The paper focuses on the estimation of the realized eigenvalues ($\int_0^t \lambda_s ds$) and eigenvectors ($\int_0^t \gamma_{g,s} ds$), while deriving their asymptotic properties and bias corrections. In this package, I focus on the PCs.


### Principal component's estimator

#### Spot covariance matrix

The first step for all is estimating a spot covariance matrix and its associated eigenvalues. To capture non-stationary dynamics, the time window $[0, t]$ in increments of $\Delta_n$ is divided into non-overlapping blocks of length $k_n \Delta_n$. Within each small block, the continuous covariance $c_s$ is assumed to be roughly constant. By dynamically estimating instantaneous matrices and summing them, the procedure continuously adapts to shifting correlations, capturing non-linearities without requiring global parameter estimates.

At each $i k_n \Delta_n$, we estimate $c_{i k_n \Delta_n}$ using the block spot covariance estimator $\hat{c}_{i k_n \Delta_n}$ defined as:

$$
    \hat c_{ik_n}\Delta_n = \frac{1}{k_n\,\Delta_n} \sum_{j=1}^{k_n} \left(\Delta_{ik_n+j}^{n}X\right)\left(\Delta_{ik_n+j}^{n}X\right)^{T} \mathbf{1}\left\{
\parallel\Delta_{ik_n+j}^{n}
\parallel\le u_n\right\}
$$

Where the return increment is: $\Delta_l^n X = X_{l \Delta_n} - X_{(l-1)\Delta_n}$, and $u_n$ is the jump truncation threshold. The $k$ and $u$ parameters can be chosen with the shorthands proposed by the literature:

- **Block Size ($k_n$):** the divisor of $[t / \Delta_n]$ closest to $\theta \Delta_n^{-1/2} \sqrt{\log(d)}$, with $\theta = 0.5$. Other choices of $\theta$ and functions of $d$ deliver similar results. It scales inversely with the sampling frequency ($\Delta_n$) and grows slightly with dimensionality ($d$) to ensure stable covariance estimates without smoothing over true time-varying dynamics.
- **Jump Truncation ($u_{i, n}$):** Set asset-by-asset as $u_{i,n} = 3\left(\int_{0}^{t}c_{ii,s}ds/t\right)^{0.5}\Delta^{0.47}_{n}$. Where the internal integral can be estimated with Bipower Variation. This removes discrete jumps while preserving continuous diffusion.

The instantaneous eigenvalues $\hat{\lambda}_{i k_n \Delta_n} = \lambda(\hat{c}_{i k_n \Delta_n})$ are obtained from the standard determinant equation:

$$
    \text{det}(\hat{c}_i - \hat{\lambda} \mathbb{I}) = 0
$$

#### Integrated principal component

The paper focuses on the integrated principal components, whose estimator is defined as:

$$
    \widehat{PC}_{g} = \sum_{i=1}^{[t/(k_n\Delta_n)]-1}\hat{\gamma}^{T}_{g,(i-1)k_n\Delta_n}\left(X_{(i+1)k_n\Delta_n}-X_{ik_n\Delta_n}\right)
$$

Because the estimator is a stochastic integral, the weight errors are uncorrelated with the forward returns. The estimation noise naturally cancels out, removing the need for a Hessian bias correction. PCs use the eigenvectors estimated from the immediately preceding block to project the current block's returns, ensuring that our dynamic portfolio weights are strictly "adapted" (determined in-sample before trading) and free from microscopic look-ahead bias.

However, we want a $T \times d$ matrix of principal components, not just the integrated one. So we basically do not want to run the summation and instead store the intermediate results. The pseudocode below outlines the algorithm.


### Estimator algorithm

To estimate the $T \times d$ matrix of principal components, we must project each individual high-frequency return observation onto the previous block's eigenvector. The Pascal-like pseudocode below outlines the procedure.

This code was written in matrix form, as it will be a much more effective way to implement in Python. Still, it should be equivalent to the original paper's algorithm.

```pascal
procedure pca_high_freq(
    X : Matrix;            // Shape: (T, d) | high-frequency return matrix
    k : Integer;           // Scalar        | observations per block
    u : Matrix;            // Shape: (d,)   | jump truncation thresholds
    evectors_prev : Matrix // Shape: (d, d) | initial eigenvectors
);

    // 1. Parameter Setup & Initialization
    T = number of rows in X
    d = number of cols in X
    Delta = 1 / T
    N = Floor(T / k) // Total number of full blocks

    // Initialize empty arrays to store outputs:
    PC = empty Matrix of size T x d
    E_vals = empty Matrix of size N x d
    E_vecs = empty Matrix of size N x d x d

    // 2. Process Data Block by Block
    for i = 0 to N - 1 do
    begin
        // Define the current block
        block_start = i * k
        block_end = (i + 1) * k
        X_block = X[block_start to block_end]

        // Calculate PCs for the current block using the previous eigenvectors:
        PC[block_start to block_end] = X_block * evectors_prev

        // Truncate X with boolean mask and calculate the spot covariance matrix:
        X_trunc := X_block * (Abs(X_block) <= u)
        cov_spot = (Transpose(X_trunc) * X_trunc) / (k * Delta)

        // Eigen decomposition:
        evalues, evectors = eigen decomposition of cov_spot, descending evalues

        // Store results and update eigenvectors for the next block:
        E_vals[i, :] = evalues
        E_vecs[i, :, :] = evectors
        evectors_prev = evectors 
    end;

    // 3. Handling the last block if (T % k != 0)
    block_start = N * k
    if block_start < T then
    begin
        X_block = X[block_start to T]
        PC[block_start to T] = X_block * evectors_prev
    end;

    return PC, E_vals, E_vecs
end;
```
