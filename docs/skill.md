---
name: hfpca
description: >
  Principal component analysis (PCA) for high-frequency time-series data, as proposed in Aït-Sahalia and Xiu (2019). Use when writing Python code that uses the hfpca package.
compatibility: Requires Python >=3.14.
---

# hfpca

Principal component analysis (PCA) for high-frequency time-series data, as proposed in Aït-Sahalia and Xiu (2019).

## Installation

```bash
pip install hfpca
```

## API overview

### Main function

Run the high-frequency PCA algorithm

- `pca_high_freq`: Run high-frequency PCA on a data matrix
- `pca_start_evectors`: Calculate the starting eigenvectors for the high-frequency PCA routine

### Analysis functions

Analyze the results

- `plot_loadings`: Plot the loading contribution between variables and principal components over time
- `plot_pcs`: Plot the principal component trajectories colored by explained variance
- `quantiles_cut`: Compute the fraction of values above the PCA cutoff per feature

## Resources

- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs
