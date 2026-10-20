
# Setup ------------------------------------------------------------------------

import polars as pl
import numpy as np

import plotnine as gg
from plotnine import ggplot, aes
from matplotlib import colormaps, colors

from numpy.typing import NDArray

from ._core import PCAHighFreqResult


def palette_cmap(
    cmap: str, n: int, lims: tuple[float, float] = (0, 1),
    rev: bool = False
) -> list[str]:
    base_cmap = colormaps[cmap]
    color_points = np.linspace(lims[0], lims[1], n)
    res = [colors.to_hex(color) for color in base_cmap(color_points)]
    return res[::-1] if rev else res



# Plots ------------------------------------------------------------------------

def plot_loadings(
    pca: PCAHighFreqResult,
    exp: float = 1,
    merge_blocks: int = 1
) -> ggplot:
    """Plot the loading contribution between variables and principal components over time.

    The absolute loadings are normalized to `[0-1]` each block-variable pair.
    Then, in a pannel for each original variable, a plot of the time serie of
    loadings for each principal component is plotted. This shows how a variable
    contributed to each PC over time. A high contribution to the first PCs
    indicates high variance explanatory power.

    Example with high-frequency factor zoo data from [Aleti
    2022](https://dx.doi.org/10.2139/ssrn.4021620):

    ![](../.assets/plot_loadings.png)

    Args:
        pca: The result of `pca_high_freq`.
        exp: Exponent applied to the absolute loadings before normalization.
            Higher values emphasize the dominant loading components.
        merge_blocks: Number of blocks to merge (sum) in a single observation
            for plotting. This is useful to reduce the number of points in the
            plot and improve visibility.

    Returns:
        A [ggplot][plotnine.ggplot] object. Print the object to show the plot,
        or save it with `plotnine.ggsave`. Modifications can be made by adding
        plotnine calls to the result: `plot_loadings(...) + plotnine.*(...)`.
    """
    gdata = (
        pca.evectors_as_polars()
        .unpivot(pca.col_names("pc"), index = ["block", "variable"], variable_name = "pc")
        .with_columns(pl.col("value").abs())
        .with_columns(
            (pl.col("value") ** exp) /
            (pl.col("value") ** exp).sum().over(["block", "variable"])
        )
        .group_by(pl.col("block") // merge_blocks, "variable", "pc")
        .agg(pl.col("value").sum())
        .with_columns(
            pl.col("block") * merge_blocks,
            pl.col("pc").str.replace("pc_", "PC ")
        )
    )

    g = (
        ggplot(gdata, aes("block", "value", fill = "pc")) +
        gg.geom_area(position = gg.position_stack(reverse = True)) +
        gg.facet_wrap("variable") +
        gg.scale_fill_manual(values = palette_cmap("viridis", 6, (0.2, 1), True)) +
        gg.labs(
            title = "Factors' absolute loadings on PCs across time",
            x = f"Block (block size = {pca.k})", y = "Absolute loading",
            fill = "",
            caption = "PC: principal component, ordered by variance explained"
        ) +
        gg.theme_bw() +
        gg.theme(legend_position = "bottom")
    )

    return g


def plot_pcs(
    pca: PCAHighFreqResult,
    merge_blocks: int = 1
) -> ggplot:
    """Plot the principal component trajectories colored by explained variance.

    Example with high-frequency factor zoo data from [Aleti
    2022](https://dx.doi.org/10.2139/ssrn.4021620):

    ![](../.assets/plot_pcs.png)

    Args:
        pca: The result of `pca_high_freq`.
        merge_blocks: Number of blocks to merge (sum) in a single observation
            for plotting. This is useful to reduce the number of points in the
            plot and improve visibility.

    Returns:
        A [ggplot][plotnine.ggplot] object. Print the object to show the plot,
        or save it with `plotnine.ggsave`. Modifications can be made by adding
        plotnine calls to the result: `plot_loadings(...) + plotnine.*(...)`.
    """
    col_names = pca.col_names("pc")

    evalues_repeated = (
        pca.evalues_as_polars()
        .with_columns(
            (pl.col(c) / pl.sum_horizontal(col_names)) for c in col_names
        )
        .unpivot(
            col_names, index = ["block"],
            variable_name = "evalue_name", value_name = "evalue"
        )
        ["evalue"]
        .repeat_by(pca.k)
        .explode()
    )

    gdata_pcs = (
        pca.pcs_as_polars()
        .unpivot(
            col_names, index = ["index", "block"],
            variable_name = "pc", value_name = "pc_value"
        )
        .with_columns(
            pl.col("pc").str.replace("pc_", "PC "),
            evalue = evalues_repeated, # Join is too expensive
        )
        .group_by("pc", index_group = pl.col("index") // 50)
        .agg(
            pl.col("pc_value").sum(),
            pl.col("evalue").mean(),
            pl.col("index").last(),
        )
        .drop("index_group")
    )

    g = (
        ggplot(gdata_pcs, aes("index", "pc_value")) +
        gg.geom_segment(
            aes(xend = "index", y = 0, yend = "pc_value", color = "evalue"),
            size = 0.1, alpha = 0.5
        ) +
        gg.facet_wrap("pc") +
        gg.labs(
            title = "Principal Components across time",
            x = "Observation", y = "Value", color = "Var. explained",
            caption = "PC: principal component, ordered by variance explained"
        ) +
        gg.theme_bw() +
        gg.theme(
            axis_text_x=gg.element_text(rotation = 45, hjust = 1)
        )
    )

    return g



# Others -----------------------------------------------------------------------

def quantiles_cut(
    X: NDArray[np.number],
    pca: PCAHighFreqResult
) -> NDArray[np.number]:
    """Compute the fraction of values above the PCA cutoff per feature.

    Args:
        X: Input data array with shape `(T, d)`, as passed to `pca_high_freq`.
        pca: the result of `pca_high_freq`.

    Returns:
        The fraction of values above the cutoff with shape `(d,)`.
    """
    return np.mean(X > pca.u, axis = 0)
