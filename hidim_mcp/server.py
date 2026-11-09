"""hidim MCP server: tools for working with high-dimensional data without
dragging the data into the model's context.

Run:  python -m hidim_mcp.server      (stdio transport, for Claude Desktop / Claude Code)
"""
from __future__ import annotations

import logging
import sys
from typing import Any

# --- SDK compatibility: mcp>=2 renamed FastMCP to MCPServer -------------------------------------
try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server, Image  # type: ignore
    from mcp.server.mcpserver.exceptions import ToolError  # type: ignore
except ModuleNotFoundError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image  # type: ignore
    from mcp.server.fastmcp.exceptions import ToolError  # type: ignore

from . import __version__
from . import engine as E

# stdio transport owns stdout: all logging goes to stderr (Claude Desktop writes it to mcp-server-hidim.log)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("hidim.server")

INSTRUCTIONS = """hidim keeps datasets in its own process and returns only budgeted summaries, projections, slices and PNG pictures.
Typical flow: load_dataset (or make_synthetic) -> describe -> project (pca/tsne/umap/isomap...) -> cluster -> plot(kind='scatter', color_by=...).
For long/tidy tables with several key columns (e.g. date, strike, maturity): tensorize -> describe -> slice_tensor (select/reduce) -> plot(kind='tensor'|'surface'|'lines').
Never try to read a whole dataset; use subset/neighbors/slice_tensor to read specific rows or cells. Text results are capped (default 24k chars), tables at 200 rows x 60 cols.
Files must live under HIDIM_ROOTS (default: the home directory). Plots are also saved as PNG under HIDIM_OUT (default ~/hidim_out)."""


def _make_server() -> Any:
    kwargs: dict[str, Any] = {"instructions": INSTRUCTIONS}
    try:
        return _Server("hidim", version=__version__, **kwargs)
    except TypeError:  # mcp 1.x FastMCP has no version kwarg
        return _Server("hidim", **kwargs)


mcp = _make_server()


def _run(fn, *args, **kwargs):
    """Translate engine errors into ToolError so the model sees the message instead of a stack trace."""
    try:
        return fn(*args, **kwargs)
    except E.EngineError as exc:
        raise ToolError(str(exc)) from exc
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        log.exception("tool failure")
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc


# ------------------------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------------------------


@mcp.tool()
def list_datasets() -> str:
    """List datasets currently held in memory: name, kind (table|tensor), shape, memory, parent, how they were made."""
    return E.REGISTRY.listing()


@mcp.tool()
def load_dataset(path: str, name: str | None = None, format: str | None = None,
                 options: dict[str, Any] | None = None) -> str:
    """Load a file into memory as a table (rows x columns) or tensor (N-d labelled array). Returns a schema summary, never the data.

    Formats by extension (or `format`): csv tsv txt parquet feather json jsonl xlsx npy npz mat h5/hdf5 nc zarr.
    options: pandas/reader keyword arguments, e.g. {"sep": ";", "index_col": 0, "parse_dates": ["date"]}, {"sheet": "Sheet2"},
    {"key": "array_name"} for npz/mat/h5/nc with several arrays. Relative paths resolve against the first HIDIM_ROOTS entry (home by default).
    npz files written by save_dataset come back as tensors with their dims and coordinates intact.
    """
    entries = _run(E.load_dataset, path, name, format, options)
    parts = []
    for e in entries:
        if e.kind == "table":
            df = e.obj
            dtypes = df.dtypes.astype(str).value_counts().to_dict()
            cols = list(df.columns)
            parts.append(f"loaded table '{e.name}': {df.shape[0]:,} rows x {df.shape[1]} cols, {E._fmt_bytes(e.nbytes())}; dtypes {dtypes}\n"
                         f"columns: {cols[:60]}{' ...' if len(cols) > 60 else ''}")
        else:
            t = e.obj
            parts.append(f"loaded tensor '{e.name}': shape {'x'.join(map(str, t.shape))}, dims {t.dims}, {E._fmt_bytes(e.nbytes())}")
    parts.append(f"next: describe('{entries[0].name}')")
    return E.clip("\n".join(parts))


@mcp.tool()
def save_dataset(name: str, path: str, format: str | None = None) -> str:
    """Write a dataset to disk under HIDIM_ROOTS. Tables: csv parquet feather json npy xlsx mat. Tensors: npz (keeps dims+coords, reloadable), npy, nc (needs xarray), mat."""
    return _run(E.save_dataset, name, path, format)


@mcp.tool()
def drop_dataset(name: str) -> str:
    """Remove a dataset from memory."""
    _run(E.REGISTRY.drop, name)
    return f"dropped '{name}'. Remaining: {E.REGISTRY.names()}"


@mcp.tool()
def make_synthetic(kind: str = "blobs", n: int = 2000, d: int = 20, k: int = 4, noise: float = 0.05,
                   seed: int = 0, name: str | None = None) -> str:
    """Create a synthetic dataset for demos and sanity checks.

    kinds: blobs (k Gaussian clusters in d dims, label column), swiss_roll / s_curve (2-D manifold embedded in d dims, colour column t),
    gaussian (iid noise: the null model for the diagnostics), lowrank (rank-k matrix plus noise),
    options_surface (long table date x strike x maturity_days -> iv, spot, volume; for tensorize demos).
    """
    return _run(E.make_synthetic, kind, n, d, k, noise, seed, name)


# ------------------------------------------------------------------------------------------------
# Reading and diagnostics
# ------------------------------------------------------------------------------------------------


@mcp.tool()
def describe(name: str, max_features: int = 40, sample: int = 5000, seed: int = 0) -> str:
    """Budgeted profile of a dataset. For tables: schema, label columns, per-feature stats, missingness, plus high-dimensional
    geometry on a standardized sample: strongest correlations, effective rank, condition number, PCA components needed for
    50/80/90/95/99% variance, intrinsic dimension (TwoNN and MLE), distance-concentration (relative contrast, CV of distances)
    and hubness. For tensors: dims, coordinate ranges, fill rate per axis, value distribution. Ends with a hint on what to do next.
    """
    return _run(E.describe, name, max_features, sample, seed)


@mcp.tool()
def subset(name: str, where: str | None = None, columns: list[str] | str | None = None, top_variance: int | None = None,
           sample: int | None = None, seed: int = 0, new_name: str | None = None, preview: int = 20,
           sort_by: str | None = None, ascending: bool = True) -> str:
    """Filter / select / sample rows and columns of a table; preview a few rows; optionally store the result as new_name.

    where: pandas query string, e.g. "iv > 0.3 and maturity_days <= 30" (backtick-quote odd column names).
    columns: list of names, "re:<regex>", or "top_variance:<k>". top_variance keeps the k highest-variance numeric features.
    This is the only way to read raw rows; preview is capped (200 rows x 60 cols).
    """
    return _run(E.subset, name, where, columns, top_variance, sample, seed, new_name, preview, sort_by, ascending)


@mcp.tool()
def neighbors(name: str, index: str | int | None = None, point: list[float] | None = None, k: int = 10,
              metric: str = "euclidean", features: list[str] | str | None = None, standardize: bool = False,
              columns: list[str] | str | None = None) -> str:
    """k nearest neighbours of one row (index = row label or integer position) or of an explicit point (vector in feature order),
    in the original high-dimensional space. Returns distances, positions and a few columns of each neighbour, plus how those
    distances compare with the distance to a random row (relative contrast). metric: any scikit-learn metric (euclidean, cosine, manhattan, ...).
    """
    return _run(E.neighbors, name, index, point, k, metric, features, standardize, columns)


# ------------------------------------------------------------------------------------------------
# Projections and clustering
# ------------------------------------------------------------------------------------------------


@mcp.tool()
def project(name: str, method: str = "pca", n_components: int = 2, features: list[str] | str | None = None,
            standardize: bool = True, sample: int | None = None, seed: int = 0, params: dict[str, Any] | None = None,
            new_name: str | None = None, pre_pca: int | None = None, ascii: bool = False) -> str:
    """Reduce a table's numeric features to n_components and store the embedding as a new table (name__method<k>) with
    label columns carried over for colouring. Returns quality metrics, not coordinates.

    methods: pca (also returns explained variance and top loadings), svd, ica, random (Gaussian random projection), kpca,
    tsne, umap (needs umap-learn), isomap, lle, mds, spectral. Non-linear methods run on a random sample (default caps:
    tsne 5000, umap 50000, isomap/lle/spectral 5000, mds 2000; override with sample) and get an automatic PCA to 50 dims first
    when there are more than 50 features (pre_pca). params passes method-specific keywords (perplexity, n_neighbors, min_dist, kernel, ...).
    Every method reports trustworthiness (neighbourhood preservation, 1.0 = perfect). ascii=True adds a coarse text density map.
    """
    return _run(E.project, name, method, n_components, features, standardize, sample, seed, params, new_name, pre_pca, ascii)


@mcp.tool()
def cluster(name: str, method: str = "kmeans", k: int | None = None, features: list[str] | str | None = None,
            standardize: bool = True, seed: int = 0, params: dict[str, Any] | None = None,
            label_col: str | None = None) -> str:
    """Cluster a table (or an embedding) and add a label column (cluster_<method>) usable as color_by in plot. If run on an
    embedding, labels are also written back to the parent table. methods: kmeans (k auto by silhouette if omitted), gmm (k auto by BIC),
    dbscan (eps auto from 5-NN distances), hdbscan, agglomerative. Returns sizes, silhouette, Davies-Bouldin and each
    cluster's centroid signature (the features most displaced from the global mean, in SD units).
    """
    return _run(E.cluster, name, method, k, features, standardize, seed, params, label_col)


# ------------------------------------------------------------------------------------------------
# Tensors
# ------------------------------------------------------------------------------------------------


@mcp.tool()
def tensorize(name: str, axes: list[str], values: list[str] | None = None, agg: str = "mean",
              new_name: str | None = None) -> str:
    """Condense a long/tidy table into a labelled N-d tensor: each column in `axes` becomes a dimension (sorted unique values are
    the coordinates) and each value column fills the cells; several value columns add a trailing 'variable' dimension.
    Duplicate axis combinations are aggregated with agg (mean|sum|first|last|max|min|median|count); missing combinations become NaN.
    Example: tensorize('opts', axes=['date','strike','maturity_days'], values=['iv']) -> a date x strike x maturity volatility cube.
    """
    return _run(E.tensorize, name, axes, values, agg, new_name)


@mcp.tool()
def tensor_from_wide(name: str, sep: str = "_", dims: list[str] | None = None, index_dim: str | None = None,
                     new_name: str | None = None) -> str:
    """Turn a wide table whose column names encode several keys (AAPL_close, AAPL_volume, MSFT_close, ...) into a tensor
    index x dims..., splitting column names on sep. dims default to ['series', 'feature'].
    """
    return _run(E.tensor_from_wide, name, sep, dims or ["series", "feature"], index_dim, new_name)


@mcp.tool()
def stack_datasets(names: list[str], axis_name: str = "source", labels: list[str] | None = None,
                   new_name: str | None = None) -> str:
    """Stack several same-shaped tables (their numeric blocks) or tensors along a new leading dimension, e.g. one CSV per
    ticker -> ticker x row x column. Shapes and column sets must match exactly (use subset/tensorize to align first).
    """
    return _run(E.stack_datasets, names, axis_name, labels, new_name)


@mcp.tool()
def slice_tensor(name: str, select: dict[str, Any] | None = None, reduce: dict[str, str] | None = None,
                 new_name: str | None = None, max_cells: int | None = None) -> str:
    """Read part of a tensor. select maps dim -> scalar (exact, or nearest for numeric/datetime axes), list (several values),
    {"range": [lo, hi]}, {"isin": [...]}, {"nearest": v}, {"iloc": [start, stop]}, {"positions": [...]}, {"every": n}.
    reduce maps dim -> mean|std|var|sum|min|max|median|count|first|last (NaN-aware). Scalar selections drop the dim.
    When 1 or 2 dims remain the values are returned as a table (evenly thinned if over max_cells, default 4000);
    with more dims you get summary stats and should select/reduce further. new_name stores the result (2-D or less becomes a table
    that describe/project/plot accept; otherwise a tensor).
    """
    return _run(E.slice_tensor, name, select, reduce, new_name, max_cells)


# ------------------------------------------------------------------------------------------------
# Pictures
# ------------------------------------------------------------------------------------------------


@mcp.tool()
def plot(name: str, kind: str = "scatter", x: str | None = None, y: str | None = None, z: str | None = None,
         color_by: str | None = None, features: list[str] | str | None = None, select: dict[str, Any] | None = None,
         reduce: dict[str, str] | None = None, series_dim: str | None = None, sample: int | None = None, seed: int = 0,
         title: str | None = None, width: float | None = None, height: float | None = None, log_color: bool = False) -> list:
    """Render a PNG picture (returned as an image you can look at; also saved under HIDIM_OUT) with a one-line caption.

    Tables: scatter (x,y[,z]; defaults to the first components of an embedding; color_by a label/cluster/numeric column, parent
    columns of an embedding work too), pairs (scatter matrix of <=6 features), parallel (parallel coordinates, min-max scaled),
    corr (correlation heatmap), heatmap (raw value matrix rows x columns, e.g. a stored 2-D slice), scree (PCA variance curve), hist (small-multiple histograms), andrews (Andrews curves by class).
    Tensors: tensor (heatmap of a 2-D slice), surface (3-D surface of a 2-D slice), lines (1-D slices over one dim, up to 8 series
    from series_dim). Use select/reduce exactly as in slice_tensor to get down to 2 dims (or 1-2 for lines).
    Large tables are randomly sampled to HIDIM_PLOT_MAX_POINTS (20000) points; categorical colours beyond 8 classes fold into 'Other'.
    """
    png, caption = _run(E.plot, name, kind, x, y, z, color_by, features, select, reduce, series_dim, sample, seed,
                        title, width, height, log_color)
    return [Image(data=png, format="png"), caption]


@mcp.resource("hidim://datasets")
def datasets_resource() -> str:
    """Current in-memory datasets."""
    return E.REGISTRY.listing()


def main() -> None:
    log.info("hidim-mcp %s starting (roots=%s, out=%s)", __version__, [str(r) for r in E.roots()], E.out_dir())
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
