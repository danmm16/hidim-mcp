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
Typical flow: load_dataset (or make_synthetic) -> describe -> [transform] -> project (pca/tsne/umap/isomap...) -> cluster -> plot(kind='scatter', color_by=...).
For long/tidy tables with several key columns (e.g. date, strike, maturity): tensorize -> describe -> [tensor_transform] -> slice_tensor (select/reduce) -> plot(kind='tensor'|'surface'|'lines');
tensor_to_table flattens a tensor (time x lat x lon -> time x gridpoints) for PCA/EOF.
Scaling: describe/project/cluster default to standardize='auto', which keeps RAW scales for data in one shared unit (pixels, counts, [0,1] images)
and z-scores heterogeneous columns; every result says which was used. Use clip_z=10 when z-scoring sparse or near-constant features.
describe reports the number of factors above the Marchenko-Pastur noise edge: when it is small and the PCA count for 90% is large, the extra rank is noise.
Integer id-like columns are skipped automatically; declare class codes / coordinates with set_labels so they are not treated as features.
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
def describe(name: str, max_features: int = 40, sample: int = 5000, seed: int = 0, standardize: bool | str = "auto") -> str:
    """Budgeted profile of a dataset. For tables: schema, label columns, identifier-like and label-like integer columns, per-feature
    stats, missingness, constant and duplicated features, plus high-dimensional geometry on a sample: strongest correlations,
    effective rank, condition number, PCA components needed for 50/80/90/95/99% variance (on the chosen scaling AND on the other
    one), the number of correlation eigenvalues above the Marchenko-Pastur noise edge (how many factors carry more than noise),
    intrinsic dimension (TwoNN and MLE), distance concentration and hubness. standardize: 'auto' (default) keeps raw scales for
    data in one shared unit such as pixels / counts / [0,1] intensities and z-scores heterogeneous columns; True/False force it.
    For tensors: dims, coordinate ranges, fill rate per axis, value distribution. Ends with one consistent hint on what to do next.
    """
    return _run(E.describe, name, max_features, sample, seed, standardize)


@mcp.tool()
def set_labels(name: str, columns: list[str] | str, unset: bool = False) -> str:
    """Declare columns of a table as labels / coordinates (integer class codes, year, hour, lat, lon, ids): they are excluded from
    the feature matrix used by describe/project/cluster/neighbors and stay available for color_by and annotate. unset=True reverses it."""
    return _run(E.set_labels, name, columns, unset)


@mcp.tool()
def transform(name: str, op: str, columns: list[str] | str | None = None, new_name: str | None = None,
              params: dict[str, Any] | None = None) -> str:
    """Preprocess the numeric features of a table into a new table (labels, ids and datetimes carried over unchanged).

    ops: log1p, log (params offset), sqrt, abs, clip (lo/hi or z), winsorize (q), rank, zscore (params clip), minmax, demean,
    detrend (params by), diff / pct_change / log_return / cumsum along rows (params periods, by=<time column>, group_by, dropna),
    row_normalize (params target; library-size normalization for counts), center_rows (subtract each row's mean: removes the
    equal-weight market / common mode of a panel), deseasonalize (params by=<datetime column>,
    period=month|dayofweek|hour|dayofyear|quarter: subtracts per-period means -> anomalies), residualize (params on=[cols]:
    regress every feature on those columns and keep residuals, e.g. remove a market factor), dropna.
    Typical chains: counts -> row_normalize -> log1p -> project(clip_z=10); prices -> log_return -> describe; monthly
    series -> deseasonalize -> project(pca, standardize=False).
    """
    return _run(E.transform, name, op, columns, new_name, params)


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
              metric: str = "euclidean", features: list[str] | str | None = None, standardize: bool | str = False,
              columns: list[str] | str | None = None) -> str:
    """k nearest neighbours of one row (index = row label or integer position) or of an explicit point (vector in feature order),
    in the original high-dimensional space. Returns distances, positions and a few columns of each neighbour, plus how those
    distances compare with the distance to a random row (relative contrast). metric: any scikit-learn metric (euclidean, cosine, manhattan, ...);
    cosine is usually the right choice for embeddings and other hub-prone spaces. standardize: False (default), True or 'auto'.
    """
    return _run(E.neighbors, name, index, point, k, metric, features, standardize, columns)


# ------------------------------------------------------------------------------------------------
# Projections and clustering
# ------------------------------------------------------------------------------------------------


@mcp.tool()
def project(name: str, method: str = "pca", n_components: int = 2, features: list[str] | str | None = None,
            standardize: bool | str = "auto", sample: int | None = None, seed: int = 0, params: dict[str, Any] | None = None,
            new_name: str | None = None, pre_pca: int | None = None, ascii: bool = False, clip_z: float | None = None) -> str:
    """Reduce a table's numeric features to n_components and store the embedding as a new table (name__method<k>) with
    label, datetime and cluster columns carried over for colouring. Returns quality metrics, not coordinates.

    methods: pca (explained variance, top loadings; full vectors via loadings()), svd, ica, random (Gaussian random projection), kpca,
    tsne, umap (needs umap-learn), isomap, lle, mds, spectral. Non-linear methods run on a random sample (default caps:
    tsne 5000, umap 50000, isomap/lle/spectral 5000, mds 2000; override with sample) and get an automatic PCA to 50 dims first
    when there are more than 50 features (pre_pca). params passes method-specific keywords (perplexity, n_neighbors, min_dist, kernel, metric, ...).
    standardize: 'auto' (default; raw scales for pixel/count/[0,1] data, z-scores otherwise), True or False. clip_z=10 clips
    z-scores so near-constant or sparse features cannot dominate. Every method reports trustworthiness (1.0 = neighbourhoods
    perfectly preserved). ascii=True adds a coarse text density map.
    """
    return _run(E.project, name, method, n_components, features, standardize, sample, seed, params, new_name, pre_pca, ascii, clip_z)


@mcp.tool()
def loadings(name: str, component: int = 1, top: int = 15, all_features: bool = False) -> str:
    """Loading vector of one principal component of a PCA embedding made by project(method='pca'): the strongest positive and
    negative features, or the whole vector in feature order (all_features=True) to read shapes such as level/slope/curvature."""
    return _run(E.loadings, name, component, top, all_features)


@mcp.tool()
def cluster(name: str, method: str = "kmeans", k: int | None = None, features: list[str] | str | None = None,
            standardize: bool | str = "auto", seed: int = 0, params: dict[str, Any] | None = None,
            label_col: str | None = None, clip_z: float | None = None, k_max: int = 12) -> str:
    """Cluster a table (or an embedding) and add a label column (cluster_<method>) usable as color_by in plot. If run on an
    embedding, labels are also written back to the parent table. methods: kmeans (k omitted -> silhouette over 2..k_max, with the
    whole silhouette-by-k table, the Calinski-Harabasz choice and the finest k within 10% of the best, because silhouette
    favours coarse splits), gmm (k by BIC), dbscan (eps from 5-NN distances), hdbscan, agglomerative. standardize 'auto'/True/False
    and clip_z as in project. Returns sizes (with a warning when clusters hold <0.5% of rows: outliers, not groups), silhouette,
    Davies-Bouldin and each cluster's centroid signature (features most displaced from the global mean, in SD units).
    """
    return _run(E.cluster, name, method, k, features, standardize, seed, params, label_col, clip_z, k_max)


# ------------------------------------------------------------------------------------------------
# Tensors
# ------------------------------------------------------------------------------------------------


@mcp.tool()
def tensorize(name: str, axes: list[str], values: list[str] | None = None, agg: str = "mean",
              new_name: str | None = None, order: dict[str, list] | None = None, sort: bool = True) -> str:
    """Condense a long/tidy table into a labelled N-d tensor: each column in `axes` becomes a dimension and each value column
    fills the cells; several value columns add a trailing 'variable' dimension. Coordinates: numeric/datetime axes sort
    numerically, text axes sort naturally (b2 before b10); sort=False keeps first-appearance order; order={'tenor': ['1m','3m','1y']}
    pins an explicit order per axis. Duplicate axis combinations are aggregated with agg (mean|sum|first|last|max|min|median|count);
    missing combinations become NaN. Note that a later reduce 'mean' over a dim averages cells, not underlying rows: to get a
    row-weighted mean, tensorize agg='sum' and agg='count' and divide.
    Example: tensorize('opts', axes=['date','strike','maturity_days'], values=['iv']) -> a date x strike x maturity volatility cube.
    """
    return _run(E.tensorize, name, axes, values, agg, new_name, order, sort)


@mcp.tool()
def tensor_from_wide(name: str, sep: str = "_", dims: list[str] | None = None, index_dim: str | None = None,
                     new_name: str | None = None, sort: bool = False, order: dict[str, list] | None = None) -> str:
    """Turn a wide table whose column names encode several keys (AAPL_close, AAPL_volume, MSFT_close, ...) into a tensor
    index x dims..., splitting column names on sep. dims default to ['series', 'feature']. The wide table's column order is
    kept for the name-derived dims (so a tenor axis 1m,3m,6m,1y,2y stays in that order); sort=True sorts naturally instead,
    order={dim: [...]} pins an explicit order.
    """
    return _run(E.tensor_from_wide, name, sep, dims or ["series", "feature"], index_dim, new_name, sort, order)


@mcp.tool()
def tensor_transform(name: str, op: str, dim: str | None = None, new_name: str | None = None,
                     params: dict[str, Any] | None = None) -> str:
    """Preprocess a tensor into a new tensor. Elementwise: log1p, log (params offset), sqrt, abs, fillna (params value, or dim
    for a mean fill). Along dim: demean, zscore, diff / pct_change / log_return (params periods; the axis shrinks), cumsum,
    normalize (params target). anomaly (dim = a datetime axis, params period=month|dayofyear|dayofweek|hour|quarter,
    baseline=[start,end]) subtracts the per-period climatology -> anomalies. area_weight (dim = latitude axis in degrees,
    params power: 0.5 for PCA/EOF, 1 for means) multiplies by cos(lat)^power. Typical EOF chain:
    tensor_transform(op='anomaly', dim='time') -> tensor_transform(op='area_weight', dim='lat') -> tensor_to_table(row_dim='time') -> project(pca, standardize=False).
    """
    return _run(E.tensor_transform, name, op, dim, new_name, params)


@mcp.tool()
def tensor_to_table(name: str, row_dim: str, new_name: str | None = None, select: dict[str, Any] | None = None,
                    reduce: dict[str, str] | None = None) -> str:
    """Flatten a tensor into a table: one row per coordinate of row_dim, one feature column per combination of the other dims
    (time x lat x lon -> time rows x gridpoint columns). All-NaN columns are dropped; the row coordinate becomes a label column.
    select/reduce (as in slice_tensor) are applied first. Then describe/project/cluster it like any table."""
    return _run(E.tensor_to_table, name, row_dim, new_name, select, reduce)


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
         title: str | None = None, width: float | None = None, height: float | None = None, log_color: bool = False,
         annotate: str | None = None, order: str = "cluster", log_axes: list[str] | str | None = None) -> list:
    """Render a PNG picture (returned as an image you can look at; also saved under HIDIM_OUT) with a one-line caption.

    Tables: scatter (x,y[,z]; defaults to the first components of an embedding; color_by a label/cluster/numeric/datetime column,
    parent columns of an embedding work too: categories get a legend of up to 20 classes, numeric and datetime columns a colour bar;
    annotate=<column> writes that column's value next to each point, up to 300 points), pairs (scatter matrix of <=6 features),
    parallel (parallel coordinates, min-max scaled), corr (correlation heatmap with feature labels, reordered by hierarchical
    clustering so correlated blocks are visible; order='given' keeps table order), heatmap (raw value matrix rows x columns),
    scree (PCA variance curve, auto scaling), hist (small-multiple histograms), andrews (Andrews curves by class).
    Tensors: tensor (heatmap of a 2-D slice), surface (3-D surface of a 2-D slice; numeric and datetime axes are drawn at their
    true values, log_axes=['maturity_days'] draws a dim on a log10 scale), lines (1-D slices over one dim, up to 12 series from series_dim; use select={'dim': {'isin': [...]}} to pick).
    Use select/reduce exactly as in slice_tensor to get down to 2 dims (or 1-2 for lines).
    Large tables are randomly sampled to HIDIM_PLOT_MAX_POINTS (20000) points; categorical colours beyond HIDIM_MAX_CLASSES (20) fold into 'Other'.
    """
    png, caption = _run(E.plot, name, kind, x, y, z, color_by, features, select, reduce, series_dim, sample, seed,
                        title, width, height, log_color, annotate, order, log_axes)
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
