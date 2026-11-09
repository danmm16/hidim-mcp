# hidim-mcp

A local MCP server ("connector") that lets Claude work with high-dimensional data **without the data ever entering the conversation**.

The model cannot read a 10,000 x 512 matrix; it reads text and images. So the data stays inside this server process and Claude gets back only what fits through the straw:

| Claude asks for | hidim returns |
|---|---|
| "what is this dataset?" | schema, per-feature stats, missingness, identifier / label-like columns, duplicated features, effective rank, PCA variance curve on both raw and standardized scales, the number of factors above the Marchenko-Pastur noise edge, intrinsic dimension (TwoNN + MLE), distance-concentration and hubness diagnostics, and one consistent hint |
| "clean it up first" | `transform` (log1p, z-score with clipping, winsorize, returns, deseasonalize, residualize, row normalization, ...) and `tensor_transform` (anomalies / climatology, area weights, differences) as new datasets, nothing in place |
| "show me the structure" | an embedding (PCA / t-SNE / UMAP / Isomap / LLE / MDS / spectral / kernel PCA / random projection) stored server-side, with trustworthiness and explained variance, then a PNG scatter it can actually look at |
| "which groups are there?" | k-means / GMM / DBSCAN / HDBSCAN / agglomerative labels, silhouette, centroid signatures |
| "what is near this point?" | k nearest neighbours with distances and context |
| "turn this long table into a cube" | a labelled N-d tensor (e.g. date x strike x maturity), readable through budgeted slices and reductions, plottable as heatmap / 3-D surface / lines, saved and reloaded as `.npz` with dims and coordinates intact |

Every text result is capped (24k chars by default; tables 200 rows x 60 cols x 4000 cells) so a careless call cannot flood the context.

## Scaling: the one default that matters

Most tools take `standardize`, and since 0.2.0 it defaults to `"auto"`:

- data whose columns share one unit (pixel intensities, read counts, [0, 1] images, spectra) is analysed on its **raw** scale.
  Z-scoring such data turns near-constant columns (border pixels, rare genes) into dominant features, inflates the
  condition number and roughly doubles the PCA rank (MNIST: 183 components for 90% standardized vs 86 raw);
- heterogeneous columns (prices next to volumes next to ratios) are **standardized**;
- every result says which scaling it used, `describe` prints the PCA counts on *both* scales, and `clip_z=10` clips
  z-scores so that sparse or near-constant features cannot hijack k-means or kNN (`True` / `False` force either choice).

`describe` also reports how many correlation eigenvalues sit above the Marchenko-Pastur noise edge. When that number is
small and the "components for 90% variance" count is large, the extra rank is noise (a return panel has 3-4 real factors
and 30 noise eigenvalues), and the hint says so instead of recommending t-SNE on noise.

## Install

Python 3.10+. Required: `mcp`, numpy, pandas, scipy, scikit-learn, matplotlib. Optional: `umap-learn` (UMAP), `pyarrow` (parquet/feather), `openpyxl` (xlsx), `h5py` (HDF5), `xarray` + `netCDF4` (NetCDF/Zarr).

```bash
cd hidim-mcp
python3 -m venv .venv && source .venv/bin/activate        # or: uv venv && source .venv/bin/activate
pip install -e ".[all]"                                     # or: pip install -e .   (core only)
python -m hidim_mcp.server --help 2>/dev/null; python tests/stdio_smoke.py   # end-to-end check (~20 s)
```

Works with both `mcp` 1.x (`FastMCP`) and 2.x (`MCPServer`).

## Connect to Claude Desktop

The short way (merges the entry into your existing config, backs the file up first, refuses to touch invalid JSON):

```bash
.venv/bin/python register.py                       # add --roots ~/data:~/research --out ~/hidim_out to change defaults
.venv/bin/python register.py --dry-run             # preview; --remove takes it out again
```

The manual way:

1. Claude menu (menu bar) -> **Settings...** -> **Developer** -> **Edit Config**. On macOS this reveals
   `~/Library/Application Support/Claude/claude_desktop_config.json` in Finder (Windows: `%APPDATA%\Claude\claude_desktop_config.json`);
   open it in a plain-text editor (straight quotes only; TextEdit's smart quotes break JSON).
2. Merge the `"hidim"` entry into the existing `mcpServers` object (do not paste a second `mcpServers` key). Use **absolute paths**;
   the interpreter must be the one where you installed the package.

```json
{
  "mcpServers": {
    "hidim": {
      "command": "/Users/you/hidim-mcp/.venv/bin/python",
      "args": ["-m", "hidim_mcp.server"],
      "env": {
        "HIDIM_ROOTS": "/Users/you/data:/Users/you/research",
        "HIDIM_OUT": "/Users/you/hidim_out"
      }
    }
  }
}
```

3. Quit Claude Desktop completely (Cmd+Q) and reopen it. The tools appear under the "+" menu -> Connectors -> hidim.
4. If it does not connect: `tail -f ~/Library/Logs/Claude/mcp-server-hidim.log` shows the server's stderr, and running the
   `command` + `args` by hand in a terminal shows import errors. The config file must be valid JSON.

## Connect to Claude Code

```bash
claude mcp add --transport stdio --scope user hidim \
  -e HIDIM_ROOTS=/Users/you/data -e HIDIM_OUT=/Users/you/hidim_out \
  -- /Users/you/hidim-mcp/.venv/bin/python -m hidim_mcp.server
```

Everything after `--` is the server command. Claude Code warns above 10k tokens of tool output and truncates at 25k by default;
hidim's own caps keep well under that.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `HIDIM_ROOTS` | home directory | `os.pathsep`-separated directories the server may read from / write to |
| `HIDIM_OUT` | `~/hidim_out` | where PNGs and exports go (always allowed) |
| `HIDIM_MAX_CHARS` | 24000 | cap on any text result |
| `HIDIM_MAX_ROWS` / `HIDIM_MAX_COLS` / `HIDIM_MAX_CELLS` | 200 / 60 / 4000 | table rendering caps |
| `HIDIM_MAX_TENSOR_BYTES` | 2e9 | refuse to build larger tensors |
| `HIDIM_PLOT_MAX_POINTS` | 20000 | scatter/pairs sample size |
| `HIDIM_MAX_CLASSES` | 20 | legend entries before rarer classes fold into 'Other' |
| `HIDIM_MAX_ANNOTATIONS` | 300 | points labelled by `plot(annotate=...)` |
| `HIDIM_MAX_PNG_BYTES` | 700000 | re-render at lower dpi above this |
| `HIDIM_ALLOW_PICKLE` | unset | set to `1` to allow `.pkl` / pickled `.npy` (can execute code: only for your own files) |

## Tools

| Tool | What it does |
|---|---|
| `list_datasets` | what is in memory (name, kind, shape, memory, parent) |
| `load_dataset(path, name, format, options)` | csv tsv txt parquet feather json jsonl xlsx npy npz mat h5 nc zarr -> table or tensor; returns only a schema |
| `save_dataset(name, path, format)` | tables: csv parquet feather json npy xlsx mat; tensors: npz (round-trips dims+coords), npy, nc, mat |
| `drop_dataset(name)` | free memory |
| `make_synthetic(kind, n, d, k, noise, seed)` | blobs, swiss_roll, s_curve, gaussian, lowrank, options_surface |
| `describe(name, standardize="auto")` | the diagnostic profile described above (tables and tensors) |
| `set_labels(name, columns)` | declare integer class codes / years / hours / lat-lon / ids as labels so they leave the feature matrix |
| `transform(name, op, columns, params)` | log1p, log, sqrt, zscore (clip), minmax, clip, winsorize, rank, demean, detrend, diff, pct_change, log_return, cumsum, row_normalize, deseasonalize, residualize, dropna -> new table |
| `subset(name, where, columns, top_variance, sample, new_name)` | pandas-query filter, column selection, sampling; the only way to read raw rows |
| `neighbors(name, index | point, k, metric)` | kNN in the original space with distance context |
| `project(name, method, n_components, standardize, clip_z, ...)` | pca svd ica random kpca tsne umap isomap lle mds spectral -> stored embedding + metrics |
| `loadings(name, component, top, all_features)` | full loading vector of a PCA component (level / slope / curvature shapes) |
| `cluster(name, method, k, standardize, clip_z, k_max)` | kmeans gmm dbscan hdbscan agglomerative -> label column + metrics; auto-k shows the whole silhouette-by-k table |
| `tensorize(name, axes, values, agg, order, sort)` | long table -> labelled N-d tensor; natural sort of text axes, explicit `order` per axis |
| `tensor_from_wide(name, sep, dims, sort, order)` | `AAPL_close, MSFT_close, ...` columns -> index x series x feature tensor, column order kept |
| `tensor_transform(name, op, dim, params)` | log1p, demean, zscore, diff, pct_change, log_return, cumsum, normalize, anomaly (climatology), area_weight, fillna -> new tensor |
| `tensor_to_table(name, row_dim)` | flatten time x lat x lon into time rows x gridpoint columns for PCA / EOF analysis |
| `stack_datasets(names, axis_name)` | same-shaped tables/tensors -> one tensor with a new leading axis |
| `slice_tensor(name, select, reduce, new_name)` | read cells: exact / nearest / range / isin / positions / stride selection, NaN-aware reductions |
| `plot(name, kind, ..., annotate, order, log_axes)` | PNG returned inline: scatter (2-D/3-D, legends up to 20 classes, colour bars for numeric and datetime columns, point labels), pairs, parallel, corr (labelled, clustered order), heatmap, scree, hist, andrews; tensor, surface (true axis values), lines |

### A typical session

```
make_synthetic(kind="blobs", n=5000, d=128, k=6)        -> 'synthetic_blobs'
describe("synthetic_blobs")                              -> rank 5 explains 90%, intrinsic dim ~40, hubness high ...
project("synthetic_blobs", method="tsne")                -> 'synthetic_blobs__tsne2', trustworthiness 0.96
cluster("synthetic_blobs__tsne2", method="hdbscan")      -> 6 clusters, silhouette 0.81, labels written back to parent
plot("synthetic_blobs__tsne2", kind="scatter", color_by="cluster_hdbscan")   -> picture
neighbors("synthetic_blobs", index=17, k=5)              -> who sits next to row 17 in 128-D

load_dataset("~/data/option_quotes.parquet", name="q")
tensorize("q", axes=["date","strike","expiry"], values=["iv","volume"])      -> 'q__tensor' date x strike x expiry x variable
slice_tensor("q__tensor", select={"variable":"iv","date":"2024-03-15"})      -> the surface for one day as a table
plot("q__tensor", kind="surface", select={"variable":"iv","date":"2024-03-15"}, log_axes=["expiry"])
plot("q__tensor", kind="lines", select={"variable":"iv","strike":{"nearest":100}}, series_dim="expiry")
tensor_transform("q__tensor", op="diff", dim="date", new_name="dq")          -> daily surface changes
tensor_to_table("dq", row_dim="date", select={"variable":"iv"}, new_name="dq_rows")
project("dq_rows", method="pca", n_components=3, standardize=False)          -> the level / term / skew modes of the surface
save_dataset("q__tensor", "~/data/iv_cube.npz")          -> reloadable with load_dataset

load_dataset("~/data/sst.mnmean.nc", name="sst", options={"key": "sst"})     -> tensor time x lat x lon
tensor_transform("sst", op="anomaly", dim="time", params={"period": "month"}) -> seasonal cycle removed
tensor_transform("sst__anomaly", op="area_weight", dim="lat")                 -> cos(lat)^0.5 weights
tensor_to_table("sst__anomaly__area_weight", row_dim="time", select={"lat": {"range": [-30, 30]}, "lon": {"range": [120, 290]}})
project(<that table>, method="pca", n_components=3, standardize=False)        -> EOF1 is ENSO
```

## What it deliberately does not do

- It never returns a whole dataset. `subset` previews are capped; `slice_tensor` thins anything over the cell budget and says so.
- Memory lives in the server process. Restarting Claude Desktop restarts the server and empties it: `save_dataset` what you want to keep.
- t-SNE, Isomap, LLE, spectral and MDS run on random samples by default (they are O(n^2)); PCA, SVD, UMAP and random projection scale to the full table.
- It is a local tool for your own data. `subset(where=...)` is evaluated by pandas' query engine, which is an expression language,
  not a security sandbox; the model is the caller, so do not expose this server over HTTP to clients you do not trust, and keep
  `HIDIM_ROOTS` to directories you are happy to have read and written.

## Provenance

Idea and direction: Daniel Margolis. Design and implementation: written with Claude (Anthropic) in a single working session,
November 2026, then run end to end over stdio against Claude Desktop. Version 0.2.0 followed a test battery of eleven real
datasets with published answers (Treasury yield curve, SPX/SPY option surfaces, Fama-French 49 industries, sklearn digits
and MNIST, UCI HAR, PBMC 3k single-cell RNA-seq, GloVe word vectors, Golub leukemia and Olivetti faces, Indian Pines and
Pavia University hyperspectral cubes, NOAA ERSST sea-surface temperature, NYC taxi trips); the numbers matched independent
scikit-learn runs everywhere, and every defect that battery exposed in the defaults, hints and plots is fixed here
(see CHANGELOG.md). The intrinsic-dimension estimators follow Facco et al.
(2017, TwoNN) and Levina & Bickel (2004, MLE); the concentration diagnostic follows Beyer et al. (1999); hubness follows
Radovanovic et al. (2010); trustworthiness is scikit-learn's implementation of Venna & Kaski (2001).

## Development

```bash
python -m pytest tests -q        # unit tests of the engine (scaling, Marchenko-Pastur, hints, transforms, ordering, plots)
python tests/stdio_smoke.py     # speaks JSON-RPC over stdio to a real server process and exercises every tool
```

The engine (`hidim_mcp/engine.py`) has no MCP dependency and can be driven directly from Python or a notebook.
