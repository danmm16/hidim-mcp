# hidim-mcp

A local MCP server ("connector") that lets Claude work with high-dimensional data **without the data ever entering the conversation**.

The model cannot read a 10,000 x 512 matrix; it reads text and images. So the data stays inside this server process and Claude gets back only what fits through the straw:

| Claude asks for | hidim returns |
|---|---|
| "what is this dataset?" | schema, per-feature stats, missingness, effective rank, PCA variance curve, intrinsic dimension (TwoNN + MLE), distance-concentration and hubness diagnostics |
| "show me the structure" | an embedding (PCA / t-SNE / UMAP / Isomap / LLE / MDS / spectral / kernel PCA / random projection) stored server-side, with trustworthiness and explained variance, then a PNG scatter it can actually look at |
| "which groups are there?" | k-means / GMM / DBSCAN / HDBSCAN / agglomerative labels, silhouette, centroid signatures |
| "what is near this point?" | k nearest neighbours with distances and context |
| "turn this long table into a cube" | a labelled N-d tensor (e.g. date x strike x maturity), readable through budgeted slices and reductions, plottable as heatmap / 3-D surface / lines, saved and reloaded as `.npz` with dims and coordinates intact |

Every text result is capped (24k chars by default; tables 200 rows x 60 cols x 4000 cells) so a careless call cannot flood the context.

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
| `describe(name)` | the diagnostic profile described above (tables and tensors) |
| `subset(name, where, columns, top_variance, sample, new_name)` | pandas-query filter, column selection, sampling; the only way to read raw rows |
| `neighbors(name, index | point, k, metric)` | kNN in the original space with distance context |
| `project(name, method, n_components, ...)` | pca svd ica random kpca tsne umap isomap lle mds spectral -> stored embedding + metrics |
| `cluster(name, method, k, ...)` | kmeans gmm dbscan hdbscan agglomerative -> label column + metrics |
| `tensorize(name, axes, values, agg)` | long table -> labelled N-d tensor |
| `tensor_from_wide(name, sep, dims)` | `AAPL_close, MSFT_close, ...` columns -> index x series x feature tensor |
| `stack_datasets(names, axis_name)` | same-shaped tables/tensors -> one tensor with a new leading axis |
| `slice_tensor(name, select, reduce, new_name)` | read cells: exact / nearest / range / isin / positions / stride selection, NaN-aware reductions |
| `plot(name, kind, ...)` | PNG returned inline: scatter (2-D/3-D), pairs, parallel, corr, heatmap, scree, hist, andrews; tensor, surface, lines |

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
plot("q__tensor", kind="surface", select={"variable":"iv","date":"2024-03-15"})
plot("q__tensor", kind="lines", select={"variable":"iv","strike":{"nearest":100}}, series_dim="expiry")
save_dataset("q__tensor", "~/data/iv_cube.npz")          -> reloadable with load_dataset
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
November 2026, then run end to end over stdio against Claude Desktop. The intrinsic-dimension estimators follow Facco et al.
(2017, TwoNN) and Levina & Bickel (2004, MLE); the concentration diagnostic follows Beyer et al. (1999); hubness follows
Radovanovic et al. (2010); trustworthiness is scikit-learn's implementation of Venna & Kaski (2001).

## Development

```bash
python tests/stdio_smoke.py     # speaks JSON-RPC over stdio to a real server process and exercises every tool
```

The engine (`hidim_mcp/engine.py`) has no MCP dependency and can be driven directly from Python or a notebook.
