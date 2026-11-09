# Changelog

## 0.2.1 (2026-11-09)

Polish from running 0.2.0 live against the morning's datasets.
- Measurement-channel families (px000.., b001.., g0042..: 20+ columns sharing a stem) and constant columns are no
  longer flagged as "possible label" columns.
- Long lists in notes are shortened (first 8 + count); `project` prints ranges for at most 10 components.
- New `transform` op `center_rows`: subtract each row's cross-sectional mean (equal-weight market removal for panels).

## 0.2.0 (2026-11-09)

Fixes for every defect found by the eleven-dataset test battery (see README, Provenance).

### Scaling (the big one)
- `standardize` defaults to `"auto"` in `describe`, `project` and `cluster`: data in one shared unit (pixels, counts,
  [0, 1] images) stays on its raw scale; heterogeneous columns are z-scored. Every result states the scaling used.
- `describe` prints the PCA component counts on both scales, so a rank that depends on scaling is visible.
- New `clip_z` on `project` / `cluster` / `_matrix`: clip z-scores (10 is the usual value) so near-constant or sparse
  features cannot dominate distances; a note warns when standardization produced |z| > 10 without clipping.
- `cluster` warns when clusters hold < 0.5% of the rows (outliers, not groups) and points at `clip_z`.

### Diagnostics and hints
- Marchenko-Pastur check: number of correlation eigenvalues above the noise edge, with the edge and p/n.
- The hint line is a decision tree that produces one main recommendation: factor + noise structure, low rank,
  curved manifold, too few rows, moderate / high rank, pure noise. It no longer prints contradictory hints, no longer
  recommends t-SNE/UMAP on noise-rank data or on < 200 rows, and `project` warns on tiny non-linear runs.
- Exact duplicate feature pairs are reported alongside constant columns.
- k-means auto-k prints the whole silhouette-by-k table, the Calinski-Harabasz choice and the finest k within 10% of
  the best silhouette, and searches 2..`k_max` (default 12) instead of 2..10.

### Features vs identifiers
- Integer columns that are unique per row or named like identifiers (`id`, `index`, `row`, `col`, `*_id`, `*ID`) are
  skipped automatically and reported; integer columns that look like class codes or coordinates (few distinct values,
  or named year/hour/lat/lon/...) are reported as suspects.
- New `set_labels(name, columns)` declares label / coordinate columns explicitly (and `unset=True` reverses it).

### Preprocessing
- New `transform` for tables: log1p, log, sqrt, abs, zscore (clip), minmax, clip, winsorize, rank, demean, detrend,
  diff / pct_change / log_return / cumsum (by time, per group), row_normalize, center_rows, deseasonalize, residualize, dropna.
- New `tensor_transform` for tensors: log1p, log, sqrt, abs, fillna, demean, zscore, diff, pct_change, log_return,
  cumsum, normalize, anomaly (per-period climatology, optional baseline window), area_weight.
- New `tensor_to_table(row_dim)`: flatten a tensor into rows x (other dims) for PCA / EOF analysis.
- New `loadings(embedding, component)`: full PCA loading vectors; `project` prints loadings for at most 10 components.

### Coordinates
- `tensorize` sorts text axes naturally (b2 < b10), accepts `order={dim: [...]}` and `sort=False`.
- `tensor_from_wide` keeps the wide table's column order for name-derived dims (tenors stay 1m, 3m, 6m, 1y, ...).
- Embeddings now carry datetime columns and all label columns (up to 12), regardless of cardinality.

### Plots
- Legends show up to 20 classes (`HIDIM_MAX_CLASSES`) before folding into 'Other'.
- Datetime `color_by` gets a continuous colour bar with date ticks (pandas 3 microsecond datetimes handled).
- `annotate=<column>` writes point labels (up to `HIDIM_MAX_ANNOTATIONS`, default 300).
- `corr` heatmaps are labelled up to 120 features and reordered by hierarchical clustering (`order="given"` keeps the
  table order).
- `surface` draws numeric and datetime axes at their true values; `log_axes=[dim]` uses a log10 scale.
- `lines` draws up to 12 series; `scree` follows the auto scaling and says so.

### Housekeeping
- `tests/test_engine.py` (pytest, no network) covers the above; the stdio smoke test exercises the new tools.

## 0.1.1 (2026-11-09)
- First public version: registry, loaders, describe, project, cluster, neighbors, subset, tensorize, slice, plot.
