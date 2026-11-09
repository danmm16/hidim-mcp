"""Data engine for hidim-mcp.

Pure numpy / pandas / scipy / scikit-learn / matplotlib. No MCP imports, so every
function here can be exercised directly from Python or a test suite.

Design rule: the model never receives raw high-dimensional data. Every function
that returns text goes through `clip()` and every table through the row/col/cell
budgets below. Large objects live in the `Registry` for the life of the process.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("hidim")

# --------------------------------------------------------------------------- #
# Budgets and configuration (environment-overridable)
# --------------------------------------------------------------------------- #


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


MAX_CHARS = _env_int("HIDIM_MAX_CHARS", 24_000)          # hard cap on any text result
MAX_ROWS = _env_int("HIDIM_MAX_ROWS", 200)               # rows of any table shown
MAX_COLS = _env_int("HIDIM_MAX_COLS", 60)                # columns of any table shown
MAX_CELLS = _env_int("HIDIM_MAX_CELLS", 4_000)           # cells of any table shown
MAX_TENSOR_BYTES = _env_int("HIDIM_MAX_TENSOR_BYTES", 2_000_000_000)
PLOT_MAX_POINTS = _env_int("HIDIM_PLOT_MAX_POINTS", 20_000)
ALLOW_PICKLE = os.environ.get("HIDIM_ALLOW_PICKLE") == "1"

SAMPLE_CAPS = {  # default row caps per projection method (runtime protection)
    "pca": 200_000, "svd": 200_000, "ica": 100_000, "random": 500_000, "kpca": 5_000,
    "tsne": 5_000, "umap": 50_000, "isomap": 5_000, "lle": 5_000, "mds": 2_000, "spectral": 5_000,
}


class EngineError(Exception):
    """User-facing error: message is returned to the model verbatim."""


def roots() -> list[Path]:
    raw = os.environ.get("HIDIM_ROOTS")
    if raw:
        return [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]
    return [Path.home().resolve()]


def out_dir() -> Path:
    p = Path(os.environ.get("HIDIM_OUT", Path.home() / "hidim_out")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_path(p: str, must_exist: bool = True) -> Path:
    """Resolve a user path and confirm it sits under an allowed root."""
    path = Path(str(p)).expanduser()
    if not path.is_absolute():
        path = roots()[0] / path
    rp = path.resolve()
    allowed = roots() + [out_dir().resolve()]
    if not any(rp == r or r in rp.parents for r in allowed):
        raise EngineError(
            f"Path {rp} is outside the allowed roots {[str(r) for r in allowed]}. "
            "Set the HIDIM_ROOTS environment variable (os.pathsep-separated) to widen access."
        )
    if must_exist and not rp.exists():
        raise EngineError(f"File not found: {rp}")
    return rp


def clip(text: str, limit: int | None = None) -> str:
    limit = limit or MAX_CHARS
    if len(text) <= limit:
        return text
    return text[: limit - 80].rstrip() + f"\n... [truncated at {limit} chars; narrow the request (fewer rows/columns/features) to see more]"


def _fmt_num(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (np.floating, float)):
        if np.isnan(x):
            return "nan"
        if x == 0:
            return "0"
        ax = abs(x)
        if 1e-3 <= ax < 1e6:
            return f"{x:.6g}"
        return f"{x:.4e}"
    if isinstance(x, (np.datetime64, pd.Timestamp)):
        ts = pd.Timestamp(x)
        return ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.isoformat()
    return str(x)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def frame_to_text(df: pd.DataFrame, max_rows: int | None = None, max_cols: int | None = None,
                  index: bool = True) -> str:
    """Compact, budgeted text rendering of a DataFrame."""
    max_rows = min(max_rows or MAX_ROWS, MAX_ROWS)
    max_cols = min(max_cols or MAX_COLS, MAX_COLS)
    # keep the cell budget too
    if max_rows * max(1, min(df.shape[1], max_cols)) > MAX_CELLS:
        max_rows = max(5, MAX_CELLS // max(1, min(df.shape[1], max_cols)))
    notes = []
    shown = df
    if df.shape[1] > max_cols:
        shown = df.iloc[:, :max_cols]
        notes.append(f"showing {max_cols} of {df.shape[1]} columns")
    if len(shown) > max_rows:
        head = shown.head(max_rows // 2)
        tail = shown.tail(max_rows - max_rows // 2)
        notes.append(f"showing {max_rows} of {len(df)} rows (head and tail)")
        txt = head.to_string(index=index, float_format=lambda v: _fmt_num(v)) + "\n...\n" + \
            tail.to_string(index=index, float_format=lambda v: _fmt_num(v))
    else:
        txt = shown.to_string(index=index, float_format=lambda v: _fmt_num(v))
    if notes:
        txt += "\n[" + "; ".join(notes) + "]"
    return txt


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Tensor:
    """A labelled N-d array: dims are axis names, coords are 1-D label arrays."""
    data: np.ndarray
    dims: list[str]
    coords: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        if len(self.dims) != self.data.ndim:
            raise EngineError(f"Tensor has {self.data.ndim} axes but {len(self.dims)} dim names")
        for i, d in enumerate(self.dims):
            if d not in self.coords:
                self.coords[d] = np.arange(self.data.shape[i])
            if len(self.coords[d]) != self.data.shape[i]:
                raise EngineError(f"coords for '{d}' have length {len(self.coords[d])} != axis size {self.data.shape[i]}")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def sizes(self) -> dict[str, int]:
        return {d: int(self.data.shape[i]) for i, d in enumerate(self.dims)}

    def axis(self, dim: str) -> int:
        if dim not in self.dims:
            raise EngineError(f"Unknown dim '{dim}'. Dims: {self.dims}")
        return self.dims.index(dim)

    def nbytes(self) -> int:
        return int(self.data.nbytes + sum(c.nbytes for c in self.coords.values()))

    def to_xarray(self):
        import xarray as xr  # optional dependency
        return xr.DataArray(self.data, dims=self.dims, coords={d: self.coords[d] for d in self.dims})


@dataclass
class Entry:
    name: str
    kind: str                       # "table" | "tensor"
    obj: Any                        # pd.DataFrame | Tensor
    meta: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.obj.shape)

    def nbytes(self) -> int:
        if self.kind == "table":
            return int(self.obj.memory_usage(deep=True).sum())
        return self.obj.nbytes()

    def derived_cols(self) -> list[str]:
        return list(self.meta.get("derived_cols", []))


class Registry:
    def __init__(self) -> None:
        self._d: dict[str, Entry] = {}

    def names(self) -> list[str]:
        return list(self._d)

    def get(self, name: str) -> Entry:
        if name not in self._d:
            raise EngineError(f"No dataset named '{name}'. Loaded: {self.names() or 'none'}")
        return self._d[name]

    def table(self, name: str) -> Entry:
        e = self.get(name)
        if e.kind != "table":
            raise EngineError(f"'{name}' is a {e.kind}, this operation needs a table (use slice_tensor to turn a tensor slice into a table).")
        return e

    def tensor(self, name: str) -> Entry:
        e = self.get(name)
        if e.kind != "tensor":
            raise EngineError(f"'{name}' is a {e.kind}, this operation needs a tensor (use tensorize to build one).")
        return e

    def put(self, entry: Entry, replace: bool = True) -> Entry:
        if entry.name in self._d and not replace:
            entry.name = self.unique(entry.name)
        self._d[entry.name] = entry
        return entry

    def drop(self, name: str) -> None:
        self.get(name)
        del self._d[name]

    def unique(self, base: str) -> str:
        if base not in self._d:
            return base
        i = 2
        while f"{base}_{i}" in self._d:
            i += 1
        return f"{base}_{i}"

    def listing(self) -> str:
        if not self._d:
            return "No datasets loaded. Use load_dataset(path) or make_synthetic(kind)."
        rows = []
        for e in self._d.values():
            extra = ""
            if e.kind == "tensor":
                extra = "dims=" + "x".join(f"{d}[{n}]" for d, n in e.obj.sizes().items())
            else:
                extra = f"{len(e.obj.columns)} cols"
                if e.meta.get("method"):
                    extra += f", embedding via {e.meta['method']}"
            rows.append({"name": e.name, "kind": e.kind, "shape": "x".join(map(str, e.shape)),
                         "memory": _fmt_bytes(e.nbytes()), "info": extra,
                         "parent": e.meta.get("parent", ""), "created": e.created})
        return frame_to_text(pd.DataFrame(rows), index=False)


REGISTRY = Registry()

# --------------------------------------------------------------------------- #
# Loading and saving
# --------------------------------------------------------------------------- #

_GENERIC_DIMS = [f"d{i}" for i in range(16)]


def _array_to_entry(arr: np.ndarray, name: str, source: str, dims: list[str] | None = None,
                    coords: dict[str, np.ndarray] | None = None) -> Entry:
    """Arrays up to 2-D become tables (rows x features); 3-D and up become tensors."""
    arr = np.asarray(arr)
    if arr.ndim <= 2 and dims is None:
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"f{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
        return Entry(name, "table", df, {"source": source})
    dims = dims or _GENERIC_DIMS[: arr.ndim]
    t = Tensor(arr, list(dims), dict(coords or {}))
    return Entry(name, "tensor", t, {"source": source})


def load_dataset(path: str, name: str | None = None, fmt: str | None = None,
                 options: dict[str, Any] | None = None) -> list[Entry]:
    """Load a file into the registry. Returns the entries created (npz/mat/hdf5 may create several)."""
    options = dict(options or {})
    rp = check_path(path)
    ext = (fmt or rp.suffix.lstrip(".")).lower()
    base = name or re.sub(r"[^A-Za-z0-9_]+", "_", rp.stem)
    src = str(rp)
    entries: list[Entry] = []

    if ext in ("csv", "tsv", "txt"):
        kw: dict[str, Any] = {"sep": "\t" if ext == "tsv" else options.pop("sep", ",")}
        kw.update(options)
        df = pd.read_csv(rp, **kw)
        entries.append(Entry(base, "table", df, {"source": src}))
    elif ext in ("parquet", "pq"):
        df = pd.read_parquet(rp, **options)
        entries.append(Entry(base, "table", df, {"source": src}))
    elif ext in ("feather", "ft"):
        df = pd.read_feather(rp, **options)
        entries.append(Entry(base, "table", df, {"source": src}))
    elif ext in ("json", "jsonl", "ndjson"):
        lines = options.pop("lines", ext != "json")
        df = pd.read_json(rp, lines=lines, **options)
        entries.append(Entry(base, "table", df, {"source": src}))
    elif ext in ("xlsx", "xlsm", "xls"):
        df = pd.read_excel(rp, sheet_name=options.pop("sheet", 0), **options)
        entries.append(Entry(base, "table", df, {"source": src}))
    elif ext == "npy":
        arr = np.load(rp, allow_pickle=ALLOW_PICKLE)
        entries.append(_array_to_entry(arr, base, src))
    elif ext == "npz":
        with np.load(rp, allow_pickle=ALLOW_PICKLE) as z:
            keys = list(z.keys())
            if "__hidim_dims__" in keys:  # tensor saved by save_dataset
                dims = list(json.loads(str(z["__hidim_dims__"])))
                coords = {d: z[f"__coord__{d}"] for d in dims}
                entries.append(Entry(base, "tensor", Tensor(z["data"], dims, coords), {"source": src}))
            else:
                key = options.get("key")
                for k in ([key] if key else keys):
                    if k not in z:
                        raise EngineError(f"Key '{k}' not in npz. Keys: {keys}")
                    nm = base if (key or len(keys) == 1) else f"{base}_{k}"
                    entries.append(_array_to_entry(z[k], nm, f"{src}::{k}"))
    elif ext == "mat":
        from scipy.io import loadmat
        m = loadmat(rp, squeeze_me=False)
        key = options.get("key")
        keys = [k for k in m if not k.startswith("__")]
        for k in ([key] if key else keys):
            if k not in m:
                raise EngineError(f"Variable '{k}' not in .mat file. Variables: {keys}")
            arr = m[k]
            if not isinstance(arr, np.ndarray) or arr.dtype == object:
                continue
            nm = base if (key or len(keys) == 1) else f"{base}_{k}"
            entries.append(_array_to_entry(arr, nm, f"{src}::{k}"))
        if not entries:
            raise EngineError(f"No numeric variables found in {rp}. Variables: {keys}")
    elif ext in ("h5", "hdf5", "hdf"):
        try:
            import h5py
        except ImportError as exc:
            raise EngineError("Reading HDF5 needs the optional package h5py (pip install h5py).") from exc
        key = options.get("key")
        with h5py.File(rp, "r") as f:
            dsets: list[str] = []
            f.visititems(lambda n, o: dsets.append(n) if isinstance(o, h5py.Dataset) else None)
            if not key:
                if len(dsets) != 1:
                    raise EngineError(f"HDF5 file has {len(dsets)} datasets; pass options={{'key': ...}}. Datasets: {dsets[:50]}")
                key = dsets[0]
            if key not in f:
                raise EngineError(f"Key '{key}' not in HDF5 file. Datasets: {dsets[:50]}")
            arr = f[key][()]
        entries.append(_array_to_entry(arr, base, f"{src}::{key}"))
    elif ext in ("nc", "netcdf", "zarr"):
        try:
            import xarray as xr
        except ImportError as exc:
            raise EngineError("Reading NetCDF/Zarr needs the optional package xarray (pip install xarray netCDF4).") from exc
        ds = xr.open_zarr(rp) if ext == "zarr" else xr.open_dataset(rp)
        var = options.get("key")
        names = [var] if var else list(ds.data_vars)
        for v in names:
            da = ds[v]
            coords = {d: np.asarray(da.coords[d].values) if d in da.coords else np.arange(da.sizes[d]) for d in da.dims}
            nm = base if (var or len(names) == 1) else f"{base}_{v}"
            entries.append(Entry(nm, "tensor", Tensor(np.asarray(da.values), list(da.dims), coords), {"source": f"{src}::{v}"}))
    elif ext in ("pkl", "pickle"):
        if not ALLOW_PICKLE:
            raise EngineError("Pickle loading is disabled (it can execute code). Set HIDIM_ALLOW_PICKLE=1 to enable.")
        obj = pd.read_pickle(rp)
        if isinstance(obj, pd.DataFrame):
            entries.append(Entry(base, "table", obj, {"source": src}))
        else:
            entries.append(_array_to_entry(np.asarray(obj), base, src))
    else:
        raise EngineError(f"Unsupported format '{ext}'. Supported: csv tsv txt parquet feather json jsonl xlsx npy npz mat h5 nc zarr (pkl if enabled).")

    out = []
    for e in entries:
        if e.kind == "table":
            e.obj = _tidy_frame(e.obj)
        out.append(REGISTRY.put(e, replace=name is not None))
    return out


def _tidy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Make column labels strings and parse obvious datetime columns."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            s = df[c].dropna()
            if len(s) and s.map(lambda v: isinstance(v, str)).all():
                sample = s.iloc[: min(len(s), 50)]
                if sample.str.match(r"^\d{4}-\d{2}-\d{2}").all():
                    try:
                        df[c] = pd.to_datetime(df[c], errors="raise")
                    except (ValueError, TypeError):
                        pass
    return df


def save_dataset(name: str, path: str, fmt: str | None = None) -> str:
    e = REGISTRY.get(name)
    rp = check_path(path, must_exist=False)
    rp.parent.mkdir(parents=True, exist_ok=True)
    ext = (fmt or rp.suffix.lstrip(".")).lower()
    if e.kind == "table":
        df: pd.DataFrame = e.obj
        if ext == "csv":
            df.to_csv(rp, index=not isinstance(df.index, pd.RangeIndex))
        elif ext in ("parquet", "pq"):
            df.to_parquet(rp)
        elif ext == "feather":
            df.reset_index(drop=isinstance(df.index, pd.RangeIndex)).to_feather(rp)
        elif ext == "json":
            df.to_json(rp, orient="records", date_format="iso")
        elif ext == "npy":
            np.save(rp, _numeric_frame(e).to_numpy(dtype=float))
        elif ext == "xlsx":
            df.to_excel(rp, index=not isinstance(df.index, pd.RangeIndex))
        elif ext == "mat":
            from scipy.io import savemat
            savemat(rp, {re.sub(r"\W", "_", name): _numeric_frame(e).to_numpy(dtype=float)})
        else:
            raise EngineError(f"Unsupported table format '{ext}' (csv parquet feather json npy xlsx mat).")
    else:
        t: Tensor = e.obj
        if ext == "npz":
            payload = {"data": t.data, "__hidim_dims__": np.array(json.dumps(t.dims))}
            for d in t.dims:
                payload[f"__coord__{d}"] = t.coords[d]
            np.savez_compressed(rp, **payload)
        elif ext == "npy":
            np.save(rp, t.data)
        elif ext in ("nc", "netcdf"):
            try:
                t.to_xarray().to_netcdf(rp)
            except ImportError as exc:
                raise EngineError("Saving NetCDF needs xarray (+ netCDF4 or scipy).") from exc
        elif ext == "mat":
            from scipy.io import savemat
            payload = {"data": t.data, "dims": np.array(t.dims, dtype=object)}
            for d in t.dims:
                c = t.coords[d]
                payload[f"coord_{d}"] = c.astype(str) if c.dtype.kind in "MOU" else c
            savemat(rp, payload)
        else:
            raise EngineError(f"Unsupported tensor format '{ext}' (npz preserves dims/coords; npy raw; nc needs xarray; mat).")
    return f"Saved '{name}' ({e.kind}, shape {'x'.join(map(str, e.shape))}) to {rp} ({_fmt_bytes(rp.stat().st_size)})."


# --------------------------------------------------------------------------- #
# Column resolution and numeric matrices
# --------------------------------------------------------------------------- #


def resolve_columns(df: pd.DataFrame, spec: Sequence[str] | str | None, entry: Entry | None = None,
                    numeric_only: bool = False) -> list[str]:
    """spec: None (all), list of names, 're:<regex>', 'top_variance:<k>', or comma-separated names."""
    if spec is None:
        cols = list(df.columns)
    elif isinstance(spec, str):
        if spec.startswith("re:"):
            pat = re.compile(spec[3:])
            cols = [c for c in df.columns if pat.search(str(c))]
        elif spec.startswith("top_variance:"):
            k = int(spec.split(":", 1)[1])
            num = df.select_dtypes(include=[np.number]).drop(columns=entry.derived_cols() if entry else [], errors="ignore")
            cols = list(num.var().sort_values(ascending=False).index[:k])
        else:
            cols = [c.strip() for c in spec.split(",") if c.strip()]
    else:
        cols = list(spec)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise EngineError(f"Unknown columns {missing[:10]}. Available: {list(df.columns)[:40]}{' ...' if df.shape[1] > 40 else ''}")
    if numeric_only:
        bad = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise EngineError(f"Non-numeric columns requested for a numeric operation: {bad[:10]}")
    if not cols:
        raise EngineError("Column selection resolved to zero columns.")
    return cols


_ID_NAMES = {"id", "index", "idx", "row", "col", "row_id", "rowid", "record", "record_id", "sample_id", "uid", "key"}
_COORD_NAMES = {"year", "month", "day", "hour", "minute", "second", "weekday", "dow", "doy", "week", "quarter",
                "lat", "lon", "latitude", "longitude", "x", "y", "zip", "zipcode", "postcode"}


def identifier_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> tuple[list[str], list[str]]:
    """Heuristic split of numeric columns into (identifiers to leave out of feature matrices, candidates to warn about).

    Left out automatically: integer-valued columns that are unique per row (an id or row number) or whose name is a
    conventional identifier (id, index, row, col, ...). Only warned about, never removed: integer-valued columns whose
    name looks like a coordinate / calendar field (year, hour, lat, lon, ...) or that take very few distinct values
    (a class code stored as a number). Pass features= or call set_labels() to decide explicitly.
    """
    exclude = set(exclude)
    ids: list[str] = []
    suspects: list[str] = []
    n = len(df)
    for c in df.columns:
        if c in exclude or not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        s = df[c]
        name = str(c).lower()
        is_int = pd.api.types.is_integer_dtype(s)
        if not is_int:
            v = s.dropna().to_numpy(dtype=float)
            if len(v) == 0:
                continue
            v = v[: min(len(v), 20_000)]
            is_int = bool(np.all(np.isfinite(v)) and np.allclose(v, np.round(v)))
        if not is_int:
            continue
        nun = s.nunique(dropna=True)
        if name in _ID_NAMES or re.search(r"(_id|_idx|_index|_key|ID|Id)$", str(c)):
            ids.append(c)
        elif n >= 20 and nun == n and s.is_monotonic_increasing:
            ids.append(c)
        elif name in _COORD_NAMES or (n >= 50 and nun <= min(50, max(2, n // 20))):
            suspects.append(c)
    return ids, suspects


def _numeric_frame(entry: Entry, features: Sequence[str] | str | None = None, notes: dict | None = None) -> pd.DataFrame:
    df: pd.DataFrame = entry.obj
    if features is None:
        num = df.select_dtypes(include=[np.number, "bool"])
        num = num.drop(columns=[c for c in entry.derived_cols() if c in num.columns], errors="ignore")
        ids, suspects = identifier_columns(num)
        if ids:
            num = num.drop(columns=ids)
            if notes is not None:
                notes["excluded_identifier_cols"] = ids
        if suspects and notes is not None:
            notes["integer_cols_that_may_be_labels_or_coordinates"] = suspects[:12]
        if num.shape[1] == 0:
            raise EngineError("No numeric feature columns in this table (identifier-like columns are skipped; pass features=[...] to use them).")
        return num.astype(float)
    cols = resolve_columns(df, features, entry, numeric_only=True)
    return df[cols].astype(float)


def scaling_profile(X: np.ndarray) -> dict[str, Any]:
    """Decide whether z-scoring every column is a sensible default for this matrix.

    Data measured in one shared unit (pixel intensities, read counts, [0,1] images, spectra) should normally NOT be
    standardized: z-scoring amplifies near-constant columns (border pixels, rare genes) into dominant features,
    inflates the condition number and doubles the PCA rank. Heterogeneous columns (prices next to volumes next to
    ratios) should be standardized.
    """
    n, d = X.shape
    sd = X.std(axis=0)
    mx = X.max(axis=0) if n else np.array([])
    mn = X.min(axis=0) if n else np.array([])
    nonneg = bool(n and np.all(mn >= 0))
    sample = X[: min(n, 5000)]
    integer_like = bool(n and np.all(np.isfinite(sample)) and np.allclose(sample, np.round(sample)))
    unit_interval = bool(n and np.all(mn >= 0) and np.all(mx <= 1.0 + 1e-9))
    byte_range = bool(n and nonneg and np.all(mx <= 255 + 1e-9) and integer_like)
    pos_sd = sd[sd > 0]
    near_const = int(np.sum((sd > 0) & (sd < 1e-3 * (pos_sd.max() if len(pos_sd) else 1.0))))
    reason = ""
    recommend_raw = False
    if d >= 8 and nonneg and integer_like:
        recommend_raw = True
        reason = "non-negative integer-valued features in one shared unit (counts / pixel intensities)"
    elif d >= 8 and unit_interval and not integer_like:
        recommend_raw = True
        reason = "all features share the [0, 1] range (normalized intensities / proportions)"
    kind = "counts_or_pixels" if (byte_range or (nonneg and integer_like)) else "unit_interval" if unit_interval else "mixed"
    return {"recommend_raw": recommend_raw, "reason": reason, "kind": kind, "near_constant_cols": near_const,
            "nonneg": nonneg, "integer_like": integer_like}


def _resolve_standardize(standardize: Any, X: np.ndarray, notes: dict[str, Any]) -> bool:
    """standardize may be True, False or 'auto'. 'auto' uses scaling_profile and records the decision in notes."""
    if isinstance(standardize, str):
        s = standardize.lower()
        if s in ("auto", "a"):
            prof = scaling_profile(X)
            if prof["recommend_raw"]:
                notes["scaling"] = f"raw (auto: {prof['reason']}; pass standardize=True to z-score anyway)"
                return False
            notes["scaling"] = "standardized (auto: heterogeneous units; pass standardize=False to keep raw scales)"
            return True
        if s in ("true", "yes", "z", "zscore", "standard"):
            return True
        if s in ("false", "no", "raw", "none"):
            return False
        raise EngineError(f"standardize must be true, false or 'auto' (got {standardize!r})")
    return bool(standardize)


def _matrix(entry: Entry, features=None, standardize: Any = False, impute: str = "mean",
            clip_z: float | None = None) -> tuple[np.ndarray, list[str], dict]:
    """Return (X, feature_names, notes). NaNs are imputed; all-NaN or constant columns are dropped if standardizing.

    standardize: True | False | 'auto' (see scaling_profile). clip_z: after z-scoring, clip every value to +-clip_z
    (the usual value is 10); this stops near-constant features from producing cells at +-50 sd that then hijack
    k-means, PCA and distance computations.
    """
    notes: dict[str, Any] = {}
    num = _numeric_frame(entry, features, notes)
    X = num.to_numpy(dtype=float)
    cols = list(num.columns)
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        notes["imputed_cells"] = int(nan_mask.sum())
        all_nan = nan_mask.all(axis=0)
        if all_nan.any():
            notes["dropped_all_nan_cols"] = [c for c, b in zip(cols, all_nan) if b]
            X = X[:, ~all_nan]
            cols = [c for c, b in zip(cols, all_nan) if not b]
            nan_mask = nan_mask[:, ~all_nan]
        fill = np.nanmean(X, axis=0) if impute == "mean" else np.nanmedian(X, axis=0)
        X = np.where(nan_mask, fill[None, :], X)
    do_std = _resolve_standardize(standardize, X, notes)
    # constant columns carry no information in either mode (zero variance, zero distance contribution): drop them
    sd0 = X.std(axis=0) if X.size else np.array([])
    const = sd0 == 0
    if const.any():
        notes["dropped_constant_cols"] = [c for c, b in zip(cols, const) if b]
        X = X[:, ~const]
        cols = [c for c, b in zip(cols, const) if not b]
    if do_std and X.size:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        Z = (X - mu) / sd
        big = np.abs(Z) > 10
        if big.any() and clip_z is None:
            bad_cols = np.where(big.any(axis=0))[0]
            # near-constant: almost every row sits at one value; heavy-tailed: a continuous column with rare extreme days
            modes = np.array([np.mean(X[:, j] == pd.Series(X[:, j]).mode().iloc[0]) for j in bad_cols])
            nc = bad_cols[modes > 0.95]
            ht = bad_cols[modes <= 0.95]
            zmax = float(np.abs(Z).max())
            if len(nc):
                notes["standardization_inflation"] = (f"|z| up to {zmax:.0f} in {len(nc)} near-constant feature(s) {[cols[j] for j in nc[:6]]}: "
                                                      f"a handful of rows become extreme outliers that dominate distances and k-means. Consider clip_z=10 or standardize=False")
            if len(ht):
                notes["heavy_tails"] = (f"|z| up to {float(np.abs(Z[:, ht]).max()):.0f} in {len(ht)} heavy-tailed feature(s) {[cols[j] for j in ht[:6]]}: "
                                        f"rare extreme rows will drive k-means and Euclidean kNN; consider clip_z=10 or transform(op='winsorize')")
        X = Z
        if clip_z is not None:
            cz = float(clip_z)
            if cz <= 0:
                raise EngineError("clip_z must be positive (10 is the usual value)")
            clipped = int(np.sum(np.abs(X) > cz))
            X = np.clip(X, -cz, cz)
            notes["clipped_cells"] = f"{clipped} cells clipped to +-{cz:g} sd"
    elif clip_z is not None:
        notes["clip_z_ignored"] = "clip_z only applies when standardizing"
    if X.shape[1] == 0:
        raise EngineError("No usable numeric columns after dropping constant/all-NaN columns.")
    notes["standardized"] = bool(do_std)
    return X, cols, notes


def _sample_rows(n: int, cap: int | None, seed: int) -> np.ndarray | None:
    if cap is None or n <= cap:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


# --------------------------------------------------------------------------- #
# High-dimensional diagnostics
# --------------------------------------------------------------------------- #


def _knn_dists(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, 1:]  # drop self


def intrinsic_dimension(X: np.ndarray, k: int = 10) -> dict[str, float]:
    """TwoNN (Facco et al. 2017) and Levina-Bickel MLE (with MacKay-Ghahramani averaging)."""
    n = len(X)
    k = int(min(k, n - 2))
    if n < 20 or k < 2:
        return {}
    d = _knn_dists(X, k)
    out: dict[str, float] = {}
    # TwoNN
    r1, r2 = d[:, 0], d[:, 1]
    ok = r1 > 0
    mu = r2[ok] / r1[ok]
    mu = np.sort(mu)
    m = len(mu)
    if m > 10:
        keep = int(math.floor(0.9 * m))  # discard the top 10% (heavy tail)
        mu_k = mu[:keep]
        F = np.arange(1, keep + 1) / m
        x = np.log(mu_k)
        y = -np.log(1 - F)
        good = (x > 0) & np.isfinite(y)
        if good.sum() > 5:
            out["twonn"] = float(np.sum(x[good] * y[good]) / np.sum(x[good] ** 2))
    # MLE
    Tk = d[:, k - 1:k]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(Tk / d[:, : k - 1])
        m_hat = (k - 1) / np.sum(logs, axis=1)
    m_hat = m_hat[np.isfinite(m_hat) & (m_hat > 0)]
    if len(m_hat):
        out[f"mle_k{k}"] = float(1.0 / np.mean(1.0 / m_hat))
    return out


def concentration_diagnostics(X: np.ndarray, seed: int = 0, n_query: int = 500, n_ref: int = 2000) -> dict[str, float]:
    """Distance concentration (Beyer et al. 1999) and hubness (Radovanovic et al. 2010)."""
    from scipy.spatial.distance import cdist
    from scipy.stats import skew
    n = len(X)
    rng = np.random.default_rng(seed)
    q = X[rng.choice(n, size=min(n_query, n), replace=False)]
    r = X[rng.choice(n, size=min(n_ref, n), replace=False)]
    D = cdist(q, r)
    D[D == 0] = np.nan  # ignore self-matches
    dmin = np.nanmin(D, axis=1)
    dmax = np.nanmax(D, axis=1)
    ok = dmin > 0
    rc = (dmax[ok] - dmin[ok]) / dmin[ok]
    flat = D[np.isfinite(D)]
    out = {
        "relative_contrast_median": float(np.median(rc)) if len(rc) else float("nan"),
        "distance_cv": float(np.std(flat) / np.mean(flat)) if len(flat) else float("nan"),
    }
    # hubness: k-occurrence skewness on the reference sample
    k = min(10, len(r) - 1)
    if k >= 2:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1).fit(r)
        _, idx = nn.kneighbors(r)
        occ = np.bincount(idx[:, 1:].ravel(), minlength=len(r))
        out["hubness_skew_k10"] = float(skew(occ))
    return out


def _singular_values(X: np.ndarray) -> tuple[np.ndarray, str]:
    n, d = X.shape
    if d > 3000 or n * d > 4e7:
        from sklearn.utils.extmath import randomized_svd
        k = min(300, d - 1, n - 1)
        _, s, _ = randomized_svd(X, n_components=k, random_state=0)
        return s, f"spectrum from randomized SVD of the top {k} components"
    return np.linalg.svd(X, compute_uv=False), ""


def pca_counts(X: np.ndarray, thresholds: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.99)) -> dict[str, Any]:
    """Number of principal components needed to reach each cumulative variance threshold (X already centred/scaled)."""
    s, note = _singular_values(X)
    s = s[s > 0]
    out: dict[str, Any] = {}
    if len(s) == 0:
        return out
    ev = s ** 2
    total = ev.sum() if not note else float((X ** 2).sum())
    cum = np.cumsum(ev) / total
    for thr in thresholds:
        idx = int(np.searchsorted(cum, thr) + 1)
        out[f"components_for_{int(thr * 100)}pct"] = idx if cum[-1] >= thr else f">{len(cum)}"
    out["top_variance_ratios"] = [round(float(v), 4) for v in (ev / total)[:10]]
    out["_s"] = s
    out["_total"] = total
    if note:
        out["note"] = note
    return out


def marchenko_pastur(Xs: np.ndarray, s: np.ndarray | None = None) -> dict[str, Any]:
    """Count correlation eigenvalues above the Marchenko-Pastur noise edge of a standardized n x p matrix.

    For iid noise the eigenvalues of the correlation matrix fall inside [(1-sqrt(p/n))^2, (1+sqrt(p/n))^2];
    eigenvalues above the upper edge are the factors that carry more than noise. This is the standard random-matrix
    test used on return panels (Laloux et al. 1999, Plerou et al. 2002) and it is the right yardstick whenever the
    PCA count for 90% variance looks large: most of that rank may be noise.
    """
    n, p = Xs.shape
    if n < 10 or p < 2:
        return {}
    if s is None:
        s, _ = _singular_values(Xs)
    lam = (s ** 2) / max(n - 1, 1)                       # eigenvalues of the correlation matrix (unit-variance columns)
    q = p / n
    edge = (1 + math.sqrt(q)) ** 2
    above = int(np.sum(lam > edge))
    return {"mp_edge": float(edge), "factors_above_edge": above, "p_over_n": float(q),
            "top_eigenvalues": [round(float(v), 2) for v in lam[:5]]}


def spectrum_diagnostics(Xs: np.ndarray, standardized: bool = True) -> dict[str, Any]:
    """Singular spectrum of a (centred, possibly standardized) matrix: effective rank, condition number, PCA variance curve,
    and when standardized the number of factors above the Marchenko-Pastur noise edge."""
    out: dict[str, Any] = {}
    pc = pca_counts(Xs)
    if not pc:
        return out
    s = pc.pop("_s")
    pc.pop("_total", None)
    out.update(pc)
    p = s / s.sum()
    out["effective_rank"] = float(np.exp(-(p * np.log(p)).sum()))
    s_pos = s[s > s[0] * 1e-12]                         # ignore numerically-zero singular values
    out["log10_condition_number"] = float(np.log10(s_pos[0] / s_pos[-1]))
    if len(s_pos) < len(s):
        out["numerical_rank"] = int(len(s_pos))
    if standardized:
        out.update(marchenko_pastur(Xs, s))
    return out


def _fmt_notes(notes: dict[str, Any]) -> str:
    """Render the notes dict from _matrix without the bookkeeping keys."""
    shown = {k: v for k, v in notes.items() if k not in ("standardized",)}
    return f"notes: {shown}" if shown else ""


def _hints(n: int, p: int, spec: dict[str, Any], idim: dict[str, float], conc: dict[str, float],
           notes: dict[str, Any], standardized: bool, other_counts: dict[str, Any] | None) -> list[str]:
    """Decision tree for the 'hint:' line of describe(). One main recommendation, never two that contradict."""
    hints: list[str] = []
    c90 = spec.get("components_for_90pct")
    c90 = c90 if isinstance(c90, int) else None
    mp = spec.get("factors_above_edge")
    id_est = min(idim.values()) if idim else None
    cv = conc.get("distance_cv")
    # 1. scaling advice first: it changes every number below
    if notes.get("scaling", "").startswith("raw"):
        hints.append("geometry computed on RAW scales (" + notes["scaling"].split("auto: ", 1)[-1].split(";")[0] + ")")
    if "standardization_inflation" in notes:
        hints.append("standardization inflated near-constant features (see notes); use clip_z=10 or standardize=False before trusting PCA counts, kNN or k-means")
    elif "heavy_tails" in notes:
        hints.append("heavy-tailed features (see notes): distance-based methods will follow the extreme rows; clip_z=10 or winsorize first")
    if other_counts and c90 is not None and isinstance(other_counts.get("components_for_90pct"), int):
        o = other_counts["components_for_90pct"]
        if max(o, c90) >= 1.5 * max(1, min(o, c90)):
            a, b = ("standardized", "raw") if standardized else ("raw", "standardized")
            hints.append(f"PCA rank depends on scaling here: {c90} components for 90% {a} vs {o} {b}; report which one you used")
    # 2. the main structural recommendation
    if n < 30:
        hints.append("too few rows for intrinsic-dimension / concentration diagnostics; PCA and pairs plots only")
    elif mp is not None and mp == 0 and p >= 10:
        hints.append(f"no correlation eigenvalue clears the Marchenko-Pastur noise edge ({spec['mp_edge']:.2f}): the feature block is "
                     "indistinguishable from independent noise at this sample size; no projection or clustering will show real structure "
                     "(compare with make_synthetic(kind='gaussian') if in doubt)")
    elif mp is not None and c90 is not None and mp <= 5 and c90 >= 3 * mp and p >= 10:
        hints.append(f"factor + noise structure: {mp} eigenvalue(s) above the Marchenko-Pastur noise edge ({spec['mp_edge']:.2f}) "
                     f"but {c90} components for 90% variance: the extra rank is noise-like. Inspect PCA loadings of the first {mp} "
                     f"(project pca then loadings()); t-SNE/UMAP on these rows would mostly draw noise")
    elif c90 is not None and c90 <= 3:
        msg = f"low linear rank: PCA to {c90} components keeps 90% of variance, so plot('pca') is nearly lossless"
        if id_est is not None and id_est > 2 * c90 + 2:
            msg += f" (intrinsic dimension ~{id_est:.0f} is the within-group spread, not extra structure)"
        hints.append(msg)
    elif id_est is not None and id_est <= 3 and c90 is not None and c90 > id_est + 2:
        hints.append(f"intrinsic dimension ~{id_est:.1f} but {c90} linear components: a curved low-dimensional manifold; isomap/umap/tsne will unroll what PCA cannot")
    elif n < 200:
        hints.append(f"only {n} rows: use PCA, pairs and correlation plots; t-SNE/UMAP are unreliable below ~200 rows")
    elif c90 is not None and c90 <= 10:
        hints.append(f"moderate rank: PCA to {c90} keeps 90%; try project(method='pca', n_components={c90}) then tsne/umap for finer structure")
    elif id_est is not None and id_est > 0.5 * p:
        hints.append(f"intrinsic dimension ~{id_est:.0f} of {p}: little manifold structure (noise-like); prefer statistics, kNN and clustering diagnostics over 2-D pictures")
    elif c90 is not None:
        hints.append(f"high rank ({c90} components for 90%) with intrinsic dimension ~{id_est:.0f}: linear 2-3D projections lose most variance; tsne/umap with the automatic pre-PCA are the right pictures" if id_est is not None
                     else f"high rank ({c90} components for 90%): use tsne/umap (pre_pca is automatic)")
    # 3. distance geometry caveats
    if cv is not None and cv < 0.1:
        hints.append("severe distance concentration: kNN, DBSCAN and k-means in the raw space are unreliable; reduce first or use cosine")
    if (conc.get("hubness_skew_k10") or 0) > 3:
        hints.append("strong hubness: a few rows appear in everyone's kNN list, so neighbors() results near them are not meaningful "
                     "(cosine distance or a lower-dimensional embedding usually helps)")
    return hints


def describe(name: str, max_features: int = 40, sample: int = 5000, seed: int = 0, standardize: Any = "auto") -> str:
    e = REGISTRY.get(name)
    if e.kind == "tensor":
        return clip(_describe_tensor(e))
    df: pd.DataFrame = e.obj
    n, d = df.shape
    L: list[str] = []
    L.append(f"# {name}  (table)  rows={n:,}  cols={d}  memory={_fmt_bytes(e.nbytes())}")
    if e.meta.get("source"):
        L.append(f"source: {e.meta['source']}")
    if e.meta.get("method"):
        L.append(f"embedding: method={e.meta['method']} parent={e.meta.get('parent')} params={e.meta.get('params')}")
    if e.derived_cols():
        L.append(f"derived label columns (excluded from numeric analysis): {e.derived_cols()}")
    # index
    idx = df.index
    if not isinstance(idx, pd.RangeIndex):
        L.append(f"index: {type(idx).__name__} [{_fmt_num(idx.min())} .. {_fmt_num(idx.max())}] unique={idx.is_unique} monotonic={idx.is_monotonic_increasing}")
    # dtype summary
    kinds = df.dtypes.astype(str).value_counts()
    L.append("dtypes: " + ", ".join(f"{k}:{v}" for k, v in kinds.items()))
    # non-numeric
    non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) or c in e.derived_cols()]
    if non_num:
        parts = []
        for c in non_num[:25]:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                parts.append(f"{c}: datetime [{_fmt_num(s.min())} .. {_fmt_num(s.max())}] unique={s.nunique()}")
            else:
                nun = s.nunique(dropna=True)
                top = s.value_counts(dropna=True).head(4)
                parts.append(f"{c}: {s.dtype} unique={nun} top={dict(zip(map(str, top.index), map(int, top.values)))}")
        L.append("non-numeric / label columns:\n  " + "\n  ".join(parts) + (f"\n  ... {len(non_num) - 25} more" if len(non_num) > 25 else ""))
    # numeric block
    fnotes: dict[str, Any] = {}
    try:
        num = _numeric_frame(e, None, fnotes)
    except EngineError:
        L.append("No numeric columns.")
        return clip("\n".join(L))
    dn = num.shape[1]
    L.append(f"\n## numeric features: {dn}")
    if fnotes.get("excluded_identifier_cols"):
        L.append(f"skipped as identifiers (unique per row or named like an id): {fnotes['excluded_identifier_cols']} -> pass features=[...] to include them")
    if fnotes.get("integer_cols_that_may_be_labels_or_coordinates"):
        L.append(f"integer columns that look like labels or coordinates (year/hour/lat/class codes) and are currently counted as features: "
                 f"{fnotes['integer_cols_that_may_be_labels_or_coordinates']} -> set_labels('{name}', [...]) to exclude them")
    nan_frac = num.isna().mean()
    const = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    if const:
        L.append(f"constant columns ({len(const)}): {const[:15]}{' ...' if len(const) > 15 else ''}")
    if 2 <= dn <= 2000 and n <= 200_000:
        dup_pairs = _duplicate_columns(num)
        if dup_pairs:
            L.append(f"identical feature pairs ({len(dup_pairs)}; keep one of each): {dup_pairs[:6]}{' ...' if len(dup_pairs) > 6 else ''}")
    if nan_frac.sum() > 0:
        worst = nan_frac.sort_values(ascending=False).head(8)
        L.append("missing: overall {:.2%}; worst: ".format(float(num.isna().to_numpy().mean())) +
                 ", ".join(f"{c}={v:.1%}" for c, v in worst.items() if v > 0))
    if n <= 500_000:
        dups = int(df.duplicated().sum())
        if dups:
            L.append(f"duplicate rows: {dups}")
    # per-feature stats (top by variance if too many)
    show_cols = list(num.columns)
    note = ""
    if dn > max_features:
        show_cols = list(num.var().sort_values(ascending=False).index[:max_features])
        note = f" (top {max_features} of {dn} by variance)"
    stats = num[show_cols].describe(percentiles=[0.05, 0.5, 0.95]).T
    stats["nan%"] = (num[show_cols].isna().mean() * 100).round(2)
    stats = stats.drop(columns=["count"]).rename(columns={"5%": "q05", "50%": "med", "95%": "q95"})
    L.append(f"per-feature stats{note}:\n" + frame_to_text(stats, max_rows=max_features, max_cols=10))

    if dn >= 2 and n >= 5:
        # geometry diagnostics on a sample; scaling chosen by `standardize` ('auto' inspects the data)
        sub_idx = _sample_rows(n, sample, seed)
        sub = Entry(name, "table", num.iloc[sub_idx] if sub_idx is not None else num, {"derived_cols": []})
        try:
            Xg, cols, notes = _matrix(sub, standardize=standardize)
        except EngineError as exc:
            L.append(f"geometry diagnostics skipped: {exc}")
            return clip("\n".join(L))
        is_std = bool(notes.get("standardized"))
        # both scalings are kept: correlations and the Marchenko-Pastur count need the standardized copy,
        # the "same counts on the other scale" line needs the raw one
        if is_std:
            Xs, scols = Xg, cols
            X_raw, _, _ = _matrix(sub, standardize=False)
        else:
            X_raw = Xg
            Xs, scols, _ = _matrix(sub, standardize=True)
        Xc = X_raw - X_raw.mean(axis=0)
        scale_word = "standardized" if is_std else "raw scales, centred"
        L.append(f"\n## geometry ({scale_word}{'' if sub_idx is None else f', random sample of {len(Xg):,} rows'}, {len(cols)} features)")
        nt = _fmt_notes({k: v for k, v in notes.items() if k not in ("excluded_identifier_cols", "integer_cols_that_may_be_labels_or_coordinates")})
        if nt:
            L.append(nt)
        # correlations (always on the standardized copy: correlation is scale-free)
        if len(scols) <= 600:
            C = np.corrcoef(Xs, rowvar=False)
            iu = np.triu_indices_from(C, k=1)
            vals = C[iu]
            order = np.argsort(-np.abs(vals))[:10]
            pairs = [f"{scols[iu[0][o]]}~{scols[iu[1][o]]}={vals[o]:+.3f}" for o in order if abs(vals[o]) > 0.3]
            L.append(f"mean |corr| off-diagonal: {np.nanmean(np.abs(vals)):.3f}; strongest pairs: {', '.join(pairs) if pairs else 'none above |0.3|'}")
        spec = spectrum_diagnostics(Xg if is_std else Xc, standardized=is_std)
        if not is_std:
            spec.update(marchenko_pastur(Xs))
        other = pca_counts(Xc if is_std else Xs) if len(cols) <= 3000 else {}
        if spec:
            L.append("spectrum: effective_rank={:.2f}, log10(cond)={:.2f}, components for 50/80/90/95/99% variance = {}/{}/{}/{}/{}".format(
                spec.get("effective_rank", float("nan")), spec.get("log10_condition_number", float("nan")),
                spec.get("components_for_50pct"), spec.get("components_for_80pct"), spec.get("components_for_90pct"),
                spec.get("components_for_95pct"), spec.get("components_for_99pct")))
            if other:
                a, b = ("standardized", "raw") if is_std else ("raw", "standardized")
                L.append(f"  same counts on {b} scales: 50/80/90/95/99% = {other.get('components_for_50pct')}/{other.get('components_for_80pct')}/"
                         f"{other.get('components_for_90pct')}/{other.get('components_for_95pct')}/{other.get('components_for_99pct')}  (shown above: {a})")
            L.append(f"top variance ratios: {spec.get('top_variance_ratios')}" + (f" ({spec['note']})" if "note" in spec else ""))
            if "factors_above_edge" in spec:
                qn = f"p/n={spec['p_over_n']:.2f}" + (" (p>=n: edge is a rough guide)" if spec["p_over_n"] >= 1 else "")
                k_mp = spec['factors_above_edge']
                verdict = ("indistinguishable from iid noise" if k_mp == 0 else
                           f"{k_mp} factor(s) carry more than noise" + ("; the rest of the spectrum is a noise bulk" if isinstance(spec.get("components_for_90pct"), int) and spec["components_for_90pct"] >= 3 * k_mp else ""))
                L.append(f"random-matrix check: {k_mp} correlation eigenvalue(s) above the Marchenko-Pastur noise edge "
                         f"{spec['mp_edge']:.2f} ({qn}); top eigenvalues {spec['top_eigenvalues']} -> {verdict}")
        idim: dict[str, float] = {}
        conc: dict[str, float] = {}
        if len(Xg) >= 30:
            idim = intrinsic_dimension(Xg)
            if idim:
                L.append("intrinsic dimension: " + ", ".join(f"{k}={v:.2f}" for k, v in idim.items()) +
                         f"  (ambient {len(cols)})")
            conc = concentration_diagnostics(Xg, seed=seed)
            rc, cv, hub = conc.get("relative_contrast_median"), conc.get("distance_cv"), conc.get("hubness_skew_k10")
            verdict = ("severe concentration: nearest-neighbour distances barely differ from farthest; kNN/clustering in raw space unreliable"
                       if cv is not None and cv < 0.1 else
                       "moderate concentration" if cv is not None and cv < 0.25 else "distances well spread")
            L.append(f"distance concentration: median relative contrast={rc:.3f}, CV of pairwise distances={cv:.3f}, hubness skew(k=10)={hub:.2f} -> {verdict}"
                     + ("; strong hubs present (skew>1)" if hub is not None and hub > 1 else ""))
        hints = _hints(len(Xg), len(cols), spec, idim, conc, notes, is_std, other)
        if hints:
            L.append("hint: " + "; ".join(hints))
    return clip("\n".join(L))


def _duplicate_columns(num: pd.DataFrame, max_pairs: int = 50) -> list[str]:
    """Pairs of numeric columns with identical values (NaN-aware), e.g. the duplicated magnitude features in UCI HAR."""
    try:
        h = pd.util.hash_pandas_object(num.T.fillna(np.inf), index=False).to_numpy()
    except Exception:  # noqa: BLE001
        return []
    groups: dict[int, list[str]] = {}
    for c, hv in zip(num.columns, h):
        groups.setdefault(int(hv), []).append(c)
    pairs = []
    for cols in groups.values():
        if len(cols) > 1:
            base = cols[0]
            for other in cols[1:]:
                if num[base].equals(num[other]):
                    pairs.append(f"{base}~{other}")
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _describe_tensor(e: Entry) -> str:
    t: Tensor = e.obj
    L = [f"# {e.name}  (tensor)  shape={'x'.join(map(str, t.shape))}  dtype={t.data.dtype}  memory={_fmt_bytes(t.nbytes())}"]
    if e.meta.get("source"):
        L.append(f"source: {e.meta['source']}")
    for k in ("op", "parent", "params"):
        if e.meta.get(k):
            L.append(f"{k}: {e.meta[k]}")
    L.append("dims:")
    for d in t.dims:
        c = t.coords[d]
        if len(c) <= 8:
            desc = "[" + ", ".join(_fmt_num(v) for v in c) + "]"
        else:
            desc = f"[{_fmt_num(c[0])}, {_fmt_num(c[1])}, ... , {_fmt_num(c[-1])}]"
        kind = "datetime" if c.dtype.kind == "M" else "numeric" if c.dtype.kind in "iuf" else "label"
        L.append(f"  {d}: size={len(c)} ({kind}) {desc}")
    data = t.data
    if data.dtype.kind in "iufb":
        flat = data.astype(float).ravel()
        fin = np.isfinite(flat)
        L.append(f"values: fill={fin.mean():.1%} (NaN/inf {(~fin).sum():,} of {flat.size:,})")
        if fin.any():
            v = flat[fin]
            L.append(f"  min={_fmt_num(v.min())} q05={_fmt_num(np.quantile(v, .05))} median={_fmt_num(np.median(v))} "
                     f"mean={_fmt_num(v.mean())} q95={_fmt_num(np.quantile(v, .95))} max={_fmt_num(v.max())} std={_fmt_num(v.std())}")
            # per-axis fill rate
            for i, d in enumerate(t.dims):
                axes = tuple(j for j in range(data.ndim) if j != i)
                fill = np.isfinite(data.astype(float)).mean(axis=axes) if axes else np.isfinite(data.astype(float))
                if fill.min() < 0.999:
                    worst = np.argsort(fill)[:3]
                    L.append(f"  fill along {d}: min={fill.min():.1%} (sparsest: {', '.join(_fmt_num(t.coords[d][w]) + f'={fill[w]:.0%}' for w in worst)})")
    L.append("read it with slice_tensor(select={dim: value | {'range':[lo,hi]} | {'isin':[...]} | {'nearest': v} | {'iloc':[start,stop]} | {'every': n}}, "
             "reduce={dim: 'mean'|'std'|'min'|'max'|'sum'|'median'|'count'}) until 1 or 2 dims remain; plot(kind='tensor'|'surface'|'lines').")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Projections
# --------------------------------------------------------------------------- #


def _ascii_density(x: np.ndarray, y: np.ndarray, w: int = 48, h: int = 16) -> str:
    H, _, _ = np.histogram2d(x, y, bins=[w, h])
    H = H.T[::-1]  # rows top->bottom = high y -> low y
    chars = " .:-=+*#%@"
    nz = H[H > 0]
    if len(nz) == 0:
        return ""
    qs = np.quantile(nz, np.linspace(0, 1, len(chars) - 1))
    out = []
    for row in H:
        line = "".join(chars[0] if v == 0 else chars[min(len(chars) - 1, 1 + int(np.searchsorted(qs, v, side="right")) - 1)] for v in row)
        out.append("|" + line + "|")
    return "\n".join(out)


def project(name: str, method: str = "pca", n_components: int = 2, features=None, standardize: Any = "auto",
            sample: int | None = None, seed: int = 0, params: dict[str, Any] | None = None,
            new_name: str | None = None, pre_pca: int | None = None, ascii: bool = False,
            clip_z: float | None = None) -> str:
    method = method.lower()
    params = dict(params or {})
    e = REGISTRY.table(name)
    df: pd.DataFrame = e.obj
    X, cols, notes = _matrix(e, features, standardize=standardize, clip_z=clip_z)
    standardized = bool(notes.get("standardized"))
    n, d = X.shape
    if method not in SAMPLE_CAPS:
        raise EngineError(f"Unknown method '{method}'. Choose from {sorted(SAMPLE_CAPS)}.")
    cap = sample if sample is not None else SAMPLE_CAPS[method]
    sub = _sample_rows(n, cap, seed)
    Xw = X if sub is None else X[sub]
    rows = np.arange(n) if sub is None else sub
    t0 = time.time()
    extra: dict[str, Any] = {}

    # optional PCA pre-reduction for manifold methods (standard practice when d is large)
    if pre_pca is None and method in ("tsne", "umap", "isomap", "lle", "spectral", "mds") and d > 50:
        pre_pca = 50
    if pre_pca and pre_pca < d and method not in ("pca", "svd", "random", "ica", "kpca"):
        from sklearn.decomposition import PCA
        p = PCA(n_components=min(pre_pca, len(Xw) - 1), random_state=seed).fit(Xw)
        Xw = p.transform(Xw)
        extra["pre_pca"] = {"components": int(Xw.shape[1]), "variance_kept": round(float(p.explained_variance_ratio_.sum()), 4)}

    k = int(n_components)
    if k < 1 or k >= min(len(Xw), Xw.shape[1] + 1):
        raise EngineError(f"n_components={k} invalid for data with shape {Xw.shape}.")

    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=k, random_state=seed, **params)
        Y = model.fit_transform(Xw)
        extra["explained_variance_ratio"] = [round(float(v), 4) for v in model.explained_variance_ratio_]
        extra["cumulative"] = round(float(model.explained_variance_ratio_.sum()), 4)
        load = model.components_
        extra["loadings"] = load                      # full matrix (k x d), read with loadings()
        extra["top_loadings"] = {
            f"c{i + 1}": [f"{cols[j]}({load[i, j]:+.2f})" for j in np.argsort(-np.abs(load[i]))[:5]] for i in range(min(k, 10))
        }
    elif method == "svd":
        from sklearn.decomposition import TruncatedSVD
        model = TruncatedSVD(n_components=k, random_state=seed, **params)
        Y = model.fit_transform(Xw)
        extra["explained_variance_ratio"] = [round(float(v), 4) for v in model.explained_variance_ratio_]
    elif method == "ica":
        from sklearn.decomposition import FastICA
        Y = FastICA(n_components=k, random_state=seed, **params).fit_transform(Xw)
    elif method == "random":
        from sklearn.random_projection import GaussianRandomProjection
        Y = GaussianRandomProjection(n_components=k, random_state=seed, **params).fit_transform(Xw)
    elif method == "kpca":
        from sklearn.decomposition import KernelPCA
        params.setdefault("kernel", "rbf")
        Y = KernelPCA(n_components=k, random_state=seed, **params).fit_transform(Xw)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        params.setdefault("perplexity", float(min(30, max(5, (len(Xw) - 1) // 3))))
        params.setdefault("init", "pca")
        params.setdefault("learning_rate", "auto")
        Y = TSNE(n_components=k, random_state=seed, **params).fit_transform(Xw)
    elif method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise EngineError("UMAP needs the optional package umap-learn (pip install umap-learn). Alternatives: tsne, isomap, spectral.") from exc
        params.setdefault("n_neighbors", 15)
        params.setdefault("min_dist", 0.1)
        Y = umap.UMAP(n_components=k, random_state=seed, **params).fit_transform(Xw)
    elif method == "isomap":
        from sklearn.manifold import Isomap
        params.setdefault("n_neighbors", min(10, len(Xw) - 1))
        Y = Isomap(n_components=k, **params).fit_transform(Xw)
    elif method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding
        params.setdefault("n_neighbors", min(12, len(Xw) - 1))
        Y = LocallyLinearEmbedding(n_components=k, random_state=seed, **params).fit_transform(Xw)
    elif method == "mds":
        from sklearn.manifold import MDS
        params.setdefault("normalized_stress", "auto")
        Y = MDS(n_components=k, random_state=seed, **params).fit_transform(Xw)
    elif method == "spectral":
        from sklearn.manifold import SpectralEmbedding
        Y = SpectralEmbedding(n_components=k, random_state=seed, **params).fit_transform(Xw)
    else:  # pragma: no cover
        raise EngineError(f"method {method} not implemented")
    runtime = time.time() - t0

    # quality: trustworthiness on a sub-sample (neighbourhood preservation, 1.0 is perfect)
    tw = None
    if method not in ("pca", "svd", "random") or True:
        from sklearn.manifold import trustworthiness
        m = min(2000, len(Xw))
        rng = np.random.default_rng(seed)
        ii = rng.choice(len(Xw), size=m, replace=False) if m < len(Xw) else np.arange(len(Xw))
        try:
            tw = float(trustworthiness(Xw[ii], Y[ii], n_neighbors=min(10, m - 2)))
        except Exception:  # noqa: BLE001
            tw = None

    out_name = new_name or REGISTRY.unique(f"{name}__{method}{k}")
    emb = pd.DataFrame(Y, columns=[f"c{i + 1}" for i in range(k)], index=df.index[rows])
    emb["__row__"] = rows
    carried: list[str] = []
    for c in df.columns:
        if c in cols:
            continue
        # label-like columns ride along for colouring and annotation: strings, categoricals, datetimes, derived labels
        if (not pd.api.types.is_numeric_dtype(df[c])) or pd.api.types.is_datetime64_any_dtype(df[c]) or c in e.derived_cols():
            if len(carried) < 12:
                emb[c] = df[c].to_numpy()[rows]
                carried.append(c)
    meta = {"method": method, "parent": name, "rows": rows, "params": params, "features": cols,
            "standardize": standardized, "clip_z": clip_z, "derived_cols": [c for c in carried if c in e.derived_cols()] + ["__row__"],
            "carried_cols": carried, "runtime_s": round(runtime, 2), **extra}
    REGISTRY.put(Entry(out_name, "table", emb, meta))

    L = [f"stored embedding '{out_name}': {len(emb):,} rows x {k} components (method={method}, {runtime:.1f}s)"]
    L.append(f"input: {n:,} rows x {d} features" + (f", random sample of {len(rows):,} rows (sample={cap})" if sub is not None else "")
             + (", standardized" if standardized else ", raw scales") + (f", clipped at +-{clip_z:g} sd" if (clip_z and standardized) else ""))
    nt = _fmt_notes(notes)
    if nt:
        L.append(nt)
    if method in ("tsne", "umap", "isomap", "lle", "spectral", "mds") and len(Xw) < 200:
        L.append(f"caution: {method} on {len(Xw)} rows is unreliable (too few points for neighbourhood statistics); prefer pca/pairs here")
    if "pre_pca" in extra:
        L.append(f"pre-PCA to {extra['pre_pca']['components']} comps kept {extra['pre_pca']['variance_kept']:.1%} variance")
    if "explained_variance_ratio" in extra:
        evr = extra["explained_variance_ratio"]
        shown = evr if len(evr) <= 12 else evr[:12]
        L.append(f"explained variance ratio: {shown}{' ...' if len(evr) > 12 else ''} (total {extra.get('cumulative', sum(evr)):.3f})")
    if "top_loadings" in extra:
        L.append("top loadings: " + "; ".join(f"{c}: {', '.join(v)}" for c, v in extra["top_loadings"].items())
                 + (f" (c11..c{k} omitted)" if k > 10 else "") + f"; full vectors: loadings('{out_name}', component=1)")
    if tw is not None:
        L.append(f"trustworthiness(k=10) = {tw:.3f}  (1.0 = neighbourhoods perfectly preserved; <0.85 means the 2-D picture misrepresents local structure)")
    if params:
        L.append(f"params: {params}")
    rng_box = ", ".join(f"c{i + 1}:[{Y[:, i].min():.3g}, {Y[:, i].max():.3g}]" for i in range(k))
    L.append(f"ranges: {rng_box}")
    if carried:
        L.append(f"carried label columns for colouring: {carried}")
    if ascii and k >= 2:
        L.append("density (c1 horizontal, c2 vertical):\n" + _ascii_density(Y[:, 0], Y[:, 1]))
    L.append(f"next: plot('{out_name}', kind='scatter', color_by=<label column>) to see it.")
    return clip("\n".join(L))


def loadings(name: str, component: int = 1, top: int = 15, all_features: bool = False) -> str:
    """Full loading vector of one principal component of a stored PCA embedding (sorted by |loading|, both signs),
    or the whole vector in feature order when all_features=True (handy for yield-curve / spectral shapes)."""
    e = REGISTRY.table(name)
    load = e.meta.get("loadings")
    if load is None:
        raise EngineError(f"'{name}' has no stored loadings; it must be an embedding made with project(method='pca').")
    feats = list(e.meta.get("features", []))
    k = load.shape[0]
    if not 1 <= int(component) <= k:
        raise EngineError(f"component must be between 1 and {k}")
    v = np.asarray(load[int(component) - 1], dtype=float)
    evr = e.meta.get("explained_variance_ratio", [])
    head = f"PC{component} of '{name}' ({len(feats)} features" + (f", {evr[component - 1]:.1%} of variance" if len(evr) >= component else "") + ")"
    if all_features:
        df = pd.DataFrame({"feature": feats, "loading": np.round(v, 4)})
        return clip(head + " in feature order:\n" + frame_to_text(df, max_rows=min(len(df), MAX_ROWS), index=False))
    order = np.argsort(-np.abs(v))[: int(top)]
    pos = [f"{feats[j]}({v[j]:+.3f})" for j in order if v[j] > 0]
    neg = [f"{feats[j]}({v[j]:+.3f})" for j in order if v[j] < 0]
    sign_note = "all loadings share one sign (a level / size factor)" if (np.all(v >= 0) or np.all(v <= 0)) else "mixed signs (a contrast)"
    return clip(f"{head}: {sign_note}\n  positive: {', '.join(pos) or 'none'}\n  negative: {', '.join(neg) or 'none'}\n"
                f"  |loading| summary: max={np.abs(v).max():.3f} median={np.median(np.abs(v)):.3f}; sum of squares = {float(v @ v):.3f}")


def set_labels(name: str, columns: Sequence[str] | str, unset: bool = False) -> str:
    """Declare columns as labels / coordinates (excluded from feature matrices, available to color_by and annotate),
    or with unset=True put them back among the features. Use it for integer class codes, years, hours, lat/lon, ids."""
    e = REGISTRY.table(name)
    df: pd.DataFrame = e.obj
    cols = resolve_columns(df, columns, e)
    e.meta.setdefault("derived_cols", [])
    if unset:
        e.meta["derived_cols"] = [c for c in e.meta["derived_cols"] if c not in cols]
        return f"{cols} are features again in '{name}'. Label columns now: {e.derived_cols()}"
    for c in cols:
        if c not in e.meta["derived_cols"]:
            e.meta["derived_cols"].append(c)
    return f"marked {cols} as label columns in '{name}' (excluded from describe/project/cluster features). Label columns now: {e.derived_cols()}"


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


def cluster(name: str, method: str = "kmeans", k: int | None = None, features=None, standardize: Any = "auto",
            seed: int = 0, params: dict[str, Any] | None = None, label_col: str | None = None,
            clip_z: float | None = None, k_max: int = 12) -> str:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    method = method.lower()
    params = dict(params or {})
    e = REGISTRY.table(name)
    X, cols, notes = _matrix(e, features, standardize=standardize, clip_z=clip_z)
    standardized = bool(notes.get("standardized"))
    n = len(X)
    rng = np.random.default_rng(seed)
    sub = rng.choice(n, size=min(5000, n), replace=False)
    t0 = time.time()
    chosen = ""
    k_table = ""
    if method == "kmeans":
        from sklearn.cluster import KMeans
        if k is None:
            rows = []
            for kk in range(2, min(int(k_max), n - 1) + 1):
                lab = KMeans(n_clusters=kk, n_init=4, random_state=seed).fit_predict(X[sub])
                rows.append((kk, silhouette_score(X[sub], lab), calinski_harabasz_score(X[sub], lab)))
            best_sil = max(rows, key=lambda r: r[1])
            best_ch = max(rows, key=lambda r: r[2])
            k = best_sil[0]
            # silhouette favours the coarsest split; show the largest k that is still within 10% of the best score
            within = [r for r in rows if r[1] >= 0.9 * best_sil[1]]
            alt = max(within, key=lambda r: r[0])[0]
            chosen = f" (k={k} chosen by silhouette over 2..{rows[-1][0]} on a sample: {best_sil[1]:.3f})"
            k_table = ("silhouette by k: " + "  ".join(f"{kk}:{s:.3f}" for kk, s, _ in rows)
                       + f"\n  silhouette rewards coarse splits; Calinski-Harabasz prefers k={best_ch[0]}"
                       + (f"; the finest k within 10% of the best silhouette is {alt}" if alt != k else "")
                       + ". If you expect more groups than this, pass k explicitly.")
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed, **params).fit_predict(X)
    elif method == "gmm":
        from sklearn.mixture import GaussianMixture
        if k is None:
            rows = []
            for kk in range(2, min(int(k_max), n - 1) + 1):
                g = GaussianMixture(n_components=kk, random_state=seed, covariance_type=params.get("covariance_type", "full")).fit(X[sub])
                rows.append((kk, g.bic(X[sub])))
            k = min(rows, key=lambda r: r[1])[0]
            chosen = f" (k={k} chosen by BIC over 2..{rows[-1][0]} on a sample)"
            k_table = "BIC by k (lower is better): " + "  ".join(f"{kk}:{b:.0f}" for kk, b in rows)
        labels = GaussianMixture(n_components=k, random_state=seed, **params).fit_predict(X)
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        if "eps" not in params:
            d = _knn_dists(X[sub], 5)[:, -1]
            params["eps"] = float(np.quantile(d, 0.9))
            chosen = f" (eps set to 90th percentile of 5-NN distance = {params['eps']:.3g})"
        params.setdefault("min_samples", 5)
        labels = DBSCAN(**params).fit_predict(X)
    elif method == "hdbscan":
        from sklearn.cluster import HDBSCAN
        params.setdefault("min_cluster_size", max(5, n // 100))
        params.setdefault("copy", True)
        labels = HDBSCAN(**params).fit_predict(X)
    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        if n > 20000:
            raise EngineError("agglomerative clustering is O(n^2) memory; use sample via subset() first or choose kmeans/hdbscan.")
        labels = AgglomerativeClustering(n_clusters=k or 4, **params).fit_predict(X)
    else:
        raise EngineError("Unknown method. Choose kmeans, gmm, dbscan, hdbscan, agglomerative.")
    runtime = time.time() - t0

    col = label_col or f"cluster_{method}"
    df: pd.DataFrame = e.obj
    df[col] = pd.Series(labels, index=df.index).astype(int).astype(str).to_numpy()
    e.meta.setdefault("derived_cols", [])
    if col not in e.meta["derived_cols"]:
        e.meta["derived_cols"].append(col)
    # propagate to parent table if this is an embedding
    propagated = ""
    if e.meta.get("parent") and e.meta.get("rows") is not None and e.meta["parent"] in REGISTRY.names():
        p = REGISTRY.get(e.meta["parent"])
        if p.kind == "table":
            pcol = col
            vals = np.full(len(p.obj), "unassigned", dtype=object)
            vals[np.asarray(e.meta["rows"])] = df[col].to_numpy()
            p.obj[pcol] = vals
            p.meta.setdefault("derived_cols", [])
            if pcol not in p.meta["derived_cols"]:
                p.meta["derived_cols"].append(pcol)
            propagated = f"; also written to parent '{p.name}' column '{pcol}'"

    uniq, counts = np.unique(labels, return_counts=True)
    L = [f"clustered '{name}' with {method}{chosen}: {len(uniq)} labels in {runtime:.1f}s; label column '{col}' added{propagated}"]
    L.append(f"features: {len(cols)}, " + ("standardized" if standardized else "raw scales") + (f", clipped at +-{clip_z:g} sd" if (clip_z and standardized) else ""))
    nt = _fmt_notes(notes)
    if nt:
        L.append(nt)
    if k_table:
        L.append(k_table)
    sizes = ", ".join(f"{u}:{c}" for u, c in zip(uniq, counts))
    L.append(f"sizes: {sizes}" + ("  (-1 = noise)" if -1 in uniq else ""))
    tiny = [(int(u), int(c)) for u, c in zip(uniq, counts) if u >= 0 and c < max(5, 0.005 * n)]
    if tiny and method in ("kmeans", "gmm", "agglomerative"):
        L.append(f"warning: {len(tiny)} cluster(s) hold < 0.5% of rows {tiny[:6]}: either outliers or a rare population; inspect them with "
                 f"subset(where=\"{col} == '<label>'\") before counting them as groups"
                 + (" (standardized near-constant or heavy-tailed features manufacture such clusters: retry with clip_z=10 or standardize=False)" if standardized and not clip_z else ""))
    mask = labels >= 0
    if len(set(labels[mask])) >= 2 and mask.sum() >= 10:
        ii = sub[mask[sub]] if len(sub) < n else np.where(mask)[0]
        if len(set(labels[ii])) >= 2:
            L.append(f"silhouette (sample) = {silhouette_score(X[ii], labels[ii]):.3f}  [-1..1, higher is better];  Davies-Bouldin = {davies_bouldin_score(X[ii], labels[ii]):.3f} [lower is better]")
    # distinguishing features per cluster (standardized centroid deviation)
    if standardized and len(cols) >= 2:
        parts = []
        for u in uniq[:12]:
            if u < 0:
                continue
            c = X[labels == u].mean(axis=0)
            top = np.argsort(-np.abs(c))[:4]
            parts.append(f"  {u}: " + ", ".join(f"{cols[j]}({c[j]:+.2f}sd)" for j in top))
        L.append("centroid signature (features most displaced from the global mean, in SD units):\n" + "\n".join(parts))
    L.append(f"next: plot(<embedding name>, kind='scatter', color_by='{col}')")
    return clip("\n".join(L))


# --------------------------------------------------------------------------- #
# Neighbourhoods and subsets
# --------------------------------------------------------------------------- #


def neighbors(name: str, index: Any = None, point: Sequence[float] | None = None, k: int = 10,
              metric: str = "euclidean", features=None, standardize: bool = False, columns=None) -> str:
    from sklearn.neighbors import NearestNeighbors
    e = REGISTRY.table(name)
    df: pd.DataFrame = e.obj
    X, cols, notes = _matrix(e, features, standardize=standardize)
    if (index is None) == (point is None):
        raise EngineError("Give exactly one of index (row label or position) or point (vector of feature values).")
    if index is not None:
        if index in df.index:
            pos = int(np.where(df.index == index)[0][0])
        else:
            try:
                pos = int(index)
            except (TypeError, ValueError) as exc:
                raise EngineError(f"index {index!r} not found in the table index and not an integer position") from exc
            if not 0 <= pos < len(df):
                raise EngineError(f"position {pos} out of range 0..{len(df) - 1}")
        q = X[pos: pos + 1]
        where = f"row {df.index[pos]!r} (position {pos})"
    else:
        q = np.asarray(point, dtype=float).reshape(1, -1)
        if q.shape[1] != X.shape[1]:
            raise EngineError(f"point has {q.shape[1]} values but the feature space has {X.shape[1]}: {cols[:20]}")
        where = "the given point"
    kk = min(k + (1 if index is not None else 0), len(X))
    nn = NearestNeighbors(n_neighbors=kk, metric=metric).fit(X)
    dist, idx = nn.kneighbors(q)
    dist, idx = dist[0], idx[0]
    if index is not None:
        keep = idx != pos
        dist, idx = dist[keep][:k], idx[keep][:k]
    # context: how these distances compare with typical distances
    rng = np.random.default_rng(0)
    ref = X[rng.choice(len(X), size=min(2000, len(X)), replace=False)]
    from scipy.spatial.distance import cdist
    allD = cdist(q, ref, metric=metric)[0]
    allD = allD[allD > 0]
    show_cols = resolve_columns(df, columns, e) if columns else list(df.columns)[: min(12, df.shape[1])]
    res = df.iloc[idx][show_cols].copy()
    res.insert(0, "distance", dist)
    res.insert(1, "position", idx)
    L = [f"{len(idx)} nearest neighbours of {where} in '{name}' ({metric}, {len(cols)} features{', standardized' if notes.get('standardized') else ''}):"]
    nt = _fmt_notes(notes)
    if nt:
        L.append(nt)
    L.append(frame_to_text(res, max_rows=k, max_cols=len(show_cols) + 2))
    if len(allD):
        L.append(f"context: distance to a random point: median={np.median(allD):.4g}, 5%={np.quantile(allD, .05):.4g}, min(sample)={allD.min():.4g}; "
                 f"relative contrast for this query={(allD.max() - allD.min()) / allD.min():.3f}")
    return clip("\n".join(L))


def subset(name: str, where: str | None = None, columns=None, top_variance: int | None = None,
           sample: int | None = None, seed: int = 0, new_name: str | None = None, preview: int = 20,
           sort_by: str | None = None, ascending: bool = True) -> str:
    e = REGISTRY.table(name)
    df: pd.DataFrame = e.obj
    out = df
    notes = []
    if where:
        try:
            out = out.query(where, engine="python")
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"query failed: {exc}. Use pandas query syntax, e.g. \"price > 10 and sector == 'tech'\"; backtick-quote odd column names.") from exc
        notes.append(f"filter kept {len(out):,} of {len(df):,} rows")
    if top_variance:
        cols = resolve_columns(out, f"top_variance:{int(top_variance)}", e)
        keep_labels = [c for c in out.columns if (not pd.api.types.is_numeric_dtype(out[c])) or c in e.derived_cols()]
        out = out[cols + [c for c in keep_labels if c not in cols]]
        notes.append(f"kept top {len(cols)} features by variance (+ {len(keep_labels)} label columns)")
    if columns:
        out = out[resolve_columns(out, columns, e)]
    if sort_by:
        out = out.sort_values(resolve_columns(out, sort_by, e), ascending=ascending)
    if sample and len(out) > sample:
        out = out.sample(n=int(sample), random_state=seed).sort_index()
        notes.append(f"random sample of {sample:,} rows")
    L = []
    if new_name:
        nm = new_name
        REGISTRY.put(Entry(nm, "table", out.copy(), {"parent": name, "op": "subset", "where": where,
                                                      "derived_cols": [c for c in e.derived_cols() if c in out.columns]}))
        L.append(f"stored '{nm}': {out.shape[0]:,} rows x {out.shape[1]} cols")
    if notes:
        L.append("; ".join(notes))
    if preview:
        L.append(frame_to_text(out.head(preview) if not sort_by else out.head(preview), max_rows=preview))
    return clip("\n".join(L) or "empty result")


# --------------------------------------------------------------------------- #
# Preprocessing transforms (tables)
# --------------------------------------------------------------------------- #

_PERIODS = {"month": lambda idx: idx.month, "dayofweek": lambda idx: idx.dayofweek, "weekday": lambda idx: idx.dayofweek,
            "hour": lambda idx: idx.hour, "dayofyear": lambda idx: idx.dayofyear, "quarter": lambda idx: idx.quarter,
            "year": lambda idx: idx.year, "minute": lambda idx: idx.minute}

TABLE_OPS = ("log1p", "log", "sqrt", "zscore", "minmax", "clip", "winsorize", "rank", "demean", "detrend", "diff",
             "pct_change", "log_return", "row_normalize", "deseasonalize", "residualize", "dropna", "abs", "cumsum")


def transform(name: str, op: str, columns=None, new_name: str | None = None, params: dict[str, Any] | None = None) -> str:
    """Apply a preprocessing step to the numeric feature columns of a table and store the result as a new table.

    Elementwise: log1p, log (params offset), sqrt, abs, clip (lo/hi values or z in sd units), winsorize (q, default 0.01),
    rank (to [0,1]). Per column: zscore (optional clip), minmax, demean, detrend (linear in row order or vs `by`).
    Along rows (time series; sort with by=<column>, optional group_by): diff, pct_change, log_return, cumsum (params periods).
    Per row: row_normalize (each row scaled to sum `target`, default 1e4: library-size normalization for counts).
    Seasonality: deseasonalize (params by=<datetime column or index>, period=month|dayofweek|hour|dayofyear|quarter)
    subtracts the per-period mean of every column -> anomalies. Factor removal: residualize (params on=[cols]) regresses
    every feature on the given columns (+ intercept) and keeps the residuals. dropna drops rows with missing features.
    Label / id / datetime columns are carried over unchanged.
    """
    op = op.lower().strip()
    if op not in TABLE_OPS:
        raise EngineError(f"Unknown op '{op}'. Choose from {TABLE_OPS}")
    params = dict(params or {})
    e = REGISTRY.table(name)
    df: pd.DataFrame = e.obj
    fnotes: dict[str, Any] = {}
    num = _numeric_frame(e, columns, fnotes)
    feats = list(num.columns)
    out = df.copy()
    X = num.to_numpy(dtype=float)
    note = ""
    sort_cols: list[str] = []
    if op in ("diff", "pct_change", "log_return", "cumsum", "detrend", "deseasonalize") and params.get("by") and params["by"] != "index":
        by = resolve_columns(df, params["by"], e)[0]
        groups = resolve_columns(df, params["group_by"], e) if params.get("group_by") else []
        sort_cols = groups + [by]
        out = out.sort_values(sort_cols, kind="stable")
        X = out[feats].to_numpy(dtype=float)
        note = f"rows sorted by {sort_cols}"
    with np.errstate(divide="ignore", invalid="ignore"):
        if op == "log1p":
            neg = int((X < -1).sum())
            Y = np.log1p(np.where(X < -1, np.nan, X))
            note = f"{neg} values below -1 became NaN" if neg else note
        elif op == "log":
            off = float(params.get("offset", 0.0))
            bad = int((X + off <= 0).sum())
            Y = np.log(np.where(X + off > 0, X + off, np.nan))
            note = f"{bad} non-positive values became NaN (use params={{'offset': c}})" if bad else note
        elif op == "sqrt":
            Y = np.sqrt(np.where(X < 0, np.nan, X))
        elif op == "abs":
            Y = np.abs(X)
        elif op == "zscore":
            mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
            Y = (X - mu) / np.where(sd > 0, sd, 1.0)
            if params.get("clip") is not None:
                cz = float(params["clip"]); Y = np.clip(Y, -cz, cz)
        elif op == "minmax":
            lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
            Y = (X - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
        elif op == "clip":
            if "z" in params:
                z = float(params["z"]); mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
                Y = np.clip(X, mu - z * sd, mu + z * sd)
            else:
                Y = np.clip(X, params.get("lo", -np.inf), params.get("hi", np.inf))
        elif op == "winsorize":
            q = float(params.get("q", 0.01))
            lo, hi = np.nanquantile(X, q, axis=0), np.nanquantile(X, 1 - q, axis=0)
            Y = np.clip(X, lo, hi)
        elif op == "rank":
            Y = pd.DataFrame(X).rank(pct=True).to_numpy()
        elif op == "demean":
            Y = X - np.nanmean(X, axis=0)
        elif op == "detrend":
            t = np.arange(len(X), dtype=float)
            if params.get("by") and params["by"] != "index":
                bycol = out[sort_cols[-1]]
                t = (bycol.astype("int64").to_numpy(dtype=float) if pd.api.types.is_datetime64_any_dtype(bycol) else bycol.to_numpy(dtype=float))
            t = (t - t.mean()) / (t.std() or 1.0)
            A = np.column_stack([np.ones_like(t), t])
            Y = np.empty_like(X)
            for j in range(X.shape[1]):
                m = np.isfinite(X[:, j])
                coef, *_ = np.linalg.lstsq(A[m], X[m, j], rcond=None)
                Y[:, j] = X[:, j] - A @ coef
        elif op in ("diff", "pct_change", "log_return", "cumsum"):
            per = int(params.get("periods", 1))
            frame = pd.DataFrame(X, columns=feats)
            gkey = out[sort_cols[:-1]].astype(str).agg("|".join, axis=1).to_numpy() if len(sort_cols) > 1 else None
            def _apply(block: pd.DataFrame) -> pd.DataFrame:
                if op == "diff":
                    return block.diff(per)
                if op == "pct_change":
                    return block.pct_change(per, fill_method=None)
                if op == "cumsum":
                    return block.cumsum()
                return np.log(block / block.shift(per))
            if gkey is None:
                Y = _apply(frame).to_numpy()
            else:
                Y = frame.groupby(gkey, sort=False, group_keys=False).apply(_apply).to_numpy()
            if params.get("dropna", True) and op != "cumsum":
                keep = np.isfinite(Y).any(axis=1)          # the first `per` rows of each series have no predecessor
                out = out.iloc[np.where(keep)[0]]
                Y = Y[keep]
                note = (note + "; " if note else "") + f"{per} leading row(s) per series dropped"
        elif op == "row_normalize":
            target = float(params.get("target", 1e4))
            s = np.nansum(X, axis=1, keepdims=True)
            Y = X / np.where(s > 0, s, np.nan) * target
            note = f"each row scaled to sum {target:g}"
        elif op == "deseasonalize":
            by = params.get("by")
            period = str(params.get("period", "month")).lower()
            if period not in _PERIODS:
                raise EngineError(f"period must be one of {list(_PERIODS)}")
            if by in (None, "index"):
                if not pd.api.types.is_datetime64_any_dtype(out.index):
                    dts = [c for c in out.columns if pd.api.types.is_datetime64_any_dtype(out[c])]
                    if len(dts) != 1:
                        raise EngineError("deseasonalize needs params={'by': <datetime column>} (no unique datetime column or index found)")
                    by = dts[0]
                    idx = pd.DatetimeIndex(out[by])
                else:
                    idx = pd.DatetimeIndex(out.index)
            else:
                by = resolve_columns(out, by, e)[0]
                if not pd.api.types.is_datetime64_any_dtype(out[by]):
                    raise EngineError(f"'{by}' is not a datetime column")
                idx = pd.DatetimeIndex(out[by])
            key = np.asarray(_PERIODS[period](idx))
            frame = pd.DataFrame(X, columns=feats)
            means = frame.groupby(key).transform("mean").to_numpy()
            Y = X - means
            note = f"subtracted per-{period} means (climatology) of each column; {len(np.unique(key))} periods"
        elif op == "residualize":
            on = params.get("on")
            if not on:
                raise EngineError("residualize needs params={'on': [columns]} (e.g. a market-return column)")
            oncols = resolve_columns(out, on, e, numeric_only=True)
            Z = out[oncols].to_numpy(dtype=float)
            A = np.column_stack([np.ones(len(Z)), Z])
            targets = [c for c in feats if c not in oncols]
            Y = X.copy()
            m = np.isfinite(A).all(axis=1)
            r2s = []
            for j, c in enumerate(feats):
                if c in oncols:
                    continue
                mm = m & np.isfinite(X[:, j])
                coef, *_ = np.linalg.lstsq(A[mm], X[mm, j], rcond=None)
                fit = A @ coef
                Y[:, j] = X[:, j] - fit
                ss_res = np.nansum((X[mm, j] - fit[mm]) ** 2); ss_tot = np.nansum((X[mm, j] - X[mm, j].mean()) ** 2)
                r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
            note = f"regressed {len(targets)} features on {oncols} (+ intercept); median R^2 removed = {np.nanmedian(r2s):.3f}"
        elif op == "dropna":
            keep = np.isfinite(X).all(axis=1)
            out = out.iloc[np.where(keep)[0]]
            Y = X[keep]
            note = f"dropped {int((~keep).sum())} rows with missing features"
        else:  # pragma: no cover
            raise EngineError(f"op {op} not implemented")
    out = out.copy()
    out[feats] = Y
    nm = new_name or REGISTRY.unique(f"{name}__{op}")
    meta = {"parent": name, "op": op, "params": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in params.items()},
            "derived_cols": [c for c in e.derived_cols() if c in out.columns], "source": e.meta.get("source")}
    REGISTRY.put(Entry(nm, "table", out, meta))
    L = [f"stored '{nm}': {out.shape[0]:,} rows x {out.shape[1]} cols = {op} of {len(feats)} feature column(s) of '{name}'" + (f"; {note}" if note else "")]
    if fnotes.get("excluded_identifier_cols"):
        L.append(f"left unchanged (identifier-like): {fnotes['excluded_identifier_cols']}")
    show = feats[: min(5, len(feats))]
    before = pd.DataFrame({"before_mean": np.nanmean(X[:, : len(show)], axis=0), "before_sd": np.nanstd(X[:, : len(show)], axis=0),
                           "after_mean": np.nanmean(Y[:, : len(show)], axis=0), "after_sd": np.nanstd(Y[:, : len(show)], axis=0),
                           "after_nan%": np.round(100 * np.mean(~np.isfinite(Y[:, : len(show)]), axis=0), 2)}, index=show)
    L.append(frame_to_text(before, max_rows=5))
    L.append(f"next: describe('{nm}') / project('{nm}', ...) / cluster('{nm}', ...)")
    return clip("\n".join(L))


# --------------------------------------------------------------------------- #
# Tensors: build, stack, slice
# --------------------------------------------------------------------------- #


def _natural_key(v: Any) -> tuple:
    """Sort key that orders embedded numbers numerically: b2 < b10, m1y < m10y, 2024-Q2 < 2024-Q10."""
    parts = re.split(r"(\d+\.?\d*)", str(v))
    key: list[Any] = []
    for p in parts:
        if p == "":
            continue
        try:
            key.append((0, float(p), ""))
        except ValueError:
            key.append((1, 0.0, p.lower()))
    return tuple(key)


def _sorted_unique(s: pd.Series, sort: bool = True, order: Sequence[Any] | None = None) -> np.ndarray:
    """Unique coordinate values along one axis.

    Numeric and datetime values sort numerically. Strings sort *naturally* (numbers inside the text compare as numbers),
    so 'b2' comes before 'b10'. sort=False keeps first-appearance order (what a wide table's column order encodes).
    order=[...] pins an explicit order; values present in the data but missing from `order` are appended (sorted).
    """
    u = pd.unique(s.dropna())
    if order is not None:
        given = [v for v in order]
        present = set(map(str, u))
        head = [v for v in given if str(v) in present]
        rest = [v for v in u if str(v) not in set(map(str, head))]
        try:
            rest = sorted(rest, key=_natural_key)
        except TypeError:
            pass
        out = list(head) + list(rest)
        return np.array(out, dtype=u.dtype if u.dtype.kind in "iufM" else object)
    if not sort:
        return np.asarray(u)
    if u.dtype.kind in "iufMm":
        return np.sort(u)
    return np.array(sorted(u, key=_natural_key), dtype=object)


def tensorize(name: str, axes: Sequence[str], values: Sequence[str] | None = None, agg: str = "mean",
              new_name: str | None = None, order: dict[str, Sequence[Any]] | None = None, sort: bool = True) -> str:
    """Long/tidy table -> labelled N-d tensor (axes -> dims; several value columns -> trailing 'variable' dim).

    Coordinates: numeric/datetime axes sort numerically, text axes sort naturally (b2 < b10); sort=False keeps
    first-appearance order; order={'tenor': ['1m','3m','1y','10y']} pins an order per axis.
    """
    e = REGISTRY.table(name)
    df: pd.DataFrame = e.obj
    axes = resolve_columns(df, list(axes), e)
    if values is None:
        values = [c for c in df.columns if c not in axes and pd.api.types.is_numeric_dtype(df[c]) and c not in e.derived_cols()]
        if not values:
            raise EngineError("No numeric value columns found besides the axes; pass values=[...].")
    else:
        values = resolve_columns(df, list(values), e, numeric_only=True)
    order = dict(order or {})
    bad_order = [k for k in order if k not in axes]
    if bad_order:
        raise EngineError(f"order keys {bad_order} are not among the axes {axes}")
    work = df[axes + list(values)].dropna(subset=axes)
    n_unique_combos = len(work[axes].drop_duplicates())
    n_dups = len(work) - n_unique_combos
    coords = {a: _sorted_unique(work[a], sort=sort, order=order.get(a)) for a in axes}
    sizes = [len(coords[a]) for a in axes]
    cells = int(np.prod(sizes, dtype=np.int64)) * len(values)
    if cells * 8 > MAX_TENSOR_BYTES:
        raise EngineError(f"Tensor would need {cells:,} cells ({_fmt_bytes(cells * 8)}) > HIDIM_MAX_TENSOR_BYTES. "
                          f"Sizes: {dict(zip(axes, sizes))}. Coarsen an axis (bin it) or drop one.")
    g = work.groupby(axes, sort=True, observed=True)[list(values)].agg(agg)
    full = pd.MultiIndex.from_product([coords[a] for a in axes], names=axes) if len(axes) > 1 else pd.Index(coords[axes[0]], name=axes[0])
    arr = g.reindex(full).to_numpy(dtype=float).reshape(sizes + [len(values)])
    dims = list(axes)
    if len(values) == 1:
        arr = arr[..., 0]
    else:
        dims = dims + ["variable"]
        coords["variable"] = np.array(list(values), dtype=object)
    t = Tensor(arr, dims, coords)
    out_name = new_name or REGISTRY.unique(f"{name}__tensor")
    fill = float(np.isfinite(arr).mean())
    REGISTRY.put(Entry(out_name, "tensor", t, {"parent": name, "op": "tensorize", "params": {"axes": axes, "values": list(values), "agg": agg}}))
    L = [f"stored tensor '{out_name}': shape {'x'.join(map(str, t.shape))}, dims {dims}, {_fmt_bytes(t.nbytes())}"]
    L.append(f"fill={fill:.1%} of the full grid has data" + (f"; {n_dups:,} duplicate axis-combinations were aggregated with '{agg}'" if n_dups else ""))
    for d in t.dims:
        c = t.coords[d]
        L.append(f"  {d}: {len(c)} [{_fmt_num(c[0])} .. {_fmt_num(c[-1])}]")
    L.append(f"next: describe('{out_name}'), slice_tensor('{out_name}', select=..., reduce=...), plot('{out_name}', kind='tensor'|'surface'|'lines', select=...)")
    return clip("\n".join(L))


def tensor_from_wide(name: str, sep: str = "_", dims: Sequence[str] = ("series", "feature"),
                     index_dim: str | None = None, new_name: str | None = None, sort: bool = False,
                     order: dict[str, Sequence[Any]] | None = None) -> str:
    """Wide table with compound column names like AAPL_close -> tensor index x series x feature.

    The column order of the wide table is kept for the name-derived dims (sort=False), because that order usually
    carries meaning (tenors, wavelengths, ordered categories); the index dim is always sorted. order={dim: [...]} pins
    an explicit order; sort=True sorts naturally instead.
    """
    e = REGISTRY.table(name)
    df: pd.DataFrame = e.obj
    dims = list(dims)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in e.derived_cols()]
    bad = [c for c in num_cols if len(str(c).split(sep)) < len(dims)]
    if bad:
        raise EngineError(f"{len(bad)} numeric columns do not split into {len(dims)} parts with sep '{sep}': {bad[:8]}")
    idx_name = index_dim or (df.index.name or "row")
    long = df[num_cols].reset_index(names=idx_name).melt(id_vars=idx_name, var_name="__col__", value_name="value")
    parts = long["__col__"].str.split(sep, n=len(dims) - 1, expand=True)
    for i, d in enumerate(dims):
        long[d] = parts[i]
    order = dict(order or {})
    if not sort:
        # first-appearance order of each name part, taken from the wide column order
        for i, d in enumerate(dims):
            if d not in order:
                seen: list[str] = []
                for c in num_cols:
                    p = str(c).split(sep, len(dims) - 1)[i]
                    if p not in seen:
                        seen.append(p)
                order[d] = seen
    tmp = Entry("__tmp_wide__", "table", long.drop(columns="__col__"), {})
    REGISTRY.put(tmp)
    try:
        return tensorize("__tmp_wide__", [idx_name] + dims, ["value"], new_name=new_name or REGISTRY.unique(f"{name}__tensor"),
                         order=order).replace("'__tmp_wide__'", f"'{name}'")
    finally:
        REGISTRY.drop("__tmp_wide__")


def stack_datasets(names: Sequence[str], axis_name: str = "source", labels: Sequence[str] | None = None,
                   new_name: str | None = None) -> str:
    """Stack same-shaped tables (numeric block) or tensors along a new leading axis."""
    if len(names) < 2:
        raise EngineError("Need at least two datasets to stack.")
    ents = [REGISTRY.get(n) for n in names]
    kinds = {e.kind for e in ents}
    if len(kinds) != 1:
        raise EngineError("All datasets must be the same kind (all tables or all tensors).")
    labels = list(labels) if labels else list(names)
    if len(labels) != len(names):
        raise EngineError("labels must match names in length.")
    if ents[0].kind == "table":
        frames = [_numeric_frame(e) for e in ents]
        cols = list(frames[0].columns)
        for e, f in zip(ents, frames):
            if list(f.columns) != cols:
                raise EngineError(f"numeric columns of '{e.name}' differ from '{ents[0].name}': {list(f.columns)[:8]} vs {cols[:8]}")
            if len(f) != len(frames[0]):
                raise EngineError(f"'{e.name}' has {len(f)} rows, '{ents[0].name}' has {len(frames[0])}; align them first (subset/tensorize).")
        arr = np.stack([f.to_numpy(dtype=float) for f in frames])
        idx0 = ents[0].obj.index
        same_index = all(e.obj.index.equals(idx0) for e in ents)
        coords = {axis_name: np.array(labels, dtype=object),
                  "row": np.asarray(idx0.values) if same_index else np.arange(arr.shape[1]),
                  "column": np.array(cols, dtype=object)}
        t = Tensor(arr, [axis_name, "row", "column"], coords)
    else:
        t0: Tensor = ents[0].obj
        for e in ents:
            t: Tensor = e.obj
            if t.shape != t0.shape or t.dims != t0.dims:
                raise EngineError(f"tensor '{e.name}' shape/dims {t.shape}/{t.dims} differ from '{ents[0].name}' {t0.shape}/{t0.dims}")
        arr = np.stack([e.obj.data for e in ents])
        coords = {axis_name: np.array(labels, dtype=object), **{d: t0.coords[d] for d in t0.dims}}
        t = Tensor(arr, [axis_name] + list(t0.dims), coords)
    if t.nbytes() > MAX_TENSOR_BYTES:
        raise EngineError(f"stacked tensor would be {_fmt_bytes(t.nbytes())} > HIDIM_MAX_TENSOR_BYTES")
    out_name = new_name or REGISTRY.unique("stack_" + "_".join(names)[:40])
    REGISTRY.put(Entry(out_name, "tensor", t, {"op": "stack", "parent": list(names), "params": {"axis_name": axis_name}}))
    return clip(f"stored tensor '{out_name}': shape {'x'.join(map(str, t.shape))}, dims {t.dims}, {_fmt_bytes(t.nbytes())}\n"
                + "\n".join(f"  {d}: {len(t.coords[d])} [{_fmt_num(t.coords[d][0])} .. {_fmt_num(t.coords[d][-1])}]" for d in t.dims)
                + f"\nnext: describe('{out_name}') or slice_tensor('{out_name}', ...)")


def _coerce_like(coord: np.ndarray, v: Any) -> Any:
    if coord.dtype.kind == "M":
        return np.datetime64(pd.Timestamp(v))
    if coord.dtype.kind in "iuf":
        return float(v)
    return v


def _select_positions(coord: np.ndarray, spec: Any, dim: str) -> tuple[np.ndarray, str]:
    """Turn a selector into integer positions along one axis; returns (positions, note)."""
    numeric_like = coord.dtype.kind in "iufM"
    if isinstance(spec, dict):
        if "range" in spec:
            lo, hi = spec["range"]
            lo = _coerce_like(coord, lo) if lo is not None else None
            hi = _coerce_like(coord, hi) if hi is not None else None
            m = np.ones(len(coord), dtype=bool)
            if lo is not None:
                m &= coord >= lo
            if hi is not None:
                m &= coord <= hi
            pos = np.where(m)[0]
            return pos, f"{dim} in [{_fmt_num(lo)}, {_fmt_num(hi)}] -> {len(pos)}"
        if "isin" in spec:
            vals = [_coerce_like(coord, v) for v in spec["isin"]]
            pos = np.where(np.isin(coord, np.array(vals, dtype=coord.dtype if numeric_like else object)))[0]
            return pos, f"{dim} in {len(vals)} values -> {len(pos)} found"
        if "nearest" in spec:
            if not numeric_like:
                raise EngineError(f"'nearest' needs a numeric/datetime axis; '{dim}' is label-valued")
            v = _coerce_like(coord, spec["nearest"])
            diff = np.abs((coord - v).astype(float)) if coord.dtype.kind != "M" else np.abs((coord - v).astype("timedelta64[s]").astype(float))
            p = int(np.argmin(diff))
            return np.array([p]), f"{dim} nearest {_fmt_num(v)} -> {_fmt_num(coord[p])}"
        if "iloc" in spec:
            a, b = spec["iloc"]
            pos = np.arange(len(coord))[slice(a, b)]
            return pos, f"{dim} positions {a}:{b} -> {len(pos)}"
        if "positions" in spec:
            pos = np.array(spec["positions"], dtype=int)
            return pos, f"{dim} {len(pos)} positions"
        if "every" in spec:
            pos = np.arange(0, len(coord), int(spec["every"]))
            return pos, f"{dim} every {spec['every']} -> {len(pos)}"
        raise EngineError(f"Unknown selector {spec} for '{dim}'. Use value, {{'range':[lo,hi]}}, {{'isin':[...]}}, {{'nearest':v}}, {{'iloc':[a,b]}}, {{'positions':[...]}}, {{'every':n}}.")
    if isinstance(spec, (list, tuple)):
        return _select_positions(coord, {"isin": list(spec)}, dim)
    # scalar: exact match, else nearest for numeric/datetime
    v = _coerce_like(coord, spec)
    hits = np.where(coord == v)[0]
    if len(hits):
        return hits[:1], f"{dim}={_fmt_num(coord[hits[0]])}"
    if numeric_like:
        return _select_positions(coord, {"nearest": spec}, dim)
    raise EngineError(f"value {spec!r} not found on axis '{dim}'. Examples: {[_fmt_num(c) for c in coord[:8]]}")


_REDUCERS = {"mean": np.nanmean, "std": np.nanstd, "var": np.nanvar, "sum": np.nansum, "min": np.nanmin,
             "max": np.nanmax, "median": np.nanmedian,
             "count": lambda a, axis: np.isfinite(a).sum(axis=axis),
             "first": lambda a, axis: np.take(a, 0, axis=axis), "last": lambda a, axis: np.take(a, -1, axis=axis)}


def slice_tensor_obj(t: Tensor, select: dict[str, Any] | None = None, reduce: dict[str, str] | None = None) -> tuple[Tensor, list[str]]:
    notes: list[str] = []
    data = t.data
    dims = list(t.dims)
    coords = dict(t.coords)
    for dim, spec in (select or {}).items():
        if dim not in dims:
            raise EngineError(f"Unknown dim '{dim}'. Dims: {dims}")
        ax = dims.index(dim)
        pos, note = _select_positions(coords[dim], spec, dim)
        notes.append(note)
        if len(pos) == 0:
            raise EngineError(f"selector for '{dim}' matched nothing ({note})")
        data = np.take(data, pos, axis=ax)
        coords[dim] = coords[dim][pos]
        if len(pos) == 1 and not isinstance(spec, (list, tuple, dict)):
            data = np.take(data, 0, axis=ax)  # scalar selection drops the axis
            dims.pop(ax)
            del coords[dim]
        elif isinstance(spec, dict) and "nearest" in spec:
            data = np.take(data, 0, axis=ax)
            dims.pop(ax)
            del coords[dim]
    for dim, fn in (reduce or {}).items():
        if dim not in dims:
            raise EngineError(f"Unknown dim '{dim}' for reduce. Remaining dims: {dims}")
        if fn not in _REDUCERS:
            raise EngineError(f"Unknown reducer '{fn}'. Choose from {list(_REDUCERS)}")
        ax = dims.index(dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            data = _REDUCERS[fn](data.astype(float), axis=ax)
        dims.pop(ax)
        del coords[dim]
        notes.append(f"{fn} over {dim}")
    return Tensor(np.asarray(data), dims, coords), notes


def _even_positions(n: int, k: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, num=min(n, k)).round().astype(int))


def tensor_to_frame(t: Tensor, max_cells: int | None = None) -> tuple[pd.DataFrame | Any, str]:
    """Render a 0/1/2-D tensor as a DataFrame (down-sampled evenly if over the cell budget)."""
    max_cells = min(max_cells or MAX_CELLS, MAX_CELLS)
    note = ""
    if t.data.ndim == 0:
        return t.data.item(), note
    if t.data.ndim == 1:
        d = t.dims[0]
        c, v = t.coords[d], t.data
        if len(v) > max_cells // 2:
            pos = _even_positions(len(v), max_cells // 2)
            c, v = c[pos], v[pos]
            note = f"showing {len(pos)} evenly spaced of {t.shape[0]} values; refine select to see exact ranges"
        return pd.DataFrame({d: c, "value": v}), note
    if t.data.ndim == 2:
        r, cdim = t.dims
        rows, cols = t.coords[r], t.coords[cdim]
        data = t.data
        if data.size > max_cells:
            nr = max(2, int(math.sqrt(max_cells * data.shape[0] / data.shape[1])))
            nc = max(2, max_cells // nr)
            pr, pc = _even_positions(data.shape[0], nr), _even_positions(data.shape[1], nc)
            data = data[np.ix_(pr, pc)]
            rows, cols = rows[pr], cols[pc]
            note = f"showing {len(pr)}x{len(pc)} evenly spaced of {t.shape[0]}x{t.shape[1]} cells; refine select to see exact ranges"
        df = pd.DataFrame(data, index=pd.Index([_fmt_num(x) for x in rows], name=r), columns=[_fmt_num(x) for x in cols])
        df.columns.name = cdim
        return df, note
    raise EngineError(f"tensor still has {t.data.ndim} dims {t.dims}; select or reduce until 1-2 remain, or store with new_name.")


def slice_tensor(name: str, select: dict[str, Any] | None = None, reduce: dict[str, str] | None = None,
                 new_name: str | None = None, max_cells: int | None = None) -> str:
    e = REGISTRY.tensor(name)
    sub, notes = slice_tensor_obj(e.obj, select, reduce)
    L = [f"slice of '{name}': " + ("; ".join(notes) if notes else "no selection") + f" -> shape {'x'.join(map(str, sub.shape)) or 'scalar'} dims {sub.dims}"]
    if new_name:
        if sub.data.ndim <= 2 and sub.data.ndim > 0:
            if sub.data.ndim == 2:
                df = pd.DataFrame(sub.data, index=pd.Index(sub.coords[sub.dims[0]], name=sub.dims[0]),
                                  columns=[str(_fmt_num(c)) for c in sub.coords[sub.dims[1]]])
            else:
                df = pd.DataFrame({sub.dims[0]: sub.coords[sub.dims[0]], "value": sub.data})
            REGISTRY.put(Entry(new_name, "table", df, {"parent": name, "op": "slice", "params": {"select": select, "reduce": reduce}}))
            L.append(f"stored as table '{new_name}' ({df.shape[0]}x{df.shape[1]}); it can be described/projected/plotted like any table")
        else:
            REGISTRY.put(Entry(new_name, "tensor", sub, {"parent": name, "op": "slice", "params": {"select": select, "reduce": reduce}}))
            L.append(f"stored as tensor '{new_name}'")
    if sub.data.ndim <= 2:
        obj, note = tensor_to_frame(sub, max_cells)
        if isinstance(obj, pd.DataFrame):
            L.append(frame_to_text(obj, max_rows=MAX_ROWS, max_cols=MAX_COLS, index=sub.data.ndim == 2))
        else:
            L.append(f"value = {_fmt_num(obj)}")
        if note:
            L.append(f"[{note}]")
    else:
        flat = sub.data.astype(float).ravel()
        fin = flat[np.isfinite(flat)]
        L.append(f"still {sub.data.ndim}-D; stats: fill={np.isfinite(flat).mean():.1%} min={_fmt_num(fin.min()) if len(fin) else 'nan'} "
                 f"mean={_fmt_num(fin.mean()) if len(fin) else 'nan'} max={_fmt_num(fin.max()) if len(fin) else 'nan'}. "
                 f"Select or reduce more dims ({sub.dims}) to read values, or pass new_name to store it.")
    return clip("\n".join(L))


TENSOR_OPS = ("log1p", "log", "sqrt", "abs", "demean", "zscore", "diff", "pct_change", "log_return", "cumsum",
              "anomaly", "area_weight", "normalize", "fillna")


def tensor_transform(name: str, op: str, dim: str | None = None, new_name: str | None = None,
                     params: dict[str, Any] | None = None) -> str:
    """Preprocess a tensor and store the result as a new tensor.

    Elementwise: log1p, log (params offset), sqrt, abs, fillna (params value, or method='mean' along dim).
    Along `dim`: demean, zscore, diff / pct_change / log_return (params periods; the axis shrinks), cumsum,
    normalize (divide by the sum along dim, params target).
    anomaly: `dim` must be a datetime axis; subtracts the per-period mean along it (params period=month|dayofyear|
    dayofweek|hour|quarter, baseline=[start, end] to compute the climatology on a reference window) -> anomalies.
    area_weight: multiply by cos(latitude)^power (params power, default 0.5 for EOF/PCA, 1 for means) along a numeric
    latitude axis in degrees.
    """
    op = op.lower().strip()
    if op not in TENSOR_OPS:
        raise EngineError(f"Unknown op '{op}'. Choose from {TENSOR_OPS}")
    params = dict(params or {})
    e = REGISTRY.tensor(name)
    t: Tensor = e.obj
    data = t.data.astype(float)
    dims = list(t.dims)
    coords = {d: np.asarray(c) for d, c in t.coords.items()}
    note = ""
    needs_dim = op in ("demean", "zscore", "diff", "pct_change", "log_return", "cumsum", "anomaly", "area_weight", "normalize")
    if needs_dim and not dim:
        raise EngineError(f"op '{op}' needs dim=<axis name>; dims are {dims}")
    ax = t.axis(dim) if dim else None
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if op == "log1p":
            out = np.log1p(np.where(data < -1, np.nan, data))
        elif op == "log":
            off = float(params.get("offset", 0.0))
            out = np.log(np.where(data + off > 0, data + off, np.nan))
        elif op == "sqrt":
            out = np.sqrt(np.where(data < 0, np.nan, data))
        elif op == "abs":
            out = np.abs(data)
        elif op == "fillna":
            if "value" in params:
                out = np.where(np.isfinite(data), data, float(params["value"]))
            else:
                if ax is None:
                    raise EngineError("fillna needs params={'value': v} or dim=<axis> for a mean fill along that axis")
                m = np.nanmean(data, axis=ax, keepdims=True)
                out = np.where(np.isfinite(data), data, m)
            note = f"{int((~np.isfinite(data)).sum()):,} missing cells filled"
        elif op == "demean":
            out = data - np.nanmean(data, axis=ax, keepdims=True)
        elif op == "zscore":
            sd = np.nanstd(data, axis=ax, keepdims=True)
            out = (data - np.nanmean(data, axis=ax, keepdims=True)) / np.where(sd > 0, sd, np.nan)
        elif op in ("diff", "pct_change", "log_return"):
            per = int(params.get("periods", 1))
            if per < 1 or per >= data.shape[ax]:
                raise EngineError(f"periods must be between 1 and {data.shape[ax] - 1}")
            a = np.take(data, np.arange(per, data.shape[ax]), axis=ax)
            b = np.take(data, np.arange(0, data.shape[ax] - per), axis=ax)
            out = a - b if op == "diff" else (a / b - 1 if op == "pct_change" else np.log(a / b))
            coords[dim] = coords[dim][per:]
            note = f"axis '{dim}' shrank by {per}"
        elif op == "cumsum":
            out = np.nancumsum(data, axis=ax)
            out = np.where(np.isfinite(data), out, np.nan)
        elif op == "normalize":
            target = float(params.get("target", 1.0))
            s = np.nansum(data, axis=ax, keepdims=True)
            out = data / np.where(s != 0, s, np.nan) * target
        elif op == "anomaly":
            c = coords[dim]
            if c.dtype.kind != "M":
                raise EngineError(f"anomaly needs a datetime axis; '{dim}' is {c.dtype}")
            period = str(params.get("period", "month")).lower()
            if period not in _PERIODS:
                raise EngineError(f"period must be one of {list(_PERIODS)}")
            idx = pd.DatetimeIndex(c)
            key = np.asarray(_PERIODS[period](idx))
            base_mask = np.ones(len(c), dtype=bool)
            if params.get("baseline"):
                lo, hi = params["baseline"]
                base_mask = (idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi))
                if not base_mask.any():
                    raise EngineError("baseline window matches no time steps")
            out = np.empty_like(data)
            moved = np.moveaxis(data, ax, 0)
            res = np.empty_like(moved)
            for kval in np.unique(key):
                sel = key == kval
                clim = np.nanmean(moved[sel & base_mask], axis=0)
                res[sel] = moved[sel] - clim
            out = np.moveaxis(res, 0, ax)
            note = f"subtracted the per-{period} climatology along '{dim}'" + (f" (baseline {params['baseline']})" if params.get("baseline") else "")
        elif op == "area_weight":
            c = coords[dim]
            if c.dtype.kind not in "iuf":
                raise EngineError(f"area_weight needs a numeric latitude axis in degrees; '{dim}' is {c.dtype}")
            power = float(params.get("power", 0.5))
            w = np.cos(np.deg2rad(c.astype(float))).clip(min=0) ** power
            shape = [1] * data.ndim
            shape[ax] = len(c)
            out = data * w.reshape(shape)
            note = f"multiplied by cos(lat)^{power:g} along '{dim}'"
        else:  # pragma: no cover
            raise EngineError(f"op {op} not implemented")
    nt = Tensor(np.asarray(out), dims, coords)
    nm = new_name or REGISTRY.unique(f"{name}__{op}")
    REGISTRY.put(Entry(nm, "tensor", nt, {"parent": name, "op": op, "params": {"dim": dim, **params}, "source": e.meta.get("source")}))
    fin = nt.data[np.isfinite(nt.data)]
    L = [f"stored tensor '{nm}': shape {'x'.join(map(str, nt.shape))}, dims {dims} = {op} of '{name}'" + (f"; {note}" if note else "")]
    L.append(f"values: fill={np.isfinite(nt.data).mean():.1%} min={_fmt_num(fin.min()) if len(fin) else 'nan'} mean={_fmt_num(fin.mean()) if len(fin) else 'nan'} "
             f"max={_fmt_num(fin.max()) if len(fin) else 'nan'} std={_fmt_num(fin.std()) if len(fin) else 'nan'}")
    L.append(f"next: describe('{nm}'), slice_tensor('{nm}', ...), plot('{nm}', kind='tensor'|'surface'|'lines', ...)")
    return clip("\n".join(L))


def tensor_to_table(name: str, row_dim: str, new_name: str | None = None, select: dict[str, Any] | None = None,
                    reduce: dict[str, str] | None = None) -> str:
    """Flatten a tensor into a table with one row per coordinate of `row_dim` and one column per combination of the other
    dims (e.g. time x lat x lon -> time rows x gridpoint columns, ready for project(pca) = EOF analysis). All-NaN columns
    are dropped; the row coordinate becomes a label column named after row_dim. Optional select/reduce are applied first."""
    e = REGISTRY.tensor(name)
    sub, notes = slice_tensor_obj(e.obj, select, reduce)
    if row_dim not in sub.dims:
        raise EngineError(f"row_dim '{row_dim}' not among dims {sub.dims}")
    ax = sub.dims.index(row_dim)
    moved = np.moveaxis(sub.data.astype(float), ax, 0)
    other = [d for d in sub.dims if d != row_dim]
    flat = moved.reshape(moved.shape[0], -1)
    if other:
        grids = np.meshgrid(*[sub.coords[d] for d in other], indexing="ij")
        names = ["_".join(f"{d}{_fmt_num(g.ravel()[i])}" for d, g in zip(other, grids)) for i in range(flat.shape[1])]
    else:
        names = ["value"]
    keep = ~np.all(~np.isfinite(flat), axis=0)
    df = pd.DataFrame(flat[:, keep], columns=[re.sub(r"[^0-9A-Za-z_.+-]+", "_", n) for n, k in zip(names, keep) if k])
    if df.shape[1] * df.shape[0] * 8 > MAX_TENSOR_BYTES:
        raise EngineError("resulting table would exceed HIDIM_MAX_TENSOR_BYTES; select/reduce first")
    df.insert(0, row_dim, sub.coords[row_dim])
    nm = new_name or REGISTRY.unique(f"{name}__{row_dim}_table")
    REGISTRY.put(Entry(nm, "table", df, {"parent": name, "op": "tensor_to_table", "params": {"row_dim": row_dim, "select": select, "reduce": reduce},
                                        "derived_cols": [], "source": e.meta.get("source")}))
    return clip(f"stored table '{nm}': {df.shape[0]:,} rows ({row_dim}) x {df.shape[1] - 1:,} feature columns ({' x '.join(other) or 'value'}; "
                f"{int((~keep).sum()):,} all-NaN columns dropped)" + (f"; {'; '.join(notes)}" if notes else "")
                + f"\nnext: describe('{nm}') or project('{nm}', method='pca', standardize=False) for EOF-style analysis")


# --------------------------------------------------------------------------- #
# Synthetic data (for demos and tests)
# --------------------------------------------------------------------------- #


def make_synthetic(kind: str = "blobs", n: int = 2000, d: int = 20, k: int = 4, noise: float = 0.05,
                   seed: int = 0, name: str | None = None) -> str:
    rng = np.random.default_rng(seed)
    kind = kind.lower()
    if kind == "blobs":
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n, n_features=d, centers=k, cluster_std=1.0 + 2.0 * noise, random_state=seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
        df["label"] = [f"c{v}" for v in y]
        meta = {"source": f"synthetic blobs n={n} d={d} k={k}", "derived_cols": []}
    elif kind in ("swiss_roll", "s_curve"):
        from sklearn.datasets import make_swiss_roll, make_s_curve
        X3, t = (make_swiss_roll if kind == "swiss_roll" else make_s_curve)(n_samples=n, noise=noise, random_state=seed)
        Q, _ = np.linalg.qr(rng.standard_normal((max(d, 3), max(d, 3))))
        X = np.zeros((n, max(d, 3)))
        X[:, :3] = X3
        X = X @ Q.T + noise * rng.standard_normal(X.shape)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["t"] = t
        meta = {"source": f"synthetic {kind} embedded in {X.shape[1]}-D (intrinsic dim 2)", "derived_cols": ["t"]}
    elif kind == "gaussian":
        df = pd.DataFrame(rng.standard_normal((n, d)), columns=[f"f{i}" for i in range(d)])
        meta = {"source": f"synthetic iid Gaussian n={n} d={d} (null model: no structure)", "derived_cols": []}
    elif kind == "lowrank":
        r = max(1, min(k, d))
        A = rng.standard_normal((n, r))
        B = rng.standard_normal((r, d))
        df = pd.DataFrame(A @ B + noise * rng.standard_normal((n, d)), columns=[f"f{i}" for i in range(d)])
        meta = {"source": f"synthetic rank-{r} matrix plus noise {noise}", "derived_cols": []}
    elif kind == "options_surface":
        dates = pd.date_range("2024-01-01", periods=max(2, n // 100), freq="B")
        strikes = np.round(np.linspace(60, 140, 17), 1)
        mats = np.array([7, 14, 30, 60, 90, 180, 365])
        recs = []
        for i, dte in enumerate(dates):
            spot = 100 * np.exp(0.01 * np.cumsum(rng.standard_normal(1))[0] + 0.002 * i)
            base = 0.20 + 0.03 * np.sin(i / 7)
            for T in mats:
                for K in strikes:
                    m = np.log(K / spot) / math.sqrt(T / 365)
                    iv = base + 0.08 * m ** 2 - 0.05 * m + 0.02 * math.sqrt(T / 365) + noise * 0.2 * rng.standard_normal()
                    if rng.random() < 0.03:  # missing quotes
                        continue
                    recs.append((dte, float(K), int(T), float(max(iv, 0.03)), float(spot), int(rng.integers(1, 5000))))
        df = pd.DataFrame(recs, columns=["date", "strike", "maturity_days", "iv", "spot", "volume"])
        meta = {"source": "synthetic implied-volatility surface in long format (date, strike, maturity_days -> iv)", "derived_cols": []}
    else:
        raise EngineError("kind must be one of: blobs, swiss_roll, s_curve, gaussian, lowrank, options_surface")
    nm = name or REGISTRY.unique(f"synthetic_{kind}")
    REGISTRY.put(Entry(nm, "table", df, meta))
    return clip(f"stored '{nm}': {df.shape[0]:,} rows x {df.shape[1]} cols ({meta['source']})\ncolumns: {list(df.columns)[:30]}\nnext: describe('{nm}')")


# --------------------------------------------------------------------------- #
# Plotting (matplotlib, object-oriented API, Agg backend, no pyplot state)
# --------------------------------------------------------------------------- #

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
           "#17becf", "#8c564b", "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
           "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h"]
OTHER = "#898781"
MAX_CLASSES = _env_int("HIDIM_MAX_CLASSES", 20)      # legend entries before folding into 'Other'
MAX_ANNOTATIONS = _env_int("HIDIM_MAX_ANNOTATIONS", 300)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
DIV_STEPS = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#7d1c1c"]


def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    return (LinearSegmentedColormap.from_list("hidim_seq", SEQ_STEPS),
            LinearSegmentedColormap.from_list("hidim_div", DIV_STEPS))


def _new_fig(w: float = 8, h: float = 6, dpi: int = 110):
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), dpi=dpi, facecolor=SURFACE)
    return fig


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


_PLOT_COUNTER = 0
MAX_PNG_BYTES = _env_int("HIDIM_MAX_PNG_BYTES", 700_000)


def _render(fig, stem: str) -> tuple[bytes, Path]:
    """Rasterize to PNG, shrinking the dpi if the file is too large for comfortable transport."""
    global _PLOT_COUNTER
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    dpi = fig.get_dpi()
    for _ in range(3):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=SURFACE, bbox_inches="tight", dpi=dpi)
        data = buf.getvalue()
        if len(data) <= MAX_PNG_BYTES or dpi <= 60:
            break
        dpi = max(60, int(dpi * 0.8))
    _PLOT_COUNTER += 1
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir() / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)}_{stamp}_{_PLOT_COUNTER:03d}.png"
    path.write_bytes(data)
    return data, path


def _class_groups(values: pd.Series, max_classes: int | None = None) -> tuple[list[str], np.ndarray, str]:
    """Order classes by frequency; fold the tail beyond max_classes (default HIDIM_MAX_CLASSES=20) into 'Other'.
    Returns (classes, codes, note)."""
    max_classes = max_classes or MAX_CLASSES
    s = values.astype(str).fillna("nan")
    counts = s.value_counts()
    classes = list(counts.index[: max_classes - 1]) if len(counts) > max_classes else list(counts.index)
    note = ""
    if len(counts) > max_classes:
        note = f"{len(counts) - len(classes)} rarer classes folded into 'Other' (raise HIDIM_MAX_CLASSES to show more)"
        classes = classes + ["Other"]
        lookup = {c: i for i, c in enumerate(classes)}
        codes = np.array([lookup.get(v, len(classes) - 1) for v in s])
    else:
        lookup = {c: i for i, c in enumerate(classes)}
        codes = np.array([lookup[v] for v in s])
    return classes, codes, note


def _is_continuous_color(cvals: pd.Series) -> bool:
    """Datetimes and numeric columns with many distinct values get a colour bar; everything else a categorical legend."""
    if pd.api.types.is_datetime64_any_dtype(cvals):
        return True
    return bool(pd.api.types.is_numeric_dtype(cvals) and not pd.api.types.is_bool_dtype(cvals) and cvals.nunique() > 12)


def _color_values(cvals: pd.Series) -> tuple[np.ndarray, Any]:
    """Numeric array for a colour bar plus a tick formatter (dates get date labels)."""
    if pd.api.types.is_datetime64_any_dtype(cvals):
        from matplotlib.ticker import FuncFormatter
        raw = pd.to_datetime(cvals).to_numpy()
        nat = pd.isna(raw)
        ns = raw.astype("datetime64[ns]").astype("int64").astype(float)   # explicit ns: pandas 3 stores microseconds by default
        ns = np.where(nat, np.nan, ns)
        span_days = (np.nanmax(ns) - np.nanmin(ns)) / 86400e9 if np.isfinite(ns).any() else 0
        fmt = "%Y" if span_days > 365 * 6 else "%Y-%m" if span_days > 120 else "%Y-%m-%d" if span_days > 2 else "%m-%d %H:%M"
        return ns, FuncFormatter(lambda v, _pos: pd.Timestamp(int(v)).strftime(fmt) if np.isfinite(v) else "")
    return cvals.to_numpy(dtype=float), None


def _annotate(ax, x, y, labels: pd.Series, note_parts: list[str], z=None) -> None:
    """Write text labels next to points (capped at MAX_ANNOTATIONS, the first ones in row order)."""
    n = len(labels)
    m = min(n, MAX_ANNOTATIONS)
    xs, ys = np.asarray(x), np.asarray(y)
    txt = labels.astype(str).to_numpy()
    for i in range(m):
        if z is None:
            ax.annotate(txt[i], (xs[i], ys[i]), fontsize=6, color=INK2, xytext=(2, 2), textcoords="offset points")
        else:
            ax.text(xs[i], ys[i], np.asarray(z)[i], txt[i], fontsize=6, color=INK2)
    if m < n:
        note_parts.append(f"first {m} of {n} points annotated (HIDIM_MAX_ANNOTATIONS)")


def _resolve_color(e: Entry, color_by: str | None) -> pd.Series | None:
    if not color_by:
        return None
    df: pd.DataFrame = e.obj
    if color_by in df.columns:
        return df[color_by]
    parent = e.meta.get("parent")
    rows = e.meta.get("rows")
    if parent and rows is not None and parent in REGISTRY.names():
        p = REGISTRY.get(parent)
        if p.kind == "table" and color_by in p.obj.columns:
            return pd.Series(p.obj[color_by].to_numpy()[np.asarray(rows)], index=df.index, name=color_by)
    raise EngineError(f"color_by column '{color_by}' not in '{e.name}'" + (f" or its parent '{parent}'" if parent else "") + f". Columns: {list(df.columns)[:30]}")


def _scatter(ax, x, y, cvals: pd.Series | None, fig, z=None, size: float = 10.0, title: str = "", log_color: bool = False) -> str:
    seq, _ = _cmaps()
    note = ""
    kw = dict(s=size, linewidths=0, alpha=0.8)
    if cvals is None:
        if z is None:
            ax.scatter(x, y, color=PALETTE[0], **kw)
        else:
            ax.scatter(x, y, z, color=PALETTE[0], **kw)
    elif _is_continuous_color(cvals):
        c, formatter = _color_values(cvals)
        norm = None
        if log_color and np.isfinite(c).any() and np.nanmin(c) > 0:
            from matplotlib.colors import LogNorm
            norm = LogNorm()
        sc = ax.scatter(x, y, c=c, cmap=seq, norm=norm, **kw) if z is None else ax.scatter(x, y, z, c=c, cmap=seq, norm=norm, **kw)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(str(cvals.name), color=INK2, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(AXIS)
        if formatter is not None:
            cb.ax.yaxis.set_major_formatter(formatter)
            note = "colour = date (continuous)"
    else:
        classes, codes, note = _class_groups(cvals)
        many = len(classes) > 10
        for i, cl in enumerate(classes):
            m = codes == i
            col = OTHER if cl == "Other" else PALETTE[i % len(PALETTE)]
            mk = MARKERS[i % len(MARKERS)]
            if z is None:
                ax.scatter(np.asarray(x)[m], np.asarray(y)[m], color=col, marker=mk, label=f"{cl} ({m.sum()})", **kw)
            else:
                ax.scatter(np.asarray(x)[m], np.asarray(y)[m], np.asarray(z)[m], color=col, marker=mk, label=f"{cl} ({m.sum()})", **kw)
        if len(classes) >= 2:
            fs = 7 if many else 8
            if z is None:
                leg = ax.legend(title=str(cvals.name), fontsize=fs, title_fontsize=8, frameon=False, markerscale=1.6,
                                loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, ncol=2 if len(classes) > 14 else 1)
            else:
                leg = ax.legend(title=str(cvals.name), fontsize=fs, title_fontsize=8, frameon=False, markerscale=1.6, loc="upper left")
            for t in leg.get_texts():
                t.set_color(INK2)
            leg.get_title().set_color(INK)
    return note


def _axis_positions(coord: np.ndarray, log: bool = False) -> tuple[np.ndarray, bool]:
    """Real-valued positions for a tensor axis (numeric values as they are, or log10 of them; datetimes as days);
    label axes get 0..n-1."""
    if coord.dtype.kind in "iuf":
        v = coord.astype(float)
        if log:
            if not np.all(v > 0):
                raise EngineError("log axis needs strictly positive coordinates")
            v = np.log10(v)
        return (v, True) if np.all(np.isfinite(v)) else (np.arange(len(coord), dtype=float), False)
    if coord.dtype.kind == "M":
        ns = coord.astype("datetime64[ns]").astype("int64").astype(float)
        return (ns - ns.min()) / 86400e9, True
    return np.arange(len(coord), dtype=float), False


def _cluster_order(C: np.ndarray) -> np.ndarray:
    """Hierarchical (average-linkage) ordering of a correlation matrix so blocks of related features sit together."""
    m = C.shape[0]
    if m < 3:
        return np.arange(m)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
        from scipy.spatial.distance import squareform
        D = 1 - np.nan_to_num(np.abs(C), nan=0.0)
        np.fill_diagonal(D, 0.0)
        D = np.clip((D + D.T) / 2, 0, None)
        Z = linkage(squareform(D, checks=False), method="average")
        try:
            Z = optimal_leaf_ordering(Z, squareform(D, checks=False))
        except Exception:  # noqa: BLE001
            pass
        return np.asarray(leaves_list(Z))
    except Exception:  # noqa: BLE001
        return np.arange(m)


def plot(name: str, kind: str = "scatter", x: str | None = None, y: str | None = None, z: str | None = None,
         color_by: str | None = None, features=None, select: dict[str, Any] | None = None,
         reduce: dict[str, str] | None = None, series_dim: str | None = None, sample: int | None = None,
         seed: int = 0, title: str | None = None, width: float | None = None, height: float | None = None,
         log_color: bool = False, annotate: str | None = None, order: str = "cluster",
         log_axes: Sequence[str] | str | None = None) -> tuple[bytes, str]:
    """Render a PNG. Returns (png_bytes, caption)."""
    log_axes = [log_axes] if isinstance(log_axes, str) else list(log_axes or [])
    kind = kind.lower()
    e = REGISTRY.get(name)
    seq, div = _cmaps()
    cap_parts: list[str] = []

    if e.kind == "tensor":
        t: Tensor = e.obj
        sub, notes = slice_tensor_obj(t, select, reduce)
        if kind in ("tensor", "heatmap", "surface"):
            if sub.data.ndim != 2:
                raise EngineError(f"'{kind}' needs a 2-D slice; after selection the tensor is {sub.data.ndim}-D {sub.dims}. Add select/reduce.")
            rows, cols = sub.coords[sub.dims[0]], sub.coords[sub.dims[1]]
            data = sub.data.astype(float)
            fin = data[np.isfinite(data)]
            if kind == "surface":
                import mpl_toolkits.mplot3d  # noqa: F401
                fig = _new_fig(width or 9, height or 7)
                ax = fig.add_subplot(projection="3d")
                # numeric / datetime axes are drawn at their real coordinate values; label axes at 0..n-1
                xlog, ylog = sub.dims[1] in log_axes, sub.dims[0] in log_axes
                xpos, x_real = _axis_positions(cols, log=xlog)
                ypos, y_real = _axis_positions(rows, log=ylog)
                X, Y = np.meshgrid(xpos, ypos)
                surf = ax.plot_surface(X, Y, data, cmap=seq, linewidth=0.2, edgecolor=GRID, antialiased=True)
                ax.set_facecolor(SURFACE)
                ax.set_xlabel(sub.dims[1] + (" (days)" if cols.dtype.kind == "M" else " (log10)" if xlog else ""), color=INK2, fontsize=9)
                ax.set_ylabel(sub.dims[0] + (" (days)" if rows.dtype.kind == "M" else " (log10)" if ylog else ""), color=INK2, fontsize=9)
                if not x_real:
                    xt = _even_positions(len(cols), 6)
                    ax.set_xticks(xt)
                    ax.set_xticklabels([_fmt_num(c) for c in cols[xt]], fontsize=7, color=MUTED)
                if not y_real:
                    yt = _even_positions(len(rows), 6)
                    ax.set_yticks(yt)
                    ax.set_yticklabels([_fmt_num(r) for r in rows[yt]], fontsize=7, color=MUTED)
                ax.tick_params(axis="x", labelsize=7, colors=MUTED)
                ax.tick_params(axis="y", labelsize=7, colors=MUTED)
                ax.tick_params(axis="z", labelsize=7, colors=MUTED)
                ax.set_title(title or f"{name}: {sub.dims[0]} x {sub.dims[1]}", color=INK, fontsize=10, loc="left")
                fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.08)
                if x_real or y_real:
                    cap_parts.append("axes at true coordinate values" + ("" if (x_real and y_real) else " (label axis evenly spaced)"))
            else:
                fig = _new_fig(width or 8, height or 6)
                ax = fig.add_subplot()
                diverging = len(fin) and fin.min() < 0 < fin.max() and min(abs(fin.min()), fin.max()) / max(abs(fin.min()), fin.max()) > 0.2
                if diverging:
                    v = float(np.nanmax(np.abs(fin)))
                    im = ax.imshow(data, aspect="auto", cmap=div, vmin=-v, vmax=v, interpolation="nearest")
                else:
                    from matplotlib.colors import LogNorm
                    norm = LogNorm() if (log_color and len(fin) and fin.min() > 0) else None
                    im = ax.imshow(data, aspect="auto", cmap=seq, norm=norm, interpolation="nearest")
                xt = _even_positions(len(cols), 16)
                yt = _even_positions(len(rows), 24)
                ax.set_xticks(xt)
                ax.set_xticklabels([_fmt_num(c) for c in cols[xt]], rotation=45, ha="right")
                ax.set_yticks(yt)
                ax.set_yticklabels([_fmt_num(r) for r in rows[yt]])
                _style(ax, title or f"{name}: {sub.dims[0]} x {sub.dims[1]}", sub.dims[1], sub.dims[0])
                ax.grid(False)
                cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
                cb.ax.tick_params(colors=MUTED, labelsize=8)
                cb.outline.set_edgecolor(AXIS)
            cap_parts.append(f"{kind} of '{name}' slice {'; '.join(notes) or '(full)'}: {len(rows)} {sub.dims[0]} x {len(cols)} {sub.dims[1]}; "
                             f"value range [{_fmt_num(fin.min()) if len(fin) else 'nan'}, {_fmt_num(fin.max()) if len(fin) else 'nan'}], fill {np.isfinite(data).mean():.0%}")
        elif kind == "lines":
            if sub.data.ndim not in (1, 2):
                raise EngineError(f"'lines' needs a 1-D or 2-D slice; got {sub.data.ndim}-D {sub.dims}.")
            fig = _new_fig(width or 9, height or 5)
            ax = fig.add_subplot()
            if sub.data.ndim == 1:
                xs = sub.coords[sub.dims[0]]
                ax.plot(xs, sub.data, color=PALETTE[0], linewidth=1.6)
                _style(ax, title or f"{name}", sub.dims[0], "value")
                cap_parts.append(f"line of '{name}' along {sub.dims[0]} ({len(xs)} points), {'; '.join(notes)}")
            else:
                sd = series_dim or (sub.dims[0] if sub.shape[0] <= sub.shape[1] else sub.dims[1])
                sax = sub.dims.index(sd)
                xd = sub.dims[1 - sax]
                xs = sub.coords[xd]
                series = sub.coords[sd]
                data = np.moveaxis(sub.data, sax, 0)
                shown = min(len(series), 12)
                pos = _even_positions(len(series), shown) if len(series) > 12 else np.arange(len(series))
                for i, p in enumerate(pos):
                    ax.plot(xs, data[p], color=PALETTE[i % len(PALETTE)], linewidth=1.6, label=f"{sd}={_fmt_num(series[p])}")
                leg = ax.legend(fontsize=8, frameon=False)
                for tx in leg.get_texts():
                    tx.set_color(INK2)
                _style(ax, title or f"{name}: value vs {xd} by {sd}", xd, "value")
                cap_parts.append(f"{len(pos)} of {len(series)} series ({sd}) over {xd} ({len(xs)} points), {'; '.join(notes)}"
                                 + ("; evenly spaced subset of series shown (use select isin to pick)" if len(series) > 12 else ""))
        else:
            raise EngineError("For tensors use kind='tensor' (heatmap of a 2-D slice), 'surface' (3-D), or 'lines'.")
        png, path = _render(fig, f"{name}_{kind}")
        cap_parts.append(f"saved {path}")
        return png, clip(" | ".join(cap_parts))

    # ---- tables ----
    df: pd.DataFrame = e.obj
    n = len(df)
    sub_idx = _sample_rows(n, sample or PLOT_MAX_POINTS, seed)
    dfp = df.iloc[sub_idx] if sub_idx is not None else df
    ep = Entry(e.name, "table", dfp, dict(e.meta, rows=(np.asarray(e.meta["rows"])[sub_idx] if (sub_idx is not None and e.meta.get("rows") is not None) else e.meta.get("rows")))) if sub_idx is not None else e
    if sub_idx is not None:
        cap_parts.append(f"random sample of {len(dfp):,} of {n:,} rows")
    cvals = _resolve_color(ep, color_by)

    if kind == "scatter":
        num_cols = [c for c in dfp.columns if pd.api.types.is_numeric_dtype(dfp[c]) and c not in e.derived_cols()]
        if x is None or y is None:
            if len(num_cols) < 2:
                raise EngineError("scatter needs two numeric columns; pass x and y.")
            x, y = num_cols[0], num_cols[1]
            if z is None and e.meta.get("method") and len(num_cols) >= 3 and num_cols[2] == "c3":
                z = "c3"
        for c in (x, y) + ((z,) if z else ()):
            if c not in dfp.columns:
                raise EngineError(f"column '{c}' not found. Columns: {list(dfp.columns)[:30]}")
        size = float(np.clip(1600 / math.sqrt(max(len(dfp), 1)), 3, 24))
        labels = _resolve_color(ep, annotate) if annotate else None
        if z:
            import mpl_toolkits.mplot3d  # noqa: F401
            fig = _new_fig(width or 9, height or 7)
            ax = fig.add_subplot(projection="3d")
            note = _scatter(ax, dfp[x], dfp[y], cvals, fig, z=dfp[z], size=size, log_color=log_color)
            if labels is not None:
                _annotate(ax, dfp[x], dfp[y], labels, cap_parts, z=dfp[z])
            ax.set_facecolor(SURFACE)
            ax.set_xlabel(x, color=INK2); ax.set_ylabel(y, color=INK2); ax.set_zlabel(z, color=INK2)
            ax.tick_params(colors=MUTED, labelsize=7)
            ax.set_title(title or f"{name}: {x}, {y}, {z}", color=INK, fontsize=10, loc="left")
        else:
            fig = _new_fig(width or 8, height or 6.5)
            ax = fig.add_subplot()
            note = _scatter(ax, dfp[x], dfp[y], cvals, fig, size=size, log_color=log_color)
            if labels is not None:
                _annotate(ax, dfp[x], dfp[y], labels, cap_parts)
            _style(ax, title or f"{name}: {y} vs {x}", x, y)
        if note:
            cap_parts.append(note)
        cap_parts.insert(0, f"scatter of '{name}' {x} vs {y}" + (f" vs {z}" if z else "") + (f", coloured by {color_by}" if color_by else "")
                         + (f", annotated with {annotate}" if annotate else "") + f" ({len(dfp):,} points)")
    elif kind == "pairs":
        cols = resolve_columns(dfp, features, e, numeric_only=True) if features else resolve_columns(dfp, "top_variance:5", e)
        if len(cols) > 6:
            raise EngineError("pairs supports at most 6 features (36 panels); pass features=[...].")
        m = len(cols)
        fig = _new_fig(width or 2.2 * m + 1, height or 2.2 * m + 1)
        axes = fig.subplots(m, m)
        size = float(np.clip(600 / math.sqrt(max(len(dfp), 1)), 2, 10))
        note = ""
        for i in range(m):
            for j in range(m):
                ax = axes[i][j] if m > 1 else axes
                if i == j:
                    ax.hist(dfp[cols[i]].dropna(), bins=30, color=PALETTE[0], alpha=0.85, linewidth=0)
                else:
                    note = _scatter(ax, dfp[cols[j]], dfp[cols[i]], cvals if (i == 0 and j == 1) else (cvals if cvals is not None else None), fig, size=size) or note
                    if (cvals is not None) and not (i == 0 and j == 1) and ax.get_legend():
                        ax.get_legend().remove()
                _style(ax, "", cols[j] if i == m - 1 else "", cols[i] if j == 0 else "")
                ax.tick_params(labelsize=6)
        fig.suptitle(title or f"{name}: pairwise scatter of {m} features", color=INK, fontsize=10, x=0.02, ha="left")
        cap_parts.insert(0, f"pairs plot of {cols}" + (f" coloured by {color_by}" if color_by else ""))
        if note:
            cap_parts.append(note)
    elif kind == "parallel":
        num = _numeric_frame(ep, features)
        if num.shape[1] > 30:
            num = num[resolve_columns(num, "top_variance:30", e)]
            cap_parts.append("top 30 features by variance shown")
        lines_cap = 1500
        if len(num) > lines_cap:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(num), lines_cap, replace=False))
            num = num.iloc[keep]
            cvals = cvals.iloc[keep] if cvals is not None else None
            cap_parts.append(f"{lines_cap} lines sampled")
        Z = num.to_numpy(dtype=float)
        lo, hi = np.nanmin(Z, axis=0), np.nanmax(Z, axis=0)
        Zn = (Z - lo) / np.where(hi - lo == 0, 1, hi - lo)
        m = Zn.shape[1]
        fig = _new_fig(width or max(8, 0.45 * m + 3), height or 5)
        ax = fig.add_subplot()
        xs = np.arange(m)
        note = ""
        if cvals is not None and not (pd.api.types.is_numeric_dtype(cvals) and cvals.nunique() > 12):
            classes, codes, note = _class_groups(cvals)
            for i, cl in enumerate(classes):
                sel = codes == i
                col = OTHER if cl == "Other" else PALETTE[i % len(PALETTE)]
                for row in Zn[sel]:
                    ax.plot(xs, row, color=col, alpha=0.25, linewidth=0.7)
                ax.plot([], [], color=col, linewidth=2, label=f"{cl} ({sel.sum()})")
            leg = ax.legend(title=str(cvals.name), fontsize=8, title_fontsize=8, frameon=False, loc="upper left",
                            bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
            for tx in leg.get_texts():
                tx.set_color(INK2)
        else:
            for row in Zn:
                ax.plot(xs, row, color=PALETTE[0], alpha=0.12, linewidth=0.6)
        for xx in xs:
            ax.axvline(xx, color=AXIS, linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(list(num.columns), rotation=60, ha="right", fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["min", "max"])
        _style(ax, title or f"{name}: parallel coordinates ({Zn.shape[0]} rows, {m} features, min-max scaled)")
        ax.grid(False)
        cap_parts.insert(0, f"parallel coordinates of '{name}'" + (f" coloured by {color_by}" if color_by else ""))
        if note:
            cap_parts.append(note)
    elif kind in ("heatmap", "tensor", "matrix"):
        num = _numeric_frame(ep, features)
        if num.shape[1] > 200:
            num = num[resolve_columns(num, "top_variance:200", e)]
            cap_parts.append("top 200 features by variance")
        if len(num) > 400:
            pos = _even_positions(len(num), 400)
            num = num.iloc[pos]
            cap_parts.append(f"{len(pos)} evenly spaced rows of {len(dfp):,}")
        M = num.to_numpy(dtype=float)
        fin = M[np.isfinite(M)]
        fig = _new_fig(width or min(12, 5 + 0.05 * M.shape[1]), height or min(10, 4 + 0.012 * M.shape[0]))
        ax = fig.add_subplot()
        diverging = len(fin) and fin.min() < 0 < fin.max() and min(abs(fin.min()), fin.max()) / max(abs(fin.min()), fin.max()) > 0.2
        if diverging:
            v = float(np.nanmax(np.abs(fin)))
            im = ax.imshow(M, aspect="auto", cmap=div, vmin=-v, vmax=v, interpolation="nearest")
        else:
            from matplotlib.colors import LogNorm
            norm = LogNorm() if (log_color and len(fin) and fin.min() > 0) else None
            im = ax.imshow(M, aspect="auto", cmap=seq, norm=norm, interpolation="nearest")
        xt = _even_positions(M.shape[1], 24)
        ax.set_xticks(xt)
        ax.set_xticklabels([str(num.columns[i]) for i in xt], rotation=60, ha="right", fontsize=7)
        yt = _even_positions(M.shape[0], 24)
        ax.set_yticks(yt)
        ax.set_yticklabels([_fmt_num(num.index[i]) for i in yt], fontsize=7)
        _style(ax, title or f"{name}: {M.shape[0]} rows x {M.shape[1]} columns", "", str(num.index.name or ""))
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(AXIS)
        cap_parts.insert(0, f"value heatmap of '{name}' ({M.shape[0]} rows x {M.shape[1]} cols), range [{_fmt_num(fin.min()) if len(fin) else 'nan'}, {_fmt_num(fin.max()) if len(fin) else 'nan'}]"
                         + (", diverging scale centred at 0" if diverging else ""))
    elif kind == "corr":
        num = _numeric_frame(ep, features)
        if num.shape[1] > 120:
            num = num[resolve_columns(num, "top_variance:120", e)]
            cap_parts.append("top 120 features by variance")
        C = num.corr().to_numpy()
        m = C.shape[0]
        names = [str(c) for c in num.columns]
        if order == "cluster" and m >= 3:
            perm = _cluster_order(C)
            C = C[np.ix_(perm, perm)]
            names = [names[i] for i in perm]
            cap_parts.append("features reordered by hierarchical clustering so correlated blocks sit together (order='given' keeps table order)")
        fig = _new_fig(width or min(13, 4 + 0.22 * m), height or min(12, 3.5 + 0.22 * m))
        ax = fig.add_subplot()
        im = ax.imshow(C, cmap=div, vmin=-1, vmax=1, interpolation="nearest")
        fs = 7 if m <= 40 else 6 if m <= 70 else 5
        ax.set_xticks(range(m)); ax.set_xticklabels(names, rotation=90, fontsize=fs)
        ax.set_yticks(range(m)); ax.set_yticklabels(names, fontsize=fs)
        if m <= 12:
            for i in range(m):
                for j in range(m):
                    ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK if abs(C[i, j]) < 0.6 else "#ffffff")
        _style(ax, title or f"{name}: correlation matrix ({m} features)")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        iu = np.triu_indices(m, 1)
        cap_parts.insert(0, f"correlation heatmap of {m} features; mean |r|={np.nanmean(np.abs(C[iu])):.3f}, max |r|={np.nanmax(np.abs(C[iu])) if m > 1 else 0:.3f}"
                         + (f"; order: {names[:12]}{' ...' if m > 12 else ''}" if order == "cluster" and m >= 3 else ""))
    elif kind == "scree":
        from sklearn.decomposition import PCA
        X, cols, notes = _matrix(ep, features, standardize="auto")
        scaled = "standardized" if notes.get("standardized") else "raw-scale"
        kmax = min(50, X.shape[1], X.shape[0] - 1)
        p = PCA(n_components=kmax, random_state=seed).fit(X)
        ev = p.explained_variance_ratio_
        cum = np.cumsum(ev)
        fig = _new_fig(width or 10, height or 4.2)
        ax1, ax2 = fig.subplots(1, 2)
        ax1.bar(np.arange(1, kmax + 1), ev, color=PALETTE[0], width=0.7, linewidth=0)
        _style(ax1, "explained variance ratio per component", "component", "ratio")
        ax2.plot(np.arange(1, kmax + 1), cum, color=PALETTE[0], linewidth=1.8, marker="o", markersize=3)
        for thr in (0.9, 0.95):
            ax2.axhline(thr, color=MUTED, linewidth=0.8, linestyle="--")
            ax2.text(kmax, thr, f" {int(thr * 100)}%", color=MUTED, fontsize=7, va="bottom", ha="right")
        ax2.set_ylim(0, 1.02)
        _style(ax2, "cumulative explained variance", "components", "cumulative ratio")
        fig.suptitle(title or f"{name}: PCA scree ({X.shape[1]} {scaled} features, {X.shape[0]:,} rows)", color=INK, fontsize=10, x=0.02, ha="left")
        k90 = int(np.searchsorted(cum, 0.9) + 1) if cum[-1] >= 0.9 else None
        cap_parts.insert(0, f"scree ({scaled}): first 5 ratios {np.round(ev[:5], 3).tolist()}; components for 90% = {k90 if k90 else f'>{kmax}'}")
    elif kind == "hist":
        num = _numeric_frame(ep, features)
        if num.shape[1] > 16:
            num = num[resolve_columns(num, "top_variance:16", e)]
            cap_parts.append("top 16 features by variance")
        m = num.shape[1]
        ncol = min(4, m)
        nrow = math.ceil(m / ncol)
        fig = _new_fig(width or 2.6 * ncol + 1, height or 2.0 * nrow + 1)
        axes = np.atleast_1d(fig.subplots(nrow, ncol)).ravel()
        for i, c in enumerate(num.columns):
            axes[i].hist(num[c].dropna(), bins=40, color=PALETTE[0], linewidth=0, alpha=0.9)
            _style(axes[i], str(c))
            axes[i].tick_params(labelsize=6)
        for j in range(m, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(title or f"{name}: distributions of {m} features", color=INK, fontsize=10, x=0.02, ha="left")
        cap_parts.insert(0, f"histograms of {list(num.columns)}")
    elif kind == "andrews":
        from pandas.plotting import andrews_curves
        if cvals is None:
            raise EngineError("andrews needs color_by=<class column>.")
        num = _numeric_frame(ep, features)
        if num.shape[1] > 20:
            num = num[resolve_columns(num, "top_variance:20", e)]
        Z = (num - num.mean()) / num.std().replace(0, 1)
        classes, codes, note = _class_groups(cvals)
        Z = Z.copy()
        Z["__class__"] = [classes[c] for c in codes]
        if len(Z) > 600:
            Z = Z.sample(600, random_state=seed)
            cap_parts.append("600 curves sampled")
        fig = _new_fig(width or 9, height or 5)
        ax = fig.add_subplot()
        andrews_curves(Z, "__class__", ax=ax, color=[OTHER if c == "Other" else PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(Z["__class__"].unique(), key=classes.index))], alpha=0.35, linewidth=0.7)
        leg = ax.legend(fontsize=8, frameon=False, title=str(cvals.name))
        for tx in leg.get_texts():
            tx.set_color(INK2)
        _style(ax, title or f"{name}: Andrews curves ({num.shape[1]} standardized features)", "t", "f(t)")
        cap_parts.insert(0, f"Andrews curves by {color_by}")
        if note:
            cap_parts.append(note)
    else:
        raise EngineError("kind must be one of: scatter, pairs, parallel, corr, heatmap, scree, hist, andrews (tables); tensor, surface, lines (tensors).")

    png, path = _render(fig, f"{name}_{kind}")
    cap_parts.append(f"saved {path}")
    return png, clip(" | ".join(cap_parts))
