"""Unit tests for the engine (no MCP, no network). Run: python -m pytest tests -q

They pin down the behaviours fixed in 0.2.0: scaling auto-detection, Marchenko-Pastur factor counting, consistent hints,
transparent auto-k, coordinate ordering, identifier exclusion, transforms, and the plot fixes (20-class legend, datetime
colour bars, annotations, ordered correlation heatmaps, true-valued surface axes).
"""
from __future__ import annotations

import os
import re
import tempfile

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("HIDIM_OUT", os.path.join(tempfile.gettempdir(), "hidim_test_out"))
os.environ.setdefault("HIDIM_ROOTS", tempfile.gettempdir())

from hidim_mcp import engine as E  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n"


def put(name: str, df: pd.DataFrame, **meta) -> None:
    E.REGISTRY.put(E.Entry(name, "table", df, dict(meta)))


@pytest.fixture(autouse=True)
def _clean_registry():
    for n in list(E.REGISTRY.names()):
        E.REGISTRY.drop(n)
    yield


@pytest.fixture
def digits():
    from sklearn.datasets import load_digits
    d = load_digits()
    df = pd.DataFrame(d.data.astype(np.uint8), columns=[f"p{i:02d}" for i in range(64)])
    df["label"] = [f"d{t}" for t in d.target]
    put("digits", df, derived_cols=[])
    return df


# ----------------------------------------------------------------------------- scaling

def test_scaling_profile_flags_pixels_and_mixed_units(digits):
    X = digits.drop(columns="label").to_numpy(dtype=float)
    prof = E.scaling_profile(X)
    assert prof["recommend_raw"] and prof["kind"] == "counts_or_pixels"
    rng = np.random.default_rng(0)
    mixed = np.column_stack([rng.normal(100, 30, 500), rng.normal(0, 0.01, 500), rng.integers(0, 5, 500)])
    assert not E.scaling_profile(np.column_stack([mixed] * 3))["recommend_raw"]


def test_auto_scaling_keeps_pixels_raw_and_reports_both_counts(digits):
    txt = E.describe("digits")
    assert "geometry (raw scales, centred" in txt
    assert "same counts on standardized scales" in txt
    m = re.search(r"components for 50/80/90/95/99% variance = (\d+)/(\d+)/(\d+)", txt)
    assert m, txt
    # raw PCA on sklearn digits needs ~21 components for 90%; the standardized count is markedly higher
    assert 15 <= int(m.group(3)) <= 25
    m2 = re.search(r"same counts on standardized scales: 50/80/90/95/99% = (\d+)/(\d+)/(\d+)", txt)
    assert int(m2.group(3)) > int(m.group(3))
    assert "random-matrix check" in txt


def test_standardize_true_is_still_available(digits):
    txt = E.describe("digits", standardize=True)
    assert "geometry (standardized" in txt
    out = E.project("digits", method="pca", n_components=2, standardize=True)
    assert "standardized" in out and "dropped_constant_cols" in out


def test_clip_z_and_inflation_warning():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 10))
    X[:, 0] = 0.0
    X[:5, 0] = 1.0                                   # a near-constant feature: z-scores of ~9 for five rows
    X[:, 1] = 0.0
    X[:2, 1] = 1.0                                   # z ~ 14
    put("t", pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]))
    e = E.REGISTRY.get("t")
    _, _, notes = E._matrix(e, standardize=True)
    assert "standardization_inflation" in notes
    Z, _, notes2 = E._matrix(e, standardize=True, clip_z=10)
    assert np.abs(Z).max() <= 10 + 1e-9 and "clipped_cells" in notes2
    with pytest.raises(E.EngineError):
        E._matrix(e, standardize="bogus")


def test_kmeans_warns_about_tiny_clusters_and_lists_silhouette_table():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=600, centers=4, n_features=6, cluster_std=0.6, random_state=0)
    X = np.column_stack([X, np.zeros(600)])
    X[:3, -1] = 50.0                                  # three outliers on an otherwise constant feature
    put("b", pd.DataFrame(X, columns=[f"f{i}" for i in range(7)]))
    out = E.cluster("b", method="kmeans", k=5, standardize=True)
    assert "warning" in out and "< 0.5% of rows" in out and "clip_z=10" in out
    out2 = E.cluster("b", method="kmeans", standardize=True, clip_z=10)
    assert "silhouette by k:" in out2 and "Calinski-Harabasz" in out2 and "clipped_cells" in out2
    put("clean", pd.DataFrame(X[:, :6], columns=[f"f{i}" for i in range(6)]))
    out3 = E.cluster("clean", method="kmeans")
    assert re.search(r"k=4 chosen by silhouette", out3), out3


# ----------------------------------------------------------------------------- spectrum and hints

def test_marchenko_pastur_counts_factors():
    rng = np.random.default_rng(1)
    n, p, r = 2000, 60, 3
    F = rng.normal(size=(n, r)) @ rng.normal(size=(r, p)) * 2.0
    X = F + rng.normal(size=(n, p))
    Xs = (X - X.mean(0)) / X.std(0)
    assert E.marchenko_pastur(Xs)["factors_above_edge"] == r
    G = rng.normal(size=(n, p))
    assert E.marchenko_pastur((G - G.mean(0)) / G.std(0))["factors_above_edge"] <= 1


def test_hints_are_single_and_consistent():
    E.make_synthetic(kind="gaussian", n=1500, d=40, name="g")
    txt = E.describe("g")
    hint = txt.split("hint:")[-1]
    assert "indistinguishable from independent noise" in hint
    assert "random-matrix check: 0 correlation eigenvalue(s)" in txt and "indistinguishable from iid noise" in txt
    assert "use tsne/umap" not in hint
    E.make_synthetic(kind="lowrank", n=2000, d=50, k=2, noise=1.0, name="lr")
    txt2 = E.describe("lr")
    assert "factor + noise structure" in txt2 or "low linear rank" in txt2
    assert not ("nearly lossless" in txt2 and "unfaithful" in txt2)
    E.make_synthetic(kind="blobs", n=120, d=20, k=3, name="small")
    txt3 = E.describe("small")
    assert "only 120 rows" in txt3 or "low linear rank" in txt3 or "too few rows" in txt3
    assert "use tsne/umap" not in txt3


def test_project_warns_on_tiny_nonlinear_runs():
    E.make_synthetic(kind="blobs", n=80, d=10, k=2, name="tiny")
    out = E.project("tiny", method="tsne", n_components=2)
    assert "caution: tsne on 80 rows" in out


# ----------------------------------------------------------------------------- identifiers, labels, loadings

def test_identifier_columns_are_skipped_and_suspects_reported():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"row_id": np.arange(n), "PULocationID": rng.integers(1, 260, n), "hour": rng.integers(0, 24, n),
                       "gt": rng.integers(1, 5, n), "f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)})
    put("idt", df)
    e = E.REGISTRY.get("idt")
    notes: dict = {}
    num = E._numeric_frame(e, None, notes)
    assert set(notes["excluded_identifier_cols"]) == {"row_id", "PULocationID"}
    assert "hour" in notes["integer_cols_that_may_be_labels_or_coordinates"]
    assert "gt" in notes["integer_cols_that_may_be_labels_or_coordinates"]
    assert "hour" in num.columns                     # suspects are reported, not removed
    txt = E.describe("idt")
    assert "skipped as identifiers" in txt and "set_labels" in txt
    E.set_labels("idt", ["hour", "gt"])
    num2 = E._numeric_frame(E.REGISTRY.get("idt"), None, {})
    assert list(num2.columns) == ["f1", "f2", "f3"]
    out = E.set_labels("idt", ["gt"], unset=True)
    assert "features again" in out


def test_loadings_tool_and_capped_output():
    E.make_synthetic(kind="lowrank", n=800, d=30, k=3, noise=0.3, name="lr")
    out = E.project("lr", method="pca", n_components=15, standardize=False)
    assert "c11..c15 omitted" in out and "loadings('lr__pca15'" in out
    txt = E.loadings("lr__pca15", component=1, top=5)
    assert "PC1 of 'lr__pca15'" in txt and "positive:" in txt
    full = E.loadings("lr__pca15", component=2, all_features=True)
    assert full.count("\n") >= 30
    with pytest.raises(E.EngineError):
        E.loadings("lr", component=1)


# ----------------------------------------------------------------------------- tensors

def test_tensorize_natural_sort_and_explicit_order():
    rows = [(t, b, float(i)) for i, (t, b) in enumerate((t, b) for t in ("2024-01", "2024-02") for b in ("b10", "b2", "b1"))]
    put("long", pd.DataFrame(rows, columns=["t", "band", "v"]))
    E.tensorize("long", axes=["t", "band"], values=["v"], new_name="cube")
    assert list(E.REGISTRY.get("cube").obj.coords["band"]) == ["b1", "b2", "b10"]
    E.tensorize("long", axes=["t", "band"], values=["v"], new_name="cube2", order={"band": ["b10", "b1"]})
    assert list(E.REGISTRY.get("cube2").obj.coords["band"]) == ["b10", "b1", "b2"]
    E.tensorize("long", axes=["t", "band"], values=["v"], new_name="cube3", sort=False)
    assert list(E.REGISTRY.get("cube3").obj.coords["band"]) == ["b10", "b2", "b1"]
    with pytest.raises(E.EngineError):
        E.tensorize("long", axes=["t", "band"], values=["v"], order={"nope": ["x"]})


def test_tensor_from_wide_keeps_column_order():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    cols = [f"{v}_{m}" for v in ("lvl", "chg") for m in ("1m", "3m", "6m", "1y", "2y", "10y")]
    df = pd.DataFrame(np.arange(5 * 12, dtype=float).reshape(5, 12), index=idx, columns=cols)
    put("wide", df)
    E.tensor_from_wide("wide", sep="_", dims=["variable", "tenor"], index_dim="date", new_name="tw")
    t = E.REGISTRY.get("tw").obj
    assert list(t.coords["tenor"]) == ["1m", "3m", "6m", "1y", "2y", "10y"]
    assert list(t.coords["variable"]) == ["lvl", "chg"]
    E.tensor_from_wide("wide", sep="_", dims=["variable", "tenor"], index_dim="date", new_name="tw2", sort=True)
    assert list(E.REGISTRY.get("tw2").obj.coords["tenor"]) == ["1m", "1y", "2y", "3m", "6m", "10y"]  # natural, not tenor-aware


def test_tensor_transform_anomaly_and_area_weight_and_to_table():
    time = pd.date_range("2000-01-01", periods=120, freq="MS")
    lat = np.array([30.0, 0.0, -30.0]); lon = np.array([120.0, 180.0, 240.0, 300.0])
    month = time.month.to_numpy()
    season = np.sin(2 * np.pi * month / 12)[:, None, None] * np.array([3, 0, -3])[None, :, None]
    enso = np.sin(np.arange(120) / 9.0)[:, None, None] * np.array([0, 1, 0])[None, :, None] * np.array([0, 0, 1, 1])[None, None, :]
    data = 20 + season + enso
    E.REGISTRY.put(E.Entry("sst", "tensor", E.Tensor(data, ["time", "lat", "lon"], {"time": time.to_numpy(), "lat": lat, "lon": lon}), {}))
    E.tensor_transform("sst", "anomaly", dim="time", new_name="an")
    an = E.REGISTRY.get("an").obj.data
    # the seasonal cycle at lat=30 is gone, the ENSO-like signal at the equator survives
    assert np.abs(an[:, 0, 0]).max() < 1e-6
    assert np.abs(an[:, 1, 2]).max() > 0.5
    E.tensor_transform("an", "area_weight", dim="lat", new_name="anw", params={"power": 1.0})
    w = E.REGISTRY.get("anw").obj.data
    assert np.allclose(w[:, 1, :], an[:, 1, :]) and np.allclose(w[:, 0, :], an[:, 0, :] * np.cos(np.deg2rad(30)))
    out = E.tensor_to_table("anw", row_dim="time", new_name="tbl")
    tbl = E.REGISTRY.get("tbl").obj
    assert tbl.shape == (120, 1 + 12) and "time" in tbl.columns and "rows (time)" in out
    proj = E.project("tbl", method="pca", n_components=2, standardize=False)
    assert "stored embedding" in proj
    with pytest.raises(E.EngineError):
        E.tensor_transform("sst", "anomaly", dim="lat")
    E.tensor_transform("sst", "diff", dim="time", new_name="d1")
    assert E.REGISTRY.get("d1").obj.shape == (119, 3, 4)


# ----------------------------------------------------------------------------- table transforms

def test_transform_log1p_zscore_rownormalize_dropna():
    rng = np.random.default_rng(0)
    counts = rng.poisson(3, size=(200, 6)).astype(float)
    counts[0, 0] = np.nan
    df = pd.DataFrame(counts, columns=[f"g{i}" for i in range(6)])
    df["cell"] = [f"c{i}" for i in range(200)]
    put("cnt", df)
    E.transform("cnt", "row_normalize", new_name="rn", params={"target": 100})
    rn = E.REGISTRY.get("rn").obj
    assert np.allclose(rn.drop(columns="cell").iloc[1:].sum(axis=1), 100)
    assert "cell" in rn.columns
    E.transform("rn", "log1p", new_name="lg")
    assert float(E.REGISTRY.get("lg").obj["g1"].max()) <= np.log1p(100)
    E.transform("lg", "dropna", new_name="dn")
    assert len(E.REGISTRY.get("dn").obj) == 199
    out = E.transform("dn", "zscore", new_name="z", params={"clip": 3})
    z = E.REGISTRY.get("z").obj.drop(columns="cell")
    assert np.abs(z.to_numpy()).max() <= 3 + 1e-9 and "stored 'z'" in out


def test_transform_log_return_diff_by_time_and_group():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": list(idx) * 2, "ticker": ["A"] * 10 + ["B"] * 10,
                       "px": np.r_[100 * 1.01 ** np.arange(10), 50 * 1.02 ** np.arange(10)]})
    put("px", df)
    E.transform("px", "log_return", new_name="r", params={"by": "date", "group_by": "ticker"})
    r = E.REGISTRY.get("r").obj
    assert len(r) == 18
    assert np.allclose(r[r.ticker == "A"].px, np.log(1.01)) and np.allclose(r[r.ticker == "B"].px, np.log(1.02))
    E.transform("px", "diff", new_name="d", params={"by": "date", "group_by": "ticker", "dropna": False})
    assert len(E.REGISTRY.get("d").obj) == 20


def test_transform_deseasonalize_and_residualize():
    idx = pd.date_range("2000-01-01", periods=240, freq="MS")
    rng = np.random.default_rng(0)
    seas = 5 * np.sin(2 * np.pi * idx.month / 12)
    df = pd.DataFrame({"date": idx, "a": seas + rng.normal(0, 0.1, 240), "b": 2 * seas + 1})
    put("s", df)
    E.transform("s", "deseasonalize", new_name="ds", params={"by": "date", "period": "month"})
    ds = E.REGISTRY.get("ds").obj
    assert np.abs(ds["b"]).max() < 1e-9 and ds["a"].std() < 0.2
    mkt = rng.normal(size=500)
    df2 = pd.DataFrame({"mkt": mkt, "x1": 1.5 * mkt + rng.normal(0, 0.1, 500), "x2": -mkt + rng.normal(0, 0.1, 500)})
    put("ret", df2)
    out = E.transform("ret", "residualize", new_name="res", params={"on": ["mkt"]})
    res = E.REGISTRY.get("res").obj
    assert abs(np.corrcoef(res["x1"], res["mkt"])[0, 1]) < 0.05 and "R^2 removed" in out
    with pytest.raises(E.EngineError):
        E.transform("ret", "residualize")
    with pytest.raises(E.EngineError):
        E.transform("ret", "nope")


# ----------------------------------------------------------------------------- plots

def test_scatter_handles_20_classes_datetime_colour_and_annotation(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n), "cls": [f"k{i % 15}" for i in range(n)],
                       "when": pd.date_range("2010-01-01", periods=n, freq="7D"), "name": [f"pt{i}" for i in range(n)]})
    put("sc", df)
    png, cap = E.plot("sc", kind="scatter", x="x", y="y", color_by="cls")
    assert png[:8] == PNG and "folded" not in cap                     # 15 classes all shown
    png, cap = E.plot("sc", kind="scatter", x="x", y="y", color_by="when")
    assert png[:8] == PNG and "colour = date" in cap
    png, cap = E.plot("sc", kind="scatter", x="x", y="y", annotate="name", sample=50)
    assert png[:8] == PNG and "annotated with name" in cap
    df["cls25"] = [f"c{i % 25}" for i in range(n)]
    put("sc25", df)
    _, cap = E.plot("sc25", kind="scatter", x="x", y="y", color_by="cls25")
    assert "folded into 'Other'" in cap


def test_corr_heatmap_is_labelled_and_reordered():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 1))
    df = pd.DataFrame(np.column_stack([rng.normal(size=(300, 1)), base + 0.1 * rng.normal(size=(300, 1)),
                                       rng.normal(size=(300, 1)), base + 0.1 * rng.normal(size=(300, 1))]), columns=["n1", "a1", "n2", "a2"])
    put("cr", df)
    png, cap = E.plot("cr", kind="corr")
    assert png[:8] == PNG and "reordered by hierarchical clustering" in cap
    order = re.search(r"order: \[(.*?)\]", cap).group(1)
    names = [s.strip("' ") for s in order.split(",")]
    assert abs(names.index("a1") - names.index("a2")) == 1
    _, cap2 = E.plot("cr", kind="corr", order="given")
    assert "reordered" not in cap2


def test_surface_uses_true_numeric_axes_and_lines_allow_12_series():
    E.make_synthetic(kind="options_surface", n=4000, name="o")
    E.tensorize("o", axes=["date", "maturity_days", "strike"], values=["iv"], new_name="iv")
    png, cap = E.plot("iv", kind="surface", select={"date": {"iloc": [0, 1]}, "maturity_days": {"range": [7, 365]}}, reduce={"date": "mean"})
    assert png[:8] == PNG and "true coordinate values" in cap
    png, cap = E.plot("iv", kind="lines", select={"strike": 100.0}, series_dim="maturity_days")
    assert png[:8] == PNG
    png, cap = E.plot("iv", kind="surface", select={"date": {"iloc": [0, 1]}}, reduce={"date": "mean"}, log_axes=["maturity_days"])
    assert png[:8] == PNG and "true coordinate values" in cap
    out = E.tensor_transform("iv", "log", new_name="ivlog")
    assert "stored tensor 'ivlog'" in out
    png, _ = E.plot("iv", kind="surface", select={"date": {"iloc": [0, 1]}}, reduce={"date": "mean"}, log_axes=["date"])  # absent dim: ignored
    assert png[:8] == PNG



def test_embedding_carries_datetime_labels_and_describe_flags_duplicates():
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": idx, "a": rng.normal(size=300), "b": rng.normal(size=300)})
    df["a_copy"] = df["a"]
    put("dup", df)
    txt = E.describe("dup")
    assert "identical feature pairs" in txt and "a~a_copy" in txt
    out = E.project("dup", method="pca", n_components=2)
    assert "carried label columns for colouring: ['date']" in out
    emb = E.REGISTRY.get("dup__pca2").obj
    assert "date" in emb.columns


def test_feature_families_and_constants_are_not_flagged_as_labels(digits):
    notes: dict = {}
    E._numeric_frame(E.REGISTRY.get("digits"), None, notes)
    assert "integer_cols_that_may_be_labels_or_coordinates" not in notes
    txt = E._fmt_notes({"dropped_constant_cols": [f"p{i}" for i in range(30)], "standardized": True})
    assert "(+22 more)" in txt and "standardized" not in txt


def test_center_rows_removes_common_component():
    rng = np.random.default_rng(0)
    mkt = rng.normal(size=300)
    df = pd.DataFrame({f"s{i}": mkt + 0.1 * rng.normal(size=300) for i in range(8)})
    put("panel", df)
    E.transform("panel", "center_rows", new_name="pc")
    out = E.REGISTRY.get("pc").obj
    assert np.allclose(out.mean(axis=1), 0) and out["s0"].std() < 0.2
