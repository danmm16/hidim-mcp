"""Drive the server over real stdio JSON-RPC (no client SDK needed) and exercise every tool.

Usage: python tests/stdio_smoke.py [--keep-images]
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TMP = Path(os.environ.get("HIDIM_TEST_DIR", HERE / "_tmp"))
TMP.mkdir(exist_ok=True)

env = dict(os.environ, HIDIM_ROOTS=str(ROOT.parent), HIDIM_OUT=str(TMP / "out"), PYTHONPATH=str(ROOT))
proc = subprocess.Popen([sys.executable, "-m", "hidim_mcp.server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=open(TMP / "server.stderr.log", "wb"), env=env, cwd=str(ROOT))
_id = 0


def send(method: str, params: dict | None = None, notify: bool = False) -> dict | None:
    global _id
    msg: dict = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if not notify:
        _id += 1
        msg["id"] = _id
    proc.stdin.write((json.dumps(msg) + "\n").encode())
    proc.stdin.flush()
    if notify:
        return None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("server closed stdout; see " + str(TMP / "server.stderr.log"))
        resp = json.loads(line)
        if resp.get("id") == _id:
            return resp


def call(tool: str, **arguments) -> tuple[str, list[bytes]]:
    t0 = time.time()
    resp = send("tools/call", {"name": tool, "arguments": arguments})
    if "error" in resp:
        raise RuntimeError(f"{tool}: JSON-RPC error {resp['error']}")
    res = resp["result"]
    texts, images = [], []
    for c in res.get("content", []):
        if c["type"] == "text":
            texts.append(c["text"])
        elif c["type"] == "image":
            images.append(base64.b64decode(c["data"]))
    text = "\n".join(texts)
    flag = "ERR " if res.get("isError") else "ok  "
    print(f"{flag}{tool:16s} {time.time() - t0:5.1f}s  {len(text):6d} chars  {len(images)} img  | {text.splitlines()[0][:110] if text else ''}")
    if res.get("isError"):
        print("      ", text[:300])
    return text, images


init = send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "smoke", "version": "0"}})
print("server:", init["result"]["serverInfo"], "| protocol", init["result"]["protocolVersion"])
send("notifications/initialized", notify=True)
tools = send("tools/list")["result"]["tools"]
print("tools:", [t["name"] for t in tools])
sizes = {t["name"]: len(json.dumps(t)) for t in tools}
print("schema bytes total:", sum(sizes.values()), "| largest:", max(sizes.items(), key=lambda kv: kv[1]))

csv = TMP / "wide.csv"
if not csv.exists():
    import numpy as np, pandas as pd
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    cols = {f"{tk}_{f}": rng.standard_normal(120).cumsum() for tk in ("AAPL", "MSFT", "NVDA") for f in ("close", "volume", "ret")}
    pd.DataFrame(cols, index=idx).rename_axis("date").to_csv(csv)

call("list_datasets")
call("make_synthetic", kind="blobs", n=3000, d=64, k=5, noise=0.2, name="blobs")
call("describe", name="blobs", max_features=8)
call("project", name="blobs", method="pca", n_components=3)
call("project", name="blobs", method="tsne", n_components=2, sample=1200, ascii=True)
call("cluster", name="blobs__tsne2", method="kmeans")
call("cluster", name="blobs", method="hdbscan")
call("neighbors", name="blobs", index=5, k=4, columns=["f0", "label"])
call("subset", name="blobs", where="label == 'c2'", top_variance=3, preview=4, new_name="c2")
_, imgs = call("plot", name="blobs__tsne2", kind="scatter", color_by="cluster_kmeans")
assert imgs and imgs[0][:8] == b"\x89PNG\r\n\x1a\n", "scatter did not return a PNG"
call("plot", name="blobs", kind="scree")
call("plot", name="blobs", kind="parallel", color_by="label", features="top_variance:10")
call("plot", name="blobs", kind="corr", features="top_variance:15")
call("make_synthetic", kind="options_surface", n=6000, noise=0.1, name="opts")
call("tensorize", name="opts", axes=["date", "strike", "maturity_days"], values=["iv"], new_name="iv")
call("describe", name="iv")
call("slice_tensor", name="iv", select={"date": "2024-01-10", "strike": {"range": [90, 110]}})
call("slice_tensor", name="iv", select={"strike": {"nearest": 100}}, reduce={"date": "mean"})
call("slice_tensor", name="iv", select={"maturity_days": 30}, new_name="iv_30d")
call("plot", name="iv", kind="surface", select={"date": "2024-01-10"})
call("plot", name="iv", kind="lines", select={"strike": 100}, series_dim="maturity_days")
call("plot", name="iv_30d", kind="tensor")
call("save_dataset", name="iv", path=str(TMP / "iv.npz"))
call("load_dataset", path=str(TMP / "iv.npz"), name="iv_reloaded")
call("load_dataset", path=str(csv), name="wide", options={"index_col": 0, "parse_dates": True})
call("tensor_from_wide", name="wide", sep="_", dims=["ticker", "feature"], index_dim="date", new_name="px")
call("slice_tensor", name="px", select={"feature": "close", "date": {"iloc": [0, 5]}})
call("make_synthetic", kind="gaussian", n=500, d=10, name="g1", seed=1)
call("make_synthetic", kind="gaussian", n=500, d=10, name="g2", seed=2)
call("stack_datasets", names=["g1", "g2"], axis_name="run", new_name="runs")
call("slice_tensor", name="runs", reduce={"row": "std"})
call("save_dataset", name="c2", path=str(TMP / "c2.csv"))
# error paths must come back as tool errors, not crashes
call("describe", name="nope")
call("load_dataset", path="/etc/passwd")
call("slice_tensor", name="iv", select={"bogus": 1})
call("list_datasets")
call("drop_dataset", name="g1")
proc.stdin.close()
proc.wait(timeout=10)
print("server exit code:", proc.returncode)
if not any(a == "--keep-images" for a in sys.argv):
    pass
