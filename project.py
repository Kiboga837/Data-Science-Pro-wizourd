"""Save and restore a whole analysis session.

Format: a ZIP containing manifest.json plus the data as Parquet (CSV fallback).
It deliberately avoids pickle: loading a pickle from an uploaded file can run
arbitrary code, which is unsafe on a hosted app. Trained model objects are
therefore NOT stored; their comparison tables, explanations and importances are.
"""
import base64
import json
import zipfile
from datetime import datetime
from io import BytesIO, StringIO

import pandas as pd
import streamlit as st

VERSION = 1
SAVED_PREFIXES = ("note_", "report_title", "report_author", "report_sections", "report_max",
                  "report_nums", "report_cats", "report_exec")


# ----------------------------------------------------------------------------
# DataFrame <-> bytes
# ----------------------------------------------------------------------------
def _df_to_file(df: pd.DataFrame, stem: str):
    buf = BytesIO()
    try:
        df.to_parquet(buf, index=False)
        return f"{stem}.parquet", buf.getvalue()
    except Exception:
        safe = df.copy()
        for c in safe.columns:
            if safe[c].dtype == object or pd.api.types.is_string_dtype(safe[c].dtype):
                safe[c] = safe[c].map(lambda v: v if pd.isna(v) else str(v))
        try:
            buf = BytesIO()
            safe.to_parquet(buf, index=False)
            return f"{stem}.parquet", buf.getvalue()
        except Exception:
            return f"{stem}.csv", safe.to_csv(index=False).encode("utf-8")


def _dtype_map(df: pd.DataFrame) -> dict:
    return {str(c): str(t) for c, t in df.dtypes.items()}


def _file_to_df(name: str, data: bytes, dtypes: dict = None) -> pd.DataFrame:
    if name.endswith(".parquet"):
        return pd.read_parquet(BytesIO(data))
    df = pd.read_csv(BytesIO(data))
    for col, dt in (dtypes or {}).items():  # CSV has no types, so restore them from the manifest
        if col not in df.columns:
            continue
        try:
            if dt.startswith("datetime"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dt.startswith(("int", "Int")) and df[col].notna().all():
                df[col] = df[col].astype("int64")
            elif dt.startswith("float"):
                df[col] = df[col].astype("float64")
            elif dt == "bool":
                df[col] = df[col].astype(bool)
        except (ValueError, TypeError):
            pass
    return df


def _split_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split", date_format="iso")


def _from_split(s: str) -> pd.DataFrame:
    return pd.read_json(StringIO(s), orient="split")


# ----------------------------------------------------------------------------
# Machine-learning summary (everything except the fitted model objects)
# ----------------------------------------------------------------------------
def ml_summary_live(res: dict, tuned_params=None) -> dict:
    return {"problem": res["problem"], "target": res["target"], "kept": list(res["kept"]),
            "classes": res["classes"], "metrics": res["metrics"].copy(),
            "perm": {k: v.head(15).copy() for k, v in res.get("perm", {}).items()},
            "tuned_params": {k: str(v) for k, v in (tuned_params or {}).items()}}


def _ml_to_json(s: dict) -> dict:
    return {"problem": s["problem"], "target": s["target"], "kept": s["kept"], "classes": s["classes"],
            "metrics": _split_json(s["metrics"]), "perm": {k: _split_json(v) for k, v in s["perm"].items()},
            "tuned_params": s["tuned_params"]}


def _ml_from_json(j: dict) -> dict:
    return {"problem": j["problem"], "target": j["target"], "kept": j["kept"], "classes": j["classes"],
            "metrics": _from_split(j["metrics"]), "perm": {k: _from_split(v) for k, v in j["perm"].items()},
            "tuned_params": j.get("tuned_params", {})}


# ----------------------------------------------------------------------------
# Q&A results
# ----------------------------------------------------------------------------
def _qa_to_json(results: dict) -> list:
    out = []
    for (label, shape, question), r in results.items():
        item = {"label": label, "shape": list(shape), "question": question,
                "answer": r.get("answer", ""), "approach": r.get("approach", []),
                "libraries": [list(x) for x in r.get("libraries", [])], "code": r.get("code", ""),
                "stdout": r.get("stdout", ""), "error": r.get("error"), "attempts": r.get("attempts", 0),
                "images": [base64.b64encode(i).decode() for i in r.get("images", [])],
                "result_kind": None, "result": None}
        res = r.get("result")
        if isinstance(res, pd.DataFrame):
            item["result_kind"], item["result"] = "df", _split_json(res)
        elif res is not None:
            item["result_kind"], item["result"] = "scalar", str(res)
        out.append(item)
    return out


def _qa_from_json(items: list) -> dict:
    results = {}
    for it in items:
        res = None
        if it["result_kind"] == "df":
            res = _from_split(it["result"])
        elif it["result_kind"] == "scalar":
            res = it["result"]
        results[(it["label"], tuple(it["shape"]), it["question"])] = {
            "question": it["question"], "answer": it["answer"], "approach": it["approach"],
            "libraries": [tuple(x) for x in it["libraries"]], "code": it["code"], "stdout": it["stdout"],
            "error": it["error"], "attempts": it["attempts"], "result": res,
            "images": [base64.b64decode(i) for i in it["images"]]}
    return results


# ----------------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------------
def save_project(df_raw: pd.DataFrame, state, dataset_name: str) -> bytes:
    df_clean = state.get("df_clean")
    manifest = {"version": VERSION, "app": "Smart Data Science Studio", "dataset_name": dataset_name,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "clean_log": [list(x) for x in state.get("clean_log", [])],
                "clean_before": state.get("clean_before"),
                "qa": _qa_to_json(state.get("qa_results", {})),
                "settings": {}}
    res = state.get("ml_result")
    if res:
        manifest["ml"] = _ml_to_json(ml_summary_live(res, state.get("ml_tuned_params")))
    elif state.get("ml_saved"):
        manifest["ml"] = _ml_to_json(state["ml_saved"])
    for k in list(state.keys()):
        if isinstance(k, str) and k.startswith(SAVED_PREFIXES):
            v = state[k]
            if isinstance(v, (str, int, float, bool, list)):
                manifest["settings"][k] = v

    out = BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        name, data = _df_to_file(df_raw, "raw")
        z.writestr(name, data)
        manifest["raw_file"], manifest["raw_dtypes"] = name, _dtype_map(df_raw)
        if df_clean is not None:
            name, data = _df_to_file(df_clean, "clean")
            z.writestr(name, data)
            manifest["clean_file"], manifest["clean_dtypes"] = name, _dtype_map(df_clean)
        z.writestr("manifest.json", json.dumps(manifest, default=str, indent=1))
    return out.getvalue()


def load_project(blob: bytes) -> dict:
    with zipfile.ZipFile(BytesIO(blob)) as z:
        names = z.namelist()
        if "manifest.json" not in names:
            raise ValueError("This is not a Smart Data Science Studio project file.")
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("version", 0) > VERSION:
            raise ValueError("This project was saved by a newer version of the app.")
        raw = _file_to_df(manifest["raw_file"], z.read(manifest["raw_file"]), manifest.get("raw_dtypes"))
        clean = None
        if manifest.get("clean_file"):
            clean = _file_to_df(manifest["clean_file"], z.read(manifest["clean_file"]), manifest.get("clean_dtypes"))
    return {"manifest": manifest, "raw": raw, "clean": clean}


def restore_state(project: dict, state):
    """Put a loaded project back into session state (call right after a reset)."""
    m = project["manifest"]
    if project["clean"] is not None:
        state["df_clean"] = project["clean"]
        state["clean_log"] = [tuple(x) for x in m.get("clean_log", [])]
        state["clean_before"] = m.get("clean_before") or {"rows": len(project["raw"]),
                                                          "cols": project["raw"].shape[1],
                                                          "missing": int(project["raw"].isna().sum().sum()),
                                                          "dups": int(project["raw"].duplicated().sum())}
    state["qa_results"] = _qa_from_json(m.get("qa", []))
    if m.get("ml"):
        state["ml_saved"] = _ml_from_json(m["ml"])
    for k, v in m.get("settings", {}).items():
        state[k] = v
    state["project_name"] = m.get("dataset_name", "project")
