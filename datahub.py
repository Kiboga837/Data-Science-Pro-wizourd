"""Shared table registry used by SQL Chat and Dashboards.

Holds the main dataset (original and cleaned), any extra tables the user uploads (CSV, Excel with every sheet,
SQLite databases), sample tables, and saved query results. Everything lives in st.session_state, so a table added
on one page is available on every other page.
"""
import re
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

import sql_engine
from common import SAMPLES, normalize_columns
from eda import detect_date_columns

TABLES, VERSIONS, SOURCES, MAIN_NAMES, MAIN_SIG, SEEN = (
    "hub_tables", "hub_versions", "hub_sources", "hub_main_names", "hub_main_sig", "seen_hub_files")


# ----------------------------------------------------------------------------
# Names and preparation
# ----------------------------------------------------------------------------
def safe_name(name, existing=()) -> str:
    base = re.sub(r"\.[A-Za-z0-9]{1,6}$", "", str(name))
    base = re.sub(r"\W+", "_", base).strip("_").lower() or "table"
    if base[0].isdigit():
        base = "t_" + base
    candidate, i = base, 2
    while candidate in existing:
        candidate, i = f"{base}_{i}", i + 1
    return candidate


def prepare_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df that SQLite can store: text column names, dates parsed, exotic objects turned into text."""
    out = normalize_columns(df)
    for c in detect_date_columns(out):
        if not pd.api.types.is_datetime64_any_dtype(out[c]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[c] = pd.to_datetime(out[c], errors="coerce")
    for c in out.columns:
        s = out[c]
        if getattr(s.dtype, "tz", None) is not None:
            out[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            out[c] = s.astype(object)
        elif s.dtype == object:
            out[c] = s.map(lambda v: v if v is None or isinstance(v, (str, int, float, bool, np.generic)) else str(v))
    return out


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
def tables() -> dict:
    return st.session_state.setdefault(TABLES, {})


def versions() -> dict:
    return st.session_state.setdefault(VERSIONS, {})


def sources() -> dict:
    return st.session_state.setdefault(SOURCES, {})


def add_table(name: str, df: pd.DataFrame, source: str = "upload", overwrite: bool = False) -> str:
    t = tables()
    final = safe_name(name) if (overwrite and safe_name(name) in t) else safe_name(name, t)
    t[final] = prepare_for_sql(df)
    versions()[final] = versions().get(final, 0) + 1
    sources()[final] = source
    return final


def remove_table(name: str):
    tables().pop(name, None)
    versions().pop(name, None)
    sources().pop(name, None)
    seen = st.session_state.get(SEEN, {})
    for sig in [s for s, names in seen.items() if name in names]:
        seen.pop(sig, None)  # allow the same file to be added again


def main_tables() -> list:
    return [n for n in st.session_state.get(MAIN_NAMES, []) if n in tables()]


def sync_main(df_raw: pd.DataFrame, df_clean, dataset_name: str):
    """Register the main dataset (and its cleaned version) as tables; refresh them when they change."""
    sig = (dataset_name, df_raw.shape, None if df_clean is None else df_clean.shape)
    if st.session_state.get(MAIN_SIG) == sig and main_tables():
        return
    for n in list(st.session_state.get(MAIN_NAMES, [])):
        remove_table(n)
    others = [n for n in tables() if sources().get(n) != "main"]
    base = safe_name(dataset_name, others)
    names = [add_table(base, df_raw, "main", overwrite=True)]
    if df_clean is not None:
        names.append(add_table(base + "_clean", df_clean, "main", overwrite=True))
    st.session_state[MAIN_NAMES] = names
    st.session_state[MAIN_SIG] = sig


def read_upload(name: str, blob: bytes) -> dict:
    """Turn an uploaded file into {table_name: DataFrame}."""
    low = name.lower()
    if low.endswith((".db", ".sqlite", ".sqlite3")):
        return sql_engine.read_sqlite_file(blob)
    if low.endswith(".csv"):
        try:
            return {name: pd.read_csv(BytesIO(blob))}
        except UnicodeDecodeError:
            return {name: pd.read_csv(BytesIO(blob), encoding="latin-1")}
    if low.endswith(".xlsx"):
        book = pd.ExcelFile(BytesIO(blob))
        if len(book.sheet_names) == 1:
            return {name: book.parse(book.sheet_names[0])}
        return {f"{name}_{sheet}": book.parse(sheet) for sheet in book.sheet_names}
    raise ValueError(f"Unsupported file type: {name}")


def _add_sample(sample_name: str):
    add_table(sample_name, SAMPLES[sample_name](), "sample")


def _remove_selected():
    name = st.session_state.get("hubw_remove")
    if name:
        remove_table(name)
        st.session_state["hubw_remove"] = ""


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def render_table_manager():
    t = tables()
    if t:
        st.dataframe(pd.DataFrame([{"Table": n, "Rows": len(df), "Columns": df.shape[1], "Source": sources().get(n, "")}
                                   for n, df in t.items()]).set_index("Table"))
    else:
        st.info("No tables yet. Load a dataset in the sidebar, or add files below.")

    files = st.file_uploader("Add more tables: CSV, Excel (every sheet becomes a table) or a SQLite database",
                             type=["csv", "xlsx", "db", "sqlite", "sqlite3"], accept_multiple_files=True,
                             key="upl_hub")
    seen = st.session_state.setdefault(SEEN, {})
    for f in files or []:
        sig = (f.name, f.size)
        if sig in seen:
            continue
        try:
            added = [add_table(n, df, "upload") for n, df in read_upload(f.name, f.getvalue()).items()]
            seen[sig] = added
            st.success(f"Added {', '.join(added)}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read {f.name}: {e}")

    c1, c2, c3 = st.columns([3, 1, 3])
    c1.selectbox("Remove a table", [""] + list(t), key="hubw_remove")
    c2.button("Remove", on_click=_remove_selected)
    sample = c3.selectbox("Add a sample table", [""] + list(SAMPLES), key="hubw_sample")
    if sample and sample not in {s for s in t}:
        st.button(f"Add '{sample}'", on_click=_add_sample, args=(sample,))
