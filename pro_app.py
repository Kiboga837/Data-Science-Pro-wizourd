import matplotlib

matplotlib.use("Agg")  # headless-safe backend (must be set before pyplot is imported)

from io import BytesIO  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import cleaning  # noqa: E402,F401
from auto import render_auto  # noqa: E402
from common import SAMPLES, normalize_columns  # noqa: E402
from cleaning import render_cleaning  # noqa: E402
from eda import render_eda, render_trends  # noqa: E402
from inspection import render_inspection  # noqa: E402
import project  # noqa: E402
from ml import render_ml  # noqa: E402
from qa import render_qa  # noqa: E402
from report import render_report  # noqa: E402
from stats_analysis import render_statistics  # noqa: E402
from bi_page import render_dashboard  # noqa: E402  (new: dashboards)
from export_page import render_export  # noqa: E402  (new: PDF / Word export)
from sql_chat import render_sql_chat  # noqa: E402  (new: SQL chat)

st.set_page_config(page_title="Smart Data Science Studio Pro", layout="wide")
st.title("Smart Data Science Studio Pro")
st.caption("Inspect, clean, analyse, query with SQL, build dashboards, model and report: all in one place.")


# Streamlit forgets the value of any widget that is not drawn in a run (e.g. when you change page).
# Re-assigning these keys at the top of every run keeps cleaning decisions, notes, report settings, etc.
PERSIST_PREFIXES = ("strat_", "const_", "note_", "report_", "api_", "q_text_main", "auto_", "nav_task", "data_source",
                    "sqlw_", "biw", "expw_", "hubw_")  # last row: widget keys of the SQL, dashboard and export pages
for _k in list(st.session_state.keys()):
    if isinstance(_k, str) and _k.startswith(PERSIST_PREFIXES):
        st.session_state[_k] = st.session_state[_k]


@st.cache_data(show_spinner=False)
def load_project_cached(blob: bytes) -> dict:
    return project.load_project(blob)


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str, sheet=None) -> pd.DataFrame:
    """Read a CSV or an Excel sheet from raw bytes."""
    buffer = BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        try:
            return normalize_columns(pd.read_csv(buffer))
        except UnicodeDecodeError:
            buffer.seek(0)
            return normalize_columns(pd.read_csv(buffer, encoding="latin-1"))
    return normalize_columns(pd.read_excel(buffer, sheet_name=sheet or 0))


@st.cache_data(show_spinner=False)
def excel_sheets(file_bytes: bytes) -> list:
    return pd.ExcelFile(BytesIO(file_bytes)).sheet_names


# ----------------------------------------------------------------------------
# Sidebar: upload + task selection
# ----------------------------------------------------------------------------
st.sidebar.header("1. Upload Data")
file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
proj_file = st.sidebar.file_uploader("Or load a saved project (.zip)", type=["zip"], key="proj_file")
use_sample = st.sidebar.checkbox("Use a built-in sample dataset (deliberately messy)", value=False)
sample_name = st.sidebar.selectbox("Sample", list(SAMPLES), label_visibility="collapsed") if use_sample else None

df_raw, sheet, proj, dataset_name = None, None, None, "sample_data"
if file:
    try:
        if file.name.lower().endswith(".xlsx"):
            sheets = excel_sheets(file.getvalue())
            sheet = sheets[0] if len(sheets) == 1 else st.sidebar.selectbox("Excel sheet", sheets)
        df_raw = load_data(file.getvalue(), file.name, sheet)
        dataset_name = file.name
        st.sidebar.success("Data Loaded Successfully")
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
elif proj_file:
    try:
        proj = load_project_cached(proj_file.getvalue())
        df_raw = proj["raw"]
        dataset_name = proj["manifest"].get("dataset_name", proj_file.name)
        st.sidebar.success(f"Project loaded (saved {proj['manifest'].get('saved_at', '')})")
    except Exception as e:
        st.sidebar.error(f"Could not load project: {e}")
elif use_sample:
    df_raw = SAMPLES[sample_name]()
    dataset_name = sample_name
    st.sidebar.info("Using synthetic sample data")

if df_raw is None:
    st.info("Upload a CSV or Excel file in the sidebar (or tick the sample data box) to get started.")
    st.stop()

# A new dataset invalidates everything computed for the previous one.
if file:
    data_id = f"{file.name}:{file.size}:{sheet}"
elif proj is not None:
    data_id = f"project:{proj_file.name}:{proj_file.size}"
else:
    data_id = f"sample:{sample_name}"
if st.session_state.get("data_id") != data_id:
    prefixes = ("strat_", "const_", "df_clean", "clean_", "ml_", "qa_results", "q_text_main", "q_loaded_id",
                "report_", "note_", "project_", "auto_run",
                "hub_", "sql_", "bi_", "exp_", "sqlw_", "biw", "expw_", "hubw_")  # + SQL / dashboard / export state
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(prefixes):
            del st.session_state[k]
    st.session_state["data_id"] = data_id
    if proj is not None:
        project.restore_state(proj, st.session_state)

TASKS = ["Full Analysis (one click)", "Data Inspection", "Data Cleaning", "Exploratory Analysis",
         "Statistical Analysis", "Time Trends", "Questions & Answers", "Predictive Model", "SQL Chat", "Dashboards",
         "Report", "Report Export (PDF / Word)"]
# Pages can request a jump (e.g. "Open the report"); it must be applied before the radio is drawn.
if "_goto" in st.session_state:
    st.session_state["nav_task"] = st.session_state.pop("_goto")
if "_data_source" in st.session_state:
    st.session_state["data_source"] = st.session_state.pop("_data_source")
if st.session_state.get("nav_task") not in TASKS:
    st.session_state["nav_task"] = TASKS[0]

st.sidebar.header("2. Select Task")
task = st.sidebar.radio("Choose an option", TASKS, key="nav_task")
st.sidebar.caption("New here? Run **Full Analysis**. Or work step by step. **SQL Chat** and **Dashboards** work on any loaded table; **Report Export** gives PDF and Word.")

df_clean = st.session_state.get("df_clean")
if df_clean is not None:
    st.sidebar.header("3. Dataset to Analyse")
    source = st.sidebar.radio("Use", ["Cleaned data", "Original data"], key="data_source", label_visibility="collapsed")
    df = df_clean if source == "Cleaned data" else df_raw
    label = source
else:
    df, label = df_raw, "Original data"

# ----------------------------------------------------------------------------
# Dataset preview (always shown)
# ----------------------------------------------------------------------------
if task in ("SQL Chat", "Dashboards", "Report Export (PDF / Word)"):
    pass  # these pages show their own data views
elif task == "Data Cleaning":
    st.subheader("Dataset Preview (original)")
    st.dataframe(df_raw.head())
else:
    st.subheader("Dataset Preview")
    st.caption(f"Showing: {label}")
    st.dataframe(df.head())

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
if task == "Full Analysis (one click)":
    render_auto(df_raw, dataset_name)
elif task == "Data Inspection":
    render_inspection(df, label)
elif task == "Data Cleaning":
    render_cleaning(df_raw)
elif task == "Exploratory Analysis":
    render_eda(df)
elif task == "Statistical Analysis":
    render_statistics(df)
elif task == "Time Trends":
    render_trends(df)
elif task == "Questions & Answers":
    render_qa(df, label)
elif task == "Predictive Model":
    render_ml(df, label)
elif task == "SQL Chat":
    render_sql_chat(df_raw, df_clean, dataset_name)
elif task == "Dashboards":
    render_dashboard(df_raw, df_clean, dataset_name)
elif task == "Report":
    render_report(df_raw, df, label, dataset_name)
elif task == "Report Export (PDF / Word)":
    render_export(df_raw, df, label, dataset_name)
