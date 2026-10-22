"""Shared helpers used by every page of Smart Data Science Studio."""
import inspect
import re

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------------
# Column-type helpers (work for both object and the newer pandas "str" dtype)
# ----------------------------------------------------------------------------
def is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s.dtype)


def numeric_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def datetime_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


def categorical_cols(df: pd.DataFrame, max_unique: int = None) -> list:
    cols = [
        c
        for c in df.columns
        if is_text(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(df[c])
    ]
    if max_unique:
        cols = [c for c in cols if df[c].nunique() <= max_unique]
    return cols


_ID_NAME = re.compile(r"(^|[_\s\-])(id|uuid|guid|idx|index|key|no|num|number|code)$|^unnamed:?\s*0$", re.I)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names safe for every page: text, and unique (duplicates get _2, _3...)."""
    df = df.copy()
    seen, names = {}, []
    for col in map(str, df.columns):
        seen[col] = seen.get(col, 0) + 1
        names.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
    df.columns = names
    return df


def id_like_columns(df: pd.DataFrame) -> list:
    """Columns that identify rows (IDs, row numbers, names) rather than describe them.
    They carry no analytical signal, so they are left out of default analyses and models."""
    out = []
    if len(df) < 2:
        return out
    for col in df.columns:
        s = df[col]
        filled = int(s.notna().sum())
        nun = s.nunique(dropna=True)
        if filled == 0 or nun <= 1:
            continue
        if _ID_NAME.search(str(col).strip()) and nun >= 0.9 * filled:
            out.append(col)
        elif pd.api.types.is_integer_dtype(s) and nun == len(df) and (s.is_monotonic_increasing or s.is_monotonic_decreasing):
            out.append(col)
        elif is_text(s) and filled > 20 and nun > 0.95 * filled:
            as_num = pd.to_numeric(s.dropna().astype(str).str.replace(",", "", regex=False), errors="coerce")
            if as_num.notna().mean() < 0.9:  # numbers stored as text are measurements, not identifiers
                out.append(col)
    return out


def analysis_numeric_cols(df: pd.DataFrame) -> list:
    """Numeric columns that are real measurements (ID-like columns removed)."""
    ids = set(id_like_columns(df))
    return [c for c in numeric_cols(df) if c not in ids]


# ----------------------------------------------------------------------------
# "Show me the code" panel: prints the real source of the functions that ran
# ----------------------------------------------------------------------------
def show_method(title: str, funcs=(), libs: dict = None):
    with st.expander(f"Python code and libraries used: {title}"):
        if libs:
            st.markdown("**Libraries and why they are needed**")
            st.table(
                pd.DataFrame(
                    {"Library": list(libs.keys()), "Why it is needed": list(libs.values())}
                ).set_index("Library")
            )
        for fn in funcs:
            try:
                target = getattr(getattr(fn, "_info", None), "func", None) or getattr(fn, "__wrapped__", None) or fn
                src = inspect.getsource(target)
            except (OSError, TypeError):
                continue
            st.code(src, language="python")


# ----------------------------------------------------------------------------
# Sample data (deliberately messy so every cleaning feature can be tried)
# ----------------------------------------------------------------------------
def make_sample_data(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")
    month = dates.month.to_numpy()
    seasonal = 1 + 0.35 * np.isin(month, [11, 12]) + 0.15 * np.isin(month, [3, 9])
    category = rng.choice(["Furniture", "Technology", "Office Supplies"], n)
    region = rng.choice(["East", "West", "Central", "South"], n)
    quantity = rng.integers(1, 10, n)
    discount = rng.choice([0, 0.1, 0.2, 0.3, 0.5], n)
    sales = rng.gamma(2.0, 120, n) * quantity / 3 * seasonal
    profit = sales * (0.25 - discount) + rng.normal(0, 15, n)
    df = pd.DataFrame(
        {
            "Order Date": dates.sort_values(),
            "Category": category,
            "Region": region,
            "Quantity": quantity,
            "Discount": discount,
            "Sales": sales.round(2),
            "Profit": profit.round(2),
        }
    )
    # realistic mess: missing values, placeholders, stray spaces, outliers, duplicates
    for col, frac in [("Sales", 0.03), ("Profit", 0.02), ("Discount", 0.06), ("Region", 0.04)]:
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "Category"] = "N/A"
    pad = rng.choice(n, 25, replace=False)
    df.loc[pad, "Region"] = df.loc[pad, "Region"].map(lambda v: f" {v} " if isinstance(v, str) else v)
    df.loc[rng.choice(n, 4, replace=False), "Sales"] = float(df["Sales"].max()) * 8
    df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)
    return df


def make_churn_data(n: int = 1200) -> pd.DataFrame:
    """Customer churn sample: a classification problem with a text-typed numeric column."""
    rng = np.random.default_rng(7)
    tenure = rng.integers(1, 72, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    monthly = np.round(rng.normal(65, 25, n).clip(18, 120), 2)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
    support = rng.choice(["Yes", "No"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    logit = (-0.5 + 0.9 * (contract == "Month-to-month") - 0.03 * tenure + 0.012 * (monthly - 65)
             + 0.6 * (internet == "Fiber optic") - 0.5 * (support == "Yes") + 0.3 * senior)
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "senior_citizen": senior,
        "tenure_months": tenure.astype(float),
        "contract": contract,
        "internet_service": internet,
        "tech_support": support,
        "payment_method": rng.choice(["Credit card", "Bank transfer", "Electronic check", "Mailed check"], n),
        "monthly_charges": monthly,
        "total_charges": np.round(tenure * monthly, 2).astype(str),
        "churn": churn,
    })
    df.loc[rng.choice(n, 11, replace=False), "total_charges"] = " "
    for col, frac in [("monthly_charges", 0.03), ("tenure_months", 0.02), ("internet_service", 0.04)]:
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    return df


def make_student_data(n: int = 600) -> pd.DataFrame:
    """Student performance sample: a regression problem with a date, a boolean and gaps."""
    rng = np.random.default_rng(11)
    hours = rng.gamma(4, 2.0, n).clip(0, 30).round(1)
    attendance = rng.normal(85, 10, n).clip(40, 100).round(1)
    previous = rng.normal(65, 12, n).clip(0, 100).round(1)
    tutoring = rng.choice([True, False], n, p=[0.3, 0.7])
    final = (0.45 * previous + 1.1 * hours + 0.18 * attendance + 4 * tutoring + rng.normal(0, 5, n)).clip(0, 100).round(1)
    df = pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "enrolment_date": pd.to_datetime("2021-09-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "gender": rng.choice(["F", "M"], n),
        "age": rng.integers(17, 26, n),
        "parental_education": rng.choice(["High school", "Bachelor", "Master", "PhD"], n, p=[0.35, 0.4, 0.2, 0.05]),
        "tutoring": tutoring,
        "study_hours": hours,
        "attendance_pct": attendance,
        "previous_score": previous,
        "final_score": final,
    })
    for col, frac in [("study_hours", 0.05), ("attendance_pct", 0.03), ("parental_education", 0.04)]:
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    return df


SAMPLES = {"Sales (messy)": make_sample_data,
           "Customer churn (classification)": make_churn_data,
           "Student performance (regression)": make_student_data}
