"""Data cleaning: standardise, decide how to treat missing values, handle outliers.

Every action is logged together with the equivalent pandas code, so the full
cleaning procedure can be downloaded as a reproducible Python script.
"""
import re
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from common import is_text, numeric_cols, show_method

PLACEHOLDER_PATTERN = r"(?i)^\s*(?:|na|n/a|nan|null|none|nil|-|--|\?)\s*$"


# ----------------------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------------------
def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "cat"
    if pd.api.types.is_numeric_dtype(s):
        return "num"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    return "cat"


def strategy_options(s: pd.Series) -> list:
    base = ["Leave as is", "Drop rows with missing", "Drop column"]
    kind = column_kind(s)
    if kind == "num":
        return base + ["Fill with mean", "Fill with median", "Fill with mode", "Fill with constant",
                       "Forward fill", "Backward fill", "Linear interpolation", "KNN imputation"]
    if kind == "date":
        return base + ["Fill with mode", "Forward fill", "Backward fill"]
    return base + ["Fill with mode", "Fill with constant", "Forward fill", "Backward fill"]


def recommend_strategy(s: pd.Series):
    """Return (strategy, reason) using the share of missing data, type and skewness."""
    n, miss = len(s), int(s.isna().sum())
    pct = miss / n * 100 if n else 0.0
    kind = column_kind(s)
    if miss == 0:
        return "Leave as is", "No missing values."
    if pct > 60:
        return "Drop column", f"{pct:.0f}% missing: too sparse to impute reliably."
    if pct < 2:
        return "Drop rows with missing", f"Only {pct:.1f}% missing: removing these rows loses almost nothing."
    if kind == "num":
        skew = s.skew()
        if abs(skew) > 0.5:
            return "Fill with median", f"Skewed distribution (skew {skew:.2f}); the median resists extreme values."
        return "Fill with mean", f"Roughly symmetric distribution (skew {skew:.2f}); the mean is a fair centre."
    if kind == "date":
        return ("Drop rows with missing", "Few missing dates; do not invent dates.") if pct <= 10 else (
            "Forward fill", "Carries the previous date forward (assumes rows are in time order).")
    if pct <= 30:
        return "Fill with mode", "Categorical column: the most frequent category is the safest filler."
    return "Fill with constant", "Many categories missing: label them explicitly as 'Unknown'."


# ----------------------------------------------------------------------------
# Step 1: standardise and fix
# ----------------------------------------------------------------------------
def count_disguised_missing(df: pd.DataFrame) -> pd.Series:
    out = {}
    for c in df.columns:
        if is_text(df[c]):
            s = df[c].dropna().astype(str)
            n = int(s.str.match(PLACEHOLDER_PATTERN).sum())
            if n:
                out[c] = n
    return pd.Series(out, dtype=int)


def _convert_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def preprocess(df: pd.DataFrame, opts: dict):
    """Apply the standardisation options. Returns (new_df, log) where log is a
    list of (description, python_code) tuples."""
    df = df.copy()
    log = []

    if opts.get("strip_colnames"):
        df.columns = df.columns.astype(str).str.strip()
        log.append(("Stripped spaces from column names", "df.columns = df.columns.astype(str).str.strip()"))
    if opts.get("snake_case"):
        df.columns = (df.columns.astype(str).str.strip().str.lower()
                      .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_"))
        log.append(("Converted column names to snake_case",
                    'df.columns = (df.columns.astype(str).str.strip().str.lower()'
                    '.str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_"))'))

    text = [c for c in df.columns if is_text(df[c])]

    if opts.get("strip_values") and text:
        for c in text:
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
        log.append((f"Stripped leading/trailing spaces in {len(text)} text column(s)",
                    f"for c in {text!r}:\n    df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)"))

    case = opts.get("case", "None")
    if case != "None" and text:
        fn = {"lower": "lower", "UPPER": "upper", "Title": "title"}[case]
        for c in text:
            df[c] = df[c].map(lambda v: getattr(v, fn)() if isinstance(v, str) else v)
        log.append((f"Converted text values to {case} case",
                    f"for c in {text!r}:\n    df[c] = df[c].map(lambda v: v.{fn}() if isinstance(v, str) else v)"))

    if opts.get("placeholders") and text:
        fixed = []
        for c in text:
            mask = df[c].notna() & df[c].astype(str).str.match(PLACEHOLDER_PATTERN)
            if mask.any():
                df.loc[mask, c] = np.nan
                fixed.append(c)
        if fixed:
            log.append((f"Converted placeholder text (N/A, null, '-', blanks...) to real missing values in {fixed}",
                        f"for c in {fixed!r}:\n"
                        f"    df.loc[df[c].notna() & df[c].astype(str).str.match(PLACEHOLDER_PATTERN), c] = np.nan"))

    if opts.get("convert_types"):
        for c in [c for c in df.columns if is_text(df[c])]:
            s = df[c].dropna()
            if s.empty:
                continue
            num = _convert_numeric(s)
            if num.notna().mean() >= 0.9:
                df[c] = _convert_numeric(df[c])
                log.append((f"Converted '{c}' from text to numbers",
                            f'df[{c!r}] = pd.to_numeric(df[{c!r}].astype(str).str.replace(",", "", regex=False), errors="coerce")'))
                continue
            sample = s.astype(str).head(300)
            if sample.str.contains(r"\d").mean() > 0.9:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.9:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df[c] = pd.to_datetime(df[c], errors="coerce")
                    log.append((f"Converted '{c}' from text to dates",
                                f'df[{c!r}] = pd.to_datetime(df[{c!r}], errors="coerce")'))

    if opts.get("drop_duplicates"):
        n = int(df.duplicated().sum())
        if n:
            df = df.drop_duplicates().reset_index(drop=True)
            log.append((f"Removed {n} exact duplicate row(s)", "df = df.drop_duplicates().reset_index(drop=True)"))

    if opts.get("drop_constant"):
        const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if const:
            df = df.drop(columns=const)
            log.append((f"Dropped constant columns {const}", f"df = df.drop(columns={const!r})"))

    return df, log


# ----------------------------------------------------------------------------
# Step 2 and 3: missing values and outliers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def preprocess_cached(df: pd.DataFrame, opts: dict):
    """Cached wrapper so the page does not recompute on every widget change."""
    return preprocess(df, opts)


def outlier_bounds(s: pd.Series, method: str, k: float):
    s = s.dropna()
    if method == "IQR rule":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr
    mu, sd = s.mean(), s.std()
    return mu - k * sd, mu + k * sd


def apply_cleaning(df: pd.DataFrame, strategies: dict, constants: dict, outlier_cfg: dict, knn_k: int = 5):
    """Apply the user's missing-value decisions and outlier treatment."""
    df = df.copy()
    log = []

    drop_cols = [c for c, s in strategies.items() if s == "Drop column" and c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log.append((f"Dropped columns with too much missing data: {drop_cols}", f"df = df.drop(columns={drop_cols!r})"))

    knn_targets = []
    for c, s in strategies.items():
        if c not in df.columns or s in ("Leave as is", "Drop rows with missing", "Drop column"):
            continue
        n_miss = int(df[c].isna().sum())
        if n_miss == 0:
            continue
        r = repr(c)
        if s == "Fill with mean":
            df[c] = df[c].fillna(df[c].mean())
            code = f"df[{r}] = df[{r}].fillna(df[{r}].mean())"
        elif s == "Fill with median":
            df[c] = df[c].fillna(df[c].median())
            code = f"df[{r}] = df[{r}].fillna(df[{r}].median())"
        elif s == "Fill with mode":
            mode = df[c].mode(dropna=True)
            if mode.empty:
                continue
            df[c] = df[c].fillna(mode.iloc[0])
            code = f"df[{r}] = df[{r}].fillna(df[{r}].mode(dropna=True).iloc[0])"
        elif s == "Fill with constant":
            raw = constants.get(c, "Unknown")
            value = raw
            if column_kind(df[c]) == "num":
                try:
                    value = float(raw)
                except ValueError:
                    continue
            df[c] = df[c].fillna(value)
            code = f"df[{r}] = df[{r}].fillna({value!r})"
        elif s == "Forward fill":
            df[c] = df[c].ffill()
            code = f"df[{r}] = df[{r}].ffill()"
        elif s == "Backward fill":
            df[c] = df[c].bfill()
            code = f"df[{r}] = df[{r}].bfill()"
        elif s == "Linear interpolation":
            df[c] = df[c].interpolate(method="linear", limit_direction="both")
            code = f'df[{r}] = df[{r}].interpolate(method="linear", limit_direction="both")'
        elif s == "KNN imputation":
            knn_targets.append(c)
            continue
        else:
            continue
        log.append((f"{s} applied to '{c}' ({n_miss} missing)", code))

    if knn_targets:
        num = [c for c in numeric_cols(df) if df[c].notna().any()]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[num])
        filled = scaler.inverse_transform(KNNImputer(n_neighbors=knn_k).fit_transform(scaled))
        imputed = pd.DataFrame(filled, columns=num, index=df.index)
        for c in knn_targets:
            df[c] = imputed[c]
        log.append((f"KNN imputation (k={knn_k}) on {knn_targets}, using all numeric columns as neighbours",
                    f"num = {num!r}\n"
                    "scaler = StandardScaler()\n"
                    "scaled = scaler.fit_transform(df[num])\n"
                    f"filled = scaler.inverse_transform(KNNImputer(n_neighbors={knn_k}).fit_transform(scaled))\n"
                    "imputed = pd.DataFrame(filled, columns=num, index=df.index)\n"
                    f"for c in {knn_targets!r}:\n    df[c] = imputed[c]"))

    drop_rows = [c for c, s in strategies.items() if s == "Drop rows with missing" and c in df.columns]
    if drop_rows:
        before = len(df)
        df = df.dropna(subset=drop_rows).reset_index(drop=True)
        log.append((f"Dropped {before - len(df)} row(s) with missing values in {drop_rows}",
                    f"df = df.dropna(subset={drop_rows!r}).reset_index(drop=True)"))

    method, cols = outlier_cfg.get("method", "Leave as is"), outlier_cfg.get("cols", [])
    cols = [c for c in cols if c in df.columns]
    if method != "Leave as is" and cols:
        k, action = outlier_cfg["k"], outlier_cfg["action"]
        keep = pd.Series(True, index=df.index)
        for c in cols:
            lo, hi = outlier_bounds(df[c], method, k)
            n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
            if n_out == 0:
                continue
            r = repr(c)
            if method == "IQR rule":
                bounds = (f"q1, q3 = df[{r}].quantile([0.25, 0.75])\niqr = q3 - q1\n"
                          f"lo, hi = q1 - {k} * iqr, q3 + {k} * iqr")
            else:
                bounds = f"lo, hi = df[{r}].mean() - {k} * df[{r}].std(), df[{r}].mean() + {k} * df[{r}].std()"
            if action.startswith("Cap"):
                df[c] = df[c].clip(lo, hi)
                log.append((f"Capped {n_out} outlier(s) in '{c}' to [{lo:.4g}, {hi:.4g}] ({method}, k={k})",
                            f"{bounds}\ndf[{r}] = df[{r}].clip(lo, hi)"))
            else:
                keep &= df[c].between(lo, hi) | df[c].isna()
                log.append((f"Flagged {n_out} outlier row(s) in '{c}' for removal ({method}, k={k})",
                            f"{bounds}\nkeep = df[{r}].between(lo, hi) | df[{r}].isna()\ndf = df[keep]"))
        if not keep.all():
            df = df[keep].reset_index(drop=True)

    return df, log


def default_options() -> dict:
    """Standardisation options used by the one-click analysis (conservative: no renaming, no case change)."""
    return {"strip_colnames": True, "snake_case": False, "strip_values": True, "placeholders": True,
            "convert_types": True, "drop_duplicates": True, "drop_constant": True, "case": "None"}


def safe_strategies(df: pd.DataFrame, max_row_loss: float = 0.05) -> dict:
    """Recommended decision per column, but never let 'drop rows' remove more than max_row_loss of the data."""
    strategies = {c: recommend_strategy(df[c])[0] for c in df.columns if df[c].isna().any()}
    drop_cols = [c for c, s in strategies.items() if s == "Drop rows with missing"]
    if drop_cols and len(df):
        lost = len(df) - len(df.dropna(subset=drop_cols))
        if lost / len(df) > max_row_loss:
            for c in drop_cols:
                kind = column_kind(df[c])
                if kind == "num":
                    strategies[c] = "Fill with median" if abs(df[c].skew()) > 0.5 else "Fill with mean"
                elif kind == "date":
                    strategies[c] = "Forward fill"
                else:
                    strategies[c] = "Fill with mode"
    return strategies


def auto_clean(df_raw: pd.DataFrame, max_row_loss: float = 0.05):
    """Apply the standard fixes plus the recommended missing-value decisions.
    Returns (cleaned_df, log, before_snapshot)."""
    prep, plog = preprocess(df_raw, default_options())
    strategies = safe_strategies(prep, max_row_loss)
    constants = {c: "Unknown" for c, s in strategies.items() if s == "Fill with constant"}
    clean, clog = apply_cleaning(prep, strategies, constants, {"method": "Leave as is"})
    before = {"rows": len(df_raw), "cols": df_raw.shape[1], "missing": int(df_raw.isna().sum().sum()),
              "dups": int(df_raw.duplicated().sum())}
    if clean.empty:
        return prep, plog, before
    return clean, plog + clog, before


def build_script(log: list) -> str:
    head = (
        '"""Reproducible cleaning script generated by Smart Data Science Studio."""\n'
        "import numpy as np\nimport pandas as pd\n"
        "from sklearn.impute import KNNImputer\nfrom sklearn.preprocessing import StandardScaler\n\n"
        f"PLACEHOLDER_PATTERN = {PLACEHOLDER_PATTERN!r}\n\n"
        'df = pd.read_csv("your_file.csv")   # or pd.read_excel("your_file.xlsx")\n\n'
    )
    body = "\n\n".join(f"# {desc.replace(chr(10), ' ')}\n{code}" for desc, code in log)
    return head + body + '\n\ndf.to_csv("cleaned_data.csv", index=False)\n'


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def _set_recommended(recs: dict):
    for key, value in recs.items():
        st.session_state[key] = value


def _snapshot(df: pd.DataFrame) -> dict:
    return {"rows": len(df), "cols": df.shape[1], "missing": int(df.isna().sum().sum()),
            "dups": int(df.duplicated().sum())}


def render_cleaning(df_raw: pd.DataFrame):
    st.subheader("Data Cleaning")
    st.caption("Your original data is never modified. Work through the three steps, then press Apply.")

    # ---- step 1 -----------------------------------------------------------
    st.markdown("#### Step 1: Standardise and fix")
    disguised = count_disguised_missing(df_raw)
    if len(disguised):
        st.info("Hidden missing values found (text such as 'N/A' or blanks): "
                + ", ".join(f"{c} ({n})" for c, n in disguised.items()))
    a, b, c_ = st.columns(3)
    opts = {
        "strip_colnames": a.checkbox("Strip spaces from column names", True),
        "snake_case": a.checkbox("Rename columns to snake_case", False),
        "strip_values": b.checkbox("Strip spaces inside text values", True),
        "placeholders": b.checkbox("Turn 'N/A', 'null', '-', blanks into missing", True),
        "convert_types": c_.checkbox("Auto-convert numbers/dates stored as text", True),
        "drop_duplicates": c_.checkbox("Remove duplicate rows", True),
        "drop_constant": a.checkbox("Drop constant columns", False),
        "case": b.selectbox("Text case", ["None", "lower", "UPPER", "Title"]),
    }
    df_prep, prep_log = preprocess_cached(df_raw, opts)
    st.caption(f"After step 1: {len(df_prep):,} rows, {df_prep.shape[1]} columns, "
               f"{int(df_prep.isna().sum().sum()):,} missing cells.")

    # ---- step 2 -----------------------------------------------------------
    st.markdown("#### Step 2: Decide what to do about missing data")
    miss_cols = [c for c in df_prep.columns if df_prep[c].isna().any()]
    miss_cols.sort(key=lambda c: -df_prep[c].isna().mean())
    strategies, constants, recs = {}, {}, {}
    if not miss_cols:
        st.success("No missing values remain after step 1.")
    else:
        st.caption("Each recommendation is based on how much is missing, the column type and skewness. "
                   "Change any decision you disagree with.")
        for c in miss_cols:
            kind = column_kind(df_prep[c])
            rec, _ = recommend_strategy(df_prep[c])
            recs[f"strat_{c}_{kind}"] = rec
        st.button("Apply recommended decisions to all columns", on_click=_set_recommended, args=(recs,))

        h = st.columns([2, 1.2, 3, 2.6])
        for col, label in zip(h, ["Column", "Missing", "Recommendation and why", "Your decision"]):
            col.markdown(f"**{label}**")
        for c in miss_cols:
            s = df_prep[c]
            kind = column_kind(s)
            rec, why = recommend_strategy(s)
            opts_list = strategy_options(s)
            row = st.columns([2, 1.2, 3, 2.6])
            row[0].write(f"{c}  \n`{s.dtype}`")
            row[1].write(f"{int(s.isna().sum()):,} ({s.isna().mean() * 100:.1f}%)")
            row[2].write(f"**{rec}**: {why}")
            key = f"strat_{c}_{kind}"
            if key not in st.session_state:
                st.session_state[key] = rec
            elif st.session_state[key] not in opts_list:
                st.session_state[key] = rec
            choice = row[3].selectbox("decision", opts_list, key=key, label_visibility="collapsed")
            strategies[c] = choice
            if choice == "Fill with constant":
                constants[c] = row[3].text_input("Constant", "Unknown" if kind == "cat" else "0",
                                                 key=f"const_{c}_{kind}")
        drop_rows_cols = [c for c, s in strategies.items() if s == "Drop rows with missing"]
        if drop_rows_cols:
            lost = len(df_prep) - len(df_prep.dropna(subset=drop_rows_cols))
            st.warning(f"'Drop rows' on {len(drop_rows_cols)} column(s) will remove {lost:,} rows "
                       f"({lost / max(len(df_prep), 1) * 100:.1f}% of the data).")
    knn_k = 5
    if any(s == "KNN imputation" for s in strategies.values()):
        knn_k = int(st.number_input("KNN neighbours (k)", 1, 30, 5))

    # ---- step 3 -----------------------------------------------------------
    st.markdown("#### Step 3: Outliers")
    nums = numeric_cols(df_prep)
    outlier_cfg = {"method": "Leave as is"}
    if nums:
        rows = []
        for c in nums:
            lo, hi = outlier_bounds(df_prep[c], "IQR rule", 1.5)
            rows.append({"column": c, "IQR outliers": int(((df_prep[c] < lo) | (df_prep[c] > hi)).sum()),
                         "lower fence": lo, "upper fence": hi})
        st.dataframe(pd.DataFrame(rows).set_index("column"))
        o1, o2, o3 = st.columns(3)
        method = o1.selectbox("Outlier method", ["Leave as is", "IQR rule", "Z-score"])
        if method != "Leave as is":
            default_k = 1.5 if method == "IQR rule" else 3.0
            k = o2.number_input("Threshold k", 0.5, 10.0, default_k, step=0.5)
            action = o3.selectbox("Action", ["Cap to bounds (winsorize)", "Remove rows"])
            cols = st.multiselect("Apply to columns", nums, default=[])
            outlier_cfg = {"method": method, "cols": cols, "k": float(k), "action": action}
        st.caption("Extreme values can be genuine. Review the counts before treating them.")

    # ---- apply ------------------------------------------------------------
    st.markdown("#### Step 4: Apply")
    if st.button("Apply cleaning", type="primary"):
        df_clean, log2 = apply_cleaning(df_prep, strategies, constants, outlier_cfg, knn_k)
        if df_clean.empty:
            st.error("These choices would remove every row. Please relax them.")
        else:
            full_log = prep_log + log2
            st.session_state["df_clean"] = df_clean
            st.session_state["clean_log"] = full_log
            st.session_state["clean_before"] = _snapshot(df_raw)
            st.success("Cleaning applied. The sidebar now lets you analyse the cleaned data.")

    df_clean = st.session_state.get("df_clean")
    if df_clean is not None:
        before, after = st.session_state["clean_before"], _snapshot(df_clean)
        m = st.columns(4)
        m[0].metric("Rows", f"{after['rows']:,}", f"{after['rows'] - before['rows']:+,}")
        m[1].metric("Columns", after["cols"], f"{after['cols'] - before['cols']:+d}")
        m[2].metric("Missing cells", f"{after['missing']:,}", f"{after['missing'] - before['missing']:+,}",
                    delta_color="inverse")
        m[3].metric("Duplicate rows", after["dups"], f"{after['dups'] - before['dups']:+d}", delta_color="inverse")

        log = st.session_state["clean_log"]
        st.write("**Cleaning log**")
        st.dataframe(pd.DataFrame({"Step": range(1, len(log) + 1), "Action": [d for d, _ in log]}).set_index("Step"))
        script = build_script(log)
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download cleaned data (CSV)", df_clean.to_csv(index=False).encode("utf-8"),
                           "cleaned_data.csv", "text/csv")
        try:
            buf = BytesIO()
            df_clean.to_excel(buf, index=False)
            d2.download_button("Download cleaned data (Excel)", buf.getvalue(), "cleaned_data.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass
        d3.download_button("Download cleaning script (.py)", script.encode("utf-8"), "cleaning_script.py", "text/x-python")
        with st.expander("View the generated Python cleaning script"):
            st.code(script, language="python")
        if st.button("Discard cleaned data"):
            for k in ("df_clean", "clean_log", "clean_before"):
                st.session_state.pop(k, None)
            st.rerun()

    show_method(
        "cleaning",
        [recommend_strategy, preprocess, apply_cleaning, outlier_bounds],
        {"pandas": "Core data structure: fillna, dropna, drop_duplicates, to_numeric, to_datetime, clip.",
         "numpy": "Represents missing values (np.nan) and numeric operations.",
         "scikit-learn": "KNNImputer fills gaps from similar rows; StandardScaler puts columns on one scale first.",
         "re / regex": "Detects hidden placeholders such as 'N/A', 'null' and '-'."},
    )
