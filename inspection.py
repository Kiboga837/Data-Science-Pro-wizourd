"""Data Inspection: find missing data and other quality problems before cleaning."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

from cleaning import count_disguised_missing, recommend_strategy
from common import is_text, numeric_cols, show_method


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = max(len(df), 1)
    miss = df.isna().sum()
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": miss,
        "% missing": (miss / n * 100).round(2),
        "unique values": df.nunique(),
    })
    out["severity"] = pd.cut(
        out["% missing"], bins=[-0.01, 0, 5, 20, 50, 100],
        labels=["None", "Low (<5%)", "Moderate (5-20%)", "High (20-50%)", "Severe (>50%)"],
    ).astype(str)
    out["suggested handling"] = [recommend_strategy(df[c])[0] for c in df.columns]
    return out


def missingness_vs_others(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Welch t-test: do other numeric columns differ when `col` is missing vs present?
    Differences suggest data are Missing At Random (MAR) rather than completely at random."""
    flag = df[col].isna()
    rows = []
    for other in numeric_cols(df):
        if other == col:
            continue
        a, b = df.loc[flag, other].dropna(), df.loc[~flag, other].dropna()
        if len(a) >= 3 and len(b) >= 3 and (a.std() > 0 or b.std() > 0):
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append({"compared column": other, "mean when missing": a.mean(),
                         "mean when present": b.mean(), "p-value": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["related to missingness (p<0.05)"] = out["p-value"] < 0.05
    return out


def quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    dups = int(df.duplicated().sum())
    if dups:
        issues.append(("Duplicate rows", "(whole table)", dups, "Remove duplicates"))
    for c in df.columns:
        s = df[c]
        if s.nunique(dropna=False) <= 1:
            issues.append(("Constant column", c, len(df), "Drop: carries no information"))
        if is_text(s):
            t = s.dropna().astype(str)
            ws = int((t != t.str.strip()).sum())
            if ws:
                issues.append(("Leading/trailing spaces", c, ws, "Strip whitespace"))
            if len(t):
                num = pd.to_numeric(t.str.replace(",", "", regex=False), errors="coerce")
                if num.notna().mean() >= 0.9:
                    issues.append(("Numbers stored as text", c, int(num.notna().sum()), "Convert to numeric"))
                if t.nunique() > 0.9 * len(t) and len(t) > 20:
                    issues.append(("Near-unique text (likely an ID)", c, t.nunique(), "Exclude from modelling"))
            if s.dropna().head(5000).map(type).nunique() > 1:
                issues.append(("Mixed data types in one column", c, len(t), "Inspect and unify types"))
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            neg = int((s < 0).sum())
            if neg:
                issues.append(("Negative values", c, neg, "Check these are valid (some measures can legitimately be negative)"))
            x = s.dropna()
            if len(x) > 4:
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                out = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
                if out:
                    issues.append(("Outliers (IQR rule)", c, out, "Review, then cap or keep"))
    hidden = count_disguised_missing(df)
    for c, n in hidden.items():
        issues.append(("Hidden missing values ('N/A', blank, '-')", c, int(n), "Convert to real missing values"))
    return pd.DataFrame(issues, columns=["Issue", "Column", "Count", "Suggested action"])


def render_inspection(df: pd.DataFrame, label: str):
    st.subheader("Data Inspection")
    st.caption(f"Inspecting: {label}")
    n_rows, n_cols = df.shape
    total_missing = int(df.isna().sum().sum())
    rows_missing = int(df.isna().any(axis=1).sum())

    m = st.columns(5)
    m[0].metric("Rows", f"{n_rows:,}")
    m[1].metric("Columns", n_cols)
    m[2].metric("Missing cells", f"{total_missing:,}", f"{total_missing / max(n_rows * n_cols, 1) * 100:.2f}%",
                delta_color="off")
    m[3].metric("Rows with any gap", f"{rows_missing:,}", f"{rows_missing / max(n_rows, 1) * 100:.1f}%",
                delta_color="off")
    m[4].metric("Duplicate rows", int(df.duplicated().sum()))

    summary = missing_summary(df)
    st.write("**Missing data by column**")
    st.dataframe(summary.sort_values("% missing", ascending=False))

    if total_missing == 0:
        st.success("No explicit missing values. Check the quality table below for hidden problems.")
    else:
        with_missing = summary[summary["missing"] > 0].sort_values("% missing", ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Percentage missing per column**")
            fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(with_missing))))
            sns.barplot(x=with_missing["% missing"], y=with_missing.index, ax=ax, color="indianred")
            ax.set_xlabel("% missing")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            st.write("**Missing-value map** (dark = missing, up to 300 sampled rows)")
            sample = df[with_missing.index].sample(min(300, n_rows), random_state=1).sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(sample.isna(), cbar=False, cmap="Greys", yticklabels=False, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        c3, c4 = st.columns(2)
        with c3:
            st.write("**How many values are missing per row**")
            per_row = df.isna().sum(axis=1).value_counts().sort_index()
            st.bar_chart(per_row.rename("rows"))
        with c4:
            if len(with_missing) > 1:
                st.write("**Do gaps occur together?** (correlation of missingness)")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df[with_missing.index].isna().astype(int).corr(), annot=True, fmt=".2f",
                            cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Only one column has gaps, so there is no co-occurrence to show.")

        st.write("**Is the missingness random?**")
        col = st.selectbox("Column to test", with_missing.index.tolist())
        res = missingness_vs_others(df, col)
        if res.empty:
            st.info("Not enough numeric data to compare.")
        else:
            st.dataframe(res.set_index("compared column"))
            if res["related to missingness (p<0.05)"].any():
                st.warning("Other columns behave differently when this value is missing, so the gaps are probably "
                           "**Missing At Random (MAR)**. Deleting rows could bias results; prefer imputation "
                           "(median/mode, or KNN for numeric columns).")
            else:
                st.success("No relationship found, which is consistent with **Missing Completely At Random "
                           "(MCAR)**. Dropping rows or simple median/mode filling is usually safe. "
                           "(A test cannot rule out data that is missing because of its own value.)")

    st.write("**Data quality checks**")
    issues = quality_issues(df)
    if issues.empty:
        st.success("No further quality issues detected.")
    else:
        st.dataframe(issues.set_index("Issue"))

    st.info("Next step: open **Data Cleaning** in the sidebar to apply the decisions.")
    show_method(
        "inspection",
        [missing_summary, missingness_vs_others, quality_issues, count_disguised_missing],
        {"pandas": "isna() finds gaps; groupby/value_counts summarise them per column and per row.",
         "scipy.stats": "Welch t-test checks whether missingness depends on other columns (MCAR vs MAR).",
         "seaborn / matplotlib": "Draw the missing-value map and bar charts.",
         "numpy": "Numeric helpers for fences and masks."},
    )
