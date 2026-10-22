"""Exploratory Analysis and Time Trends pages."""
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from common import analysis_numeric_cols, categorical_cols, is_text, numeric_cols


def detect_date_columns(df: pd.DataFrame) -> list:
    """Columns that already are, or can be parsed as, dates."""
    found = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            found.append(col)
        elif is_text(s):
            sample = s.dropna().astype(str).head(200)
            if sample.empty or sample.str.contains(r"\d").mean() <= 0.9:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.9:
                found.append(col)
    return found


def _to_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


# ----------------------------------------------------------------------------
def render_eda(df: pd.DataFrame):
    st.subheader("Exploratory Data Analysis")
    num_cols = numeric_cols(df)
    cat_cols = categorical_cols(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate rows", int(df.duplicated().sum()))

    left, right = st.columns(2)
    with left:
        st.write("**Missing Values**")
        missing = df.isnull().sum().rename("Missing").to_frame()
        missing["% Missing"] = (missing["Missing"] / max(len(df), 1) * 100).round(2)
        st.dataframe(missing)
    with right:
        st.write("**Column Types**")
        st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame())

    if num_cols:
        st.write("**Summary Statistics**")
        st.dataframe(df[num_cols].describe().T)

        st.subheader("Numeric Columns Distribution")
        selected_col = st.selectbox("Select Column", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color="skyblue")
        ax.set_xlabel(selected_col)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Outlier Check")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_col].dropna(), ax=ax, color="lightcoral")
        st.pyplot(fig)
        plt.close(fig)

        if len(num_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Scatter Plot")
            sc1, sc2 = st.columns(2)
            x_axis = sc1.selectbox("X axis", num_cols, index=0, key="sx")
            y_axis = sc2.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1), key="sy")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_axis, y=y_axis, alpha=0.6, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("No numeric columns found in this dataset.")

    if cat_cols:
        st.subheader("Categorical Columns")
        cat_choice = st.selectbox("Select Categorical Column", cat_cols)
        counts = df[cat_choice].value_counts().head(15)
        fig, ax = plt.subplots()
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="steelblue")
        ax.set_xlabel("Count")
        ax.set_ylabel(cat_choice)
        st.pyplot(fig)
        plt.close(fig)

        if num_cols:
            st.write("**Average of a numeric column by category**")
            metric_col = st.selectbox("Numeric column", num_cols, key="grp_metric")
            grouped = df.groupby(cat_choice)[metric_col].mean().sort_values(ascending=False).head(15)
            st.bar_chart(grouped)


# ----------------------------------------------------------------------------
def render_trends(df: pd.DataFrame):
    st.subheader("Time-Based Trends")
    num_cols = numeric_cols(df)
    date_cols = detect_date_columns(df)
    if not date_cols or not num_cols:
        st.warning("This view needs at least one date column (e.g. Order Date) and one numeric column.")
        return

    c1, c2, c3, c4 = st.columns(4)
    date_col = c1.selectbox("Date column", date_cols)
    preferred = ("sales", "revenue", "amount", "total", "price", "value", "profit", "cost", "income", "score", "count")
    pool = analysis_numeric_cols(df) or num_cols
    first = next((n for p in preferred for n in pool if p in n.lower()), pool[0])
    default_idx = num_cols.index(first)
    value_col = c2.selectbox("Value column", num_cols, index=default_idx)
    agg_label = c3.selectbox("Aggregation", ["Sum", "Mean", "Count"])
    freq_label = c4.selectbox("Period", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], index=2)

    agg_fn = {"Sum": "sum", "Mean": "mean", "Count": "count"}[agg_label]
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}[freq_label]

    ts = df[[date_col, value_col]].copy()
    ts[date_col] = _to_datetime(ts[date_col])
    ts = ts.dropna(subset=[date_col])
    if ts.empty:
        st.error("No valid dates found in that column.")
        return

    ts["period"] = ts[date_col].dt.to_period(freq).dt.to_timestamp()
    series = ts.groupby("period")[value_col].agg(agg_fn).sort_index()
    st.caption(f"Data range: {ts[date_col].min().date()} to {ts[date_col].max().date()}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Periods", len(series))
    k2.metric("Best period", series.idxmax().strftime("%Y-%m-%d"), f"{series.max():,.0f}")
    k3.metric("Weakest period", series.idxmin().strftime("%Y-%m-%d"), f"{series.min():,.0f}")
    if len(series) > 1 and series.iloc[-2] != 0:
        growth = (series.iloc[-1] / series.iloc[-2] - 1) * 100
        k4.metric("Latest period change", f"{series.iloc[-1]:,.0f}", f"{growth:+.1f}%")
    else:
        k4.metric("Latest period", f"{series.iloc[-1]:,.0f}")

    window = st.slider("Moving average window (periods)", 1, 12, 3)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, marker="o", markersize=3, label=f"{agg_label} of {value_col}")
    if window > 1 and len(series) >= window:
        ax.plot(series.index, series.rolling(window).mean().values, color="crimson", linewidth=2,
                label=f"{window}-period moving average")
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Seasonality")
    s1, s2 = st.columns(2)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    by_month = ts.groupby(ts[date_col].dt.month)[value_col].agg(agg_fn).reindex(range(1, 13))
    by_month.index = months
    with s1:
        st.write("**By month of year**")
        st.bar_chart(by_month.fillna(0))
    by_dow = ts.groupby(ts[date_col].dt.dayofweek)[value_col].agg(agg_fn).reindex(range(7))
    by_dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    with s2:
        st.write("**By day of week**")
        st.bar_chart(by_dow.fillna(0))

    group_options = [c for c in categorical_cols(df, max_unique=15) if c != date_col]
    if group_options:
        st.subheader("Trend by Category")
        group_col = st.selectbox("Break down by", ["None"] + group_options)
        if group_col != "None":
            tmp = df[[date_col, value_col, group_col]].copy()
            tmp[date_col] = _to_datetime(tmp[date_col])
            tmp = tmp.dropna(subset=[date_col])
            tmp["period"] = tmp[date_col].dt.to_period(freq).dt.to_timestamp()
            pivot = tmp.pivot_table(index="period", columns=group_col, values=value_col,
                                    aggfunc=agg_fn).fillna(0)
            st.line_chart(pivot)
