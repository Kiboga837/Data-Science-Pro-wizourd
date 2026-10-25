"""Dashboard engine (the logic behind the Dashboards page).

A dashboard is driven by a few *parameters* (measure, aggregation, dimension, second dimension, date field,
granularity, top N) plus *filters* (slicers). Every visual is a small *spec*; specs may refer to the parameters
(for example "@measure"), so changing a parameter re-draws every visual with a different view of the data.

This module has no Streamlit code: it turns (data, params, filters, spec) into plain DataFrames ("plans") that both
the interactive renderer and the static (PDF/Word) renderer draw.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import id_like_columns, is_text, numeric_cols
from eda import detect_date_columns

ROWS = "(Row count)"
AGGS = ["Sum", "Average", "Median", "Min", "Max", "Count", "Distinct count"]
_AGG_FN = {"Sum": "sum", "Average": "mean", "Median": "median", "Min": "min", "Max": "max",
           "Count": "count", "Distinct count": "nunique"}
GRANS = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}
KINDS = ["Bar", "Horizontal bar", "Line", "Area", "Donut", "Scatter", "Histogram", "Box", "Heatmap", "Treemap",
         "Correlation", "Table"]
PALETTES = {
    "Power BI": ["#01B8AA", "#374649", "#FD625E", "#F2C80F", "#5F6B6D", "#8AD4EB", "#FE9666", "#A66999"],
    "Ocean": ["#0B3C5D", "#328CC1", "#D9B310", "#1D2731", "#8FB8DE", "#5C6B73", "#9DB4C0", "#C2DFE3"],
    "Sunset": ["#FF6B6B", "#FFA94D", "#FFD43B", "#845EF7", "#F06595", "#4DABF7", "#69DB7C", "#868E96"],
    "Forest": ["#2B9348", "#55A630", "#80B918", "#AACC00", "#BFD200", "#D4D700", "#DDDF00", "#EEEF20"],
    "Slate": ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#6D6875", "#B5838D", "#457B9D"],
}


# ----------------------------------------------------------------------------
# Dataset and columns
# ----------------------------------------------------------------------------
def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with date-like text columns converted to real dates."""
    out = df.copy()
    for c in detect_date_columns(out):
        if not pd.api.types.is_datetime64_any_dtype(out[c]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def column_roles(df: pd.DataFrame) -> dict:
    ids = set(id_like_columns(df))
    nums = [c for c in numeric_cols(df) if c not in ids and not pd.api.types.is_bool_dtype(df[c])]
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cats = [c for c in df.columns if c not in ids and (is_text(df[c]) or pd.api.types.is_bool_dtype(df[c])
                                                        or isinstance(df[c].dtype, pd.CategoricalDtype))
            and 1 < df[c].nunique() <= 60]
    codes = [c for c in nums if pd.api.types.is_integer_dtype(df[c]) and 2 <= df[c].nunique() <= 12 and c not in cats]
    return {"nums": nums, "dates": dates, "cats": cats, "dims": cats + codes, "ids": ids}


def default_params(roles: dict) -> dict:
    preferred = ("sales", "revenue", "amount", "charge", "fee", "total", "price", "value", "profit", "income", "cost", "score", "count")
    nums = roles["nums"]
    pool = [n for n in nums if n not in roles["dims"]] or nums  # prefer real measures over small-integer codes and flags
    measure = next((n for p in preferred for n in pool if p in n.lower()), pool[0] if pool else ROWS)
    dims = roles["dims"]
    return {"measure": measure, "agg": "Sum" if nums else "Count", "dim": dims[0] if dims else None,
            "dim2": dims[1] if len(dims) > 1 else None, "date": roles["dates"][0] if roles["dates"] else None,
            "gran": "Month", "top_n": 10, "theme": "Power BI"}


# ----------------------------------------------------------------------------
# Filters (slicers)
# ----------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, vals in f.get("cats", {}).items():
        if vals and col in df.columns:
            mask &= df[col].astype(str).isin([str(v) for v in vals])
    for col, (lo, hi) in f.get("nums", {}).items():
        if col in df.columns:
            mask &= df[col].between(lo, hi)
    for col, (d1, d2) in f.get("dates", {}).items():
        if col in df.columns:
            mask &= (df[col] >= pd.Timestamp(d1)) & (df[col] < pd.Timestamp(d2) + pd.Timedelta(days=1))
    return df[mask]


def describe_filters(f: dict) -> list:
    lines = []
    for c, v in f.get("cats", {}).items():
        if v:
            lines.append(f"{c}: {', '.join(map(str, v[:6]))}{' ...' if len(v) > 6 else ''}")
    for c, (lo, hi) in f.get("nums", {}).items():
        lines.append(f"{c}: {lo:g} to {hi:g}")
    for c, (a, b) in f.get("dates", {}).items():
        lines.append(f"{c}: {pd.Timestamp(a):%Y-%m-%d} to {pd.Timestamp(b):%Y-%m-%d}")
    return lines


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def _label(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return s.astype(object).where(s.notna(), "(blank)").astype(str)


def value_label(measure, agg) -> str:
    return "Row count" if measure in (None, ROWS) else f"{agg} of {measure}"


def effective_agg(df, measure, agg) -> str:
    """Sum/Average/... need numbers; for text columns fall back to Count."""
    if measure in (None, ROWS):
        return "Count"
    if agg not in ("Count", "Distinct count") and not pd.api.types.is_numeric_dtype(df[measure]):
        return "Count"
    return agg


def scalar(df: pd.DataFrame, measure, agg):
    if len(df) == 0:
        return np.nan
    if measure in (None, ROWS):
        return float(len(df))
    agg = effective_agg(df, measure, agg)
    return float(getattr(df[measure], _AGG_FN[agg])())


def agg_frame(df: pd.DataFrame, by: list, measure, agg) -> pd.DataFrame:
    by = [b for b in by if b]
    use_rows = measure in (None, ROWS)
    work = df[by + ([] if use_rows else [measure])].copy()
    for b in by:
        work[b] = _label(work[b])
    if use_rows:
        return work.groupby(by).size().rename("value").reset_index()
    agg = effective_agg(df, measure, agg)
    return work.groupby(by)[measure].agg(_AGG_FN[agg]).rename("value").reset_index()


def time_frame(df, date_col, gran, measure, agg, color=None) -> pd.DataFrame:
    need = [date_col] + ([color] if color else []) + ([] if measure in (None, ROWS) else [measure])
    work = df[need].dropna(subset=[date_col]).copy()
    work["_period"] = pd.to_datetime(work[date_col]).dt.to_period(GRANS[gran]).dt.to_timestamp()
    frame = agg_frame(work, ["_period"] + ([color] if color else []), measure, agg)
    return frame.sort_values("_period")


def top_x(frame: pd.DataFrame, x: str, n: int) -> list:
    return frame.groupby(x)["value"].sum().sort_values(ascending=False).head(n).index.tolist()


# ----------------------------------------------------------------------------
# Visual specs and plans
# ----------------------------------------------------------------------------
def resolve(spec: dict, params: dict) -> dict:
    """Replace '@measure', '@dimension', '@dimension2', '@date', '@agg' with the current parameters."""
    table = {"@measure": params.get("measure"), "@dimension": params.get("dim"), "@dimension2": params.get("dim2"),
             "@date": params.get("date"), "@agg": params.get("agg")}
    out = {k: table.get(v, v) if isinstance(v, str) else v for k, v in spec.items()}
    out.setdefault("top_n", params.get("top_n", 10))
    if out.get("top_n") in (None, "@top_n"):
        out["top_n"] = params.get("top_n", 10)
    out.setdefault("agg", params.get("agg", "Sum"))
    return out


def auto_specs(roles: dict) -> list:
    """Default Overview visuals; they follow the dashboard parameters."""
    specs = []
    if roles["dims"]:
        specs.append({"id": "a1", "title": "@measure by @dimension", "kind": "Bar", "x": "@dimension", "y": "@measure", "agg": "@agg"})
        specs.append({"id": "a3", "title": "Share of @measure by @dimension", "kind": "Donut", "x": "@dimension", "y": "@measure", "agg": "@agg"})
    if roles["dates"]:
        specs.append({"id": "a2", "title": "@measure over time", "kind": "Line", "x": "@date", "y": "@measure", "agg": "@agg", "color": "@dimension"})
    if roles["nums"]:
        specs.append({"id": "a4", "title": "Distribution of @measure", "kind": "Histogram", "x": "@measure"})
    if len(roles["nums"]) >= 2:
        specs.append({"id": "a5", "title": "Relationship between numeric fields", "kind": "Scatter", "x": "@measure", "y": "@y2", "color": "@dimension"})
    if roles["dims"] and len(roles["dims"]) > 1:
        specs.append({"id": "a6", "title": "@measure: @dimension by @dimension2", "kind": "Heatmap", "x": "@dimension", "y": "@measure", "color": "@dimension2", "agg": "@agg"})
    if roles["dims"] and roles["nums"]:
        specs.append({"id": "a7", "title": "Spread of @measure by @dimension", "kind": "Box", "x": "@dimension", "y": "@measure"})
    if len(roles["nums"]) >= 3:
        specs.append({"id": "a8", "title": "Correlation of numeric fields", "kind": "Correlation"})
    return specs


def _fill_title(title: str, s: dict) -> str:
    return (title or "").replace("@measure", str(s.get("y") or "rows")).replace("@dimension2", str(s.get("color") or "")) \
        .replace("@dimension", str(s.get("x") or ""))


def make_plan(spec: dict, df: pd.DataFrame, params: dict, roles: dict) -> dict:
    """Turn a spec plus the filtered data into the data a chart needs. Never raises for bad inputs; returns
    {'empty': True, 'reason': ...} instead."""
    s = resolve(spec, params)
    if s.get("y") == "@y2":  # second numeric field for the default scatter
        others = [n for n in roles["nums"] if n != params.get("measure")]
        s["y"] = others[0] if others else None
    kind, n = s["kind"], int(s.get("top_n") or 10)
    plan = {"kind": kind, "title": _fill_title(s.get("title", ""), {**s, "y": s.get("y") if s.get("y") != ROWS else "rows"}),
            "x": s.get("x"), "y": s.get("y"), "color": s.get("color"), "data": None, "empty": False, "reason": "",
            "value_label": value_label(s.get("y"), s.get("agg")), "bins": int(s.get("bins") or 30)}

    def empty(reason):
        plan.update(empty=True, reason=reason)
        return plan

    if len(df) == 0:
        return empty("No rows match the current filters.")
    x, y, color, agg = s.get("x"), s.get("y"), s.get("color"), s.get("agg", "Sum")
    cols = set(df.columns)
    for field in (x, color):
        if field and field not in cols:
            return empty(f"Field '{field}' is not in this dataset.")
    if kind not in ("Scatter", "Histogram", "Box", "Correlation") and y not in (None, ROWS) and y not in cols:
        return empty(f"Field '{y}' is not in this dataset.")

    if kind == "Correlation":
        nums = [c for c in roles["nums"] if c in cols][:12]
        if len(nums) < 2:
            return empty("Needs at least two numeric fields.")
        plan["data"] = df[nums].corr()
        return plan

    if kind in ("Bar", "Horizontal bar", "Donut", "Treemap", "Table"):
        if not x:
            return empty("Choose a field to group by.")
        second = color if (color and color != x and kind in ("Bar", "Horizontal bar")) else None
        frame = agg_frame(df, [x] + ([second] if second else []), y, agg)
        keep = top_x(frame, x, n)
        rest = frame[~frame[x].isin(keep)]
        frame = frame[frame[x].isin(keep)]
        if kind == "Donut" and len(rest) and effective_agg(df, y, agg) in ("Sum", "Count") and second is None:
            frame = pd.concat([frame, pd.DataFrame({x: ["Other"], "value": [rest["value"].sum()]})], ignore_index=True)
        if kind in ("Donut", "Treemap") and (frame["value"] < 0).any():
            return empty("Share charts need non-negative values, but this measure has negative totals "
                         "(for example losses). Pick another measure, or use a bar chart.")
        order = frame.groupby(x)["value"].sum().sort_values(ascending=(kind == "Horizontal bar")).index.tolist()
        frame[x] = pd.Categorical(frame[x], categories=order, ordered=True)
        plan["data"] = frame.sort_values([x] + ([second] if second else [])).reset_index(drop=True)
        plan["color"] = second
        return plan

    if kind in ("Line", "Area"):
        if not x:
            return empty("Choose a field for the horizontal axis.")
        second = color if (color and color != x) else None
        if second and df[second].nunique() > 6:
            keep = df[second].astype(str).value_counts().head(6).index
            df = df[df[second].astype(str).isin(keep)]
        if pd.api.types.is_datetime64_any_dtype(df[x]):
            frame = time_frame(df, x, params.get("gran", "Month"), y, agg, second)
            frame = frame.rename(columns={"_period": x})
        else:
            frame = agg_frame(df, [x] + ([second] if second else []), y, agg)
            try:
                frame = frame.assign(_k=pd.to_numeric(frame[x])).sort_values("_k").drop(columns="_k")
            except (ValueError, TypeError):
                frame = frame.sort_values(x)
        if frame.empty:
            return empty("No dated rows to plot.")
        plan["data"], plan["color"] = frame.reset_index(drop=True), second
        return plan

    if kind == "Scatter":
        if not (x and y) or x not in cols or y not in cols:
            return empty("Choose two numeric fields.")
        use = [x, y] + ([color] if color and color in cols else []) + ([s["size"]] if s.get("size") in cols else [])
        data = df[use].dropna(subset=[x, y])
        if len(data) > 3000:
            data = data.sample(3000, random_state=1)
        if color and color in cols and data[color].nunique() > 8:
            top = data[color].astype(str).value_counts().head(8).index
            data = data[data[color].astype(str).isin(top)]
        if color in data.columns:
            data[color] = data[color].astype(str)
        plan["data"] = data.reset_index(drop=True)
        return plan

    if kind == "Histogram":
        if not x or x not in cols or not pd.api.types.is_numeric_dtype(df[x]):
            return empty("Choose a numeric field.")
        plan["data"] = df[[x]].dropna().reset_index(drop=True)
        return plan

    if kind == "Box":
        if not (x and y) or y not in cols or not pd.api.types.is_numeric_dtype(df[y]):
            return empty("Choose a category and a numeric field.")
        data = df[[x, y]].dropna(subset=[y]).copy()
        data[x] = _label(data[x])
        keep = data[x].value_counts().head(n).index
        plan["data"] = data[data[x].isin(keep)].reset_index(drop=True)
        return plan

    if kind == "Heatmap":
        second = color if color and color != x else None
        if not x:
            return empty("Choose a field for the columns.")
        if second:
            frame = agg_frame(df, [x, second], y, agg)
            keep_x, keep_y = top_x(frame, x, 15), top_x(frame, second, 15)
            frame = frame[frame[x].isin(keep_x) & frame[second].isin(keep_y)]
            plan["data"] = frame.pivot_table(index=second, columns=x, values="value", aggfunc="sum")
        elif params.get("date") in cols:
            frame = time_frame(df, params["date"], "Month", y, agg, x)
            frame["_period"] = frame["_period"].dt.strftime("%Y-%m")
            keep_x = top_x(frame.rename(columns={x: x}), x, 15)
            frame = frame[frame[x].isin(keep_x)]
            plan["data"] = frame.pivot_table(index="_period", columns=x, values="value", aggfunc="sum")
        else:
            return empty("Choose a second dimension.")
        if plan["data"].empty:
            return empty("Nothing to show.")
        return plan
    return empty(f"Unknown visual type '{kind}'.")


# ----------------------------------------------------------------------------
# KPIs and insights
# ----------------------------------------------------------------------------
def fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    a = abs(v)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= limit:
            return f"{v / limit:.1f}{suffix}"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def _latest_change(df, params):
    date, measure, agg = params.get("date"), params.get("measure"), params.get("agg")
    if not date or date not in df.columns:
        return None
    frame = time_frame(df, date, params.get("gran", "Month"), measure, agg)
    if len(frame) < 2:
        return None
    last, prev = frame.iloc[-1], frame.iloc[-2]
    pct = (last["value"] / prev["value"] - 1) * 100 if prev["value"] else np.nan
    return last["_period"], last["value"], pct


def kpis(df, full, params) -> list:
    m, agg, dim = params["measure"], params["agg"], params.get("dim")
    cards = []
    now, base = scalar(df, m, agg), scalar(full, m, agg)
    share = f"{now / base * 100:.0f}% of all data" if base and not np.isnan(now) and effective_agg(full, m, agg) in ("Sum", "Count") else None
    cards.append({"label": value_label(m, agg), "value": fmt_num(now), "delta": share})
    cards.append({"label": "Records", "value": f"{len(df):,}", "delta": f"{len(df) / max(len(full), 1) * 100:.0f}% of {len(full):,}"})
    if dim and dim in df.columns and len(df):
        frame = agg_frame(df, [dim], m, agg).sort_values("value", ascending=False)
        top = frame.iloc[0]
        tot = frame["value"].sum()
        cards.append({"label": f"Top {dim}", "value": str(top[dim])[:22],
                      "delta": f"{top['value'] / tot * 100:.0f}% of total" if tot and effective_agg(df, m, agg) in ("Sum", "Count") else fmt_num(top["value"])})
        cards.append({"label": f"Distinct {dim}", "value": f"{df[dim].nunique():,}", "delta": None})
    chg = _latest_change(df, params)
    if chg:
        period, val, pct = chg
        cards.append({"label": f"Latest {params.get('gran', 'Month').lower()} ({period:%Y-%m-%d})", "value": fmt_num(val),
                      "delta": None if np.isnan(pct) else f"{pct:+.1f}% vs previous"})
    return cards


def insights(df, full, params, roles) -> list:
    out = []
    m, agg, dim = params["measure"], params["agg"], params.get("dim")
    label = value_label(m, agg)
    if len(df) == 0:
        return ["No rows match the current filters."]
    if len(df) < len(full):
        now, base = scalar(df, m, agg), scalar(full, m, agg)
        extra = f"; {label.lower()} is {fmt_num(now)} versus {fmt_num(base)} overall" if not np.isnan(now) else ""
        out.append(f"Filters keep {len(df):,} of {len(full):,} records ({len(df) / len(full) * 100:.0f}%){extra}.")
    if dim and dim in df.columns:
        frame = agg_frame(df, [dim], m, agg).sort_values("value", ascending=False)
        if len(frame) >= 2:
            tot = frame["value"].sum()
            additive = effective_agg(df, m, agg) in ("Sum", "Count")
            first, last = frame.iloc[0], frame.iloc[-1]
            out.append(f"{first[dim]} leads on {label.lower()} at {fmt_num(first['value'])}"
                       + (f" ({first['value'] / tot * 100:.0f}% of the total)" if additive and tot else "") + ".")
            out.append(f"{last[dim]} is lowest at {fmt_num(last['value'])}.")
            if additive and len(frame) >= 4 and tot:
                out.append(f"The top 3 {dim} values account for {frame['value'].head(3).sum() / tot * 100:.0f}% of the total.")
    chg = _latest_change(df, params)
    if chg:
        period, val, pct = chg
        if not np.isnan(pct):
            out.append(f"In the latest {params.get('gran', 'Month').lower()} ({period:%Y-%m-%d}), {label.lower()} "
                       f"{'rose' if pct >= 0 else 'fell'} {abs(pct):.1f}% versus the previous period.")
        frame = time_frame(df, params["date"], params.get("gran", "Month"), m, agg)
        if len(frame) >= 3:
            peak = frame.loc[frame["value"].idxmax()]
            out.append(f"The peak {params.get('gran', 'Month').lower()} is {peak['_period']:%Y-%m-%d} at {fmt_num(peak['value'])}.")
    if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
        others = [c for c in roles["nums"] if c != m and c in df.columns]
        best = None
        for c in others:
            pair = df[[m, c]].dropna()
            if len(pair) > 10 and pair[m].std() > 0 and pair[c].std() > 0:
                r = pair[m].corr(pair[c])
                if best is None or abs(r) > abs(best[1]):
                    best = (c, r)
        if best and abs(best[1]) >= 0.3:
            out.append(f"{m} is most strongly related to {best[0]} (correlation {best[1]:+.2f}).")
        x = df[m].dropna()
        if len(x) > 10:
            q1, q3 = x.quantile([0.25, 0.75])
            n_out = int(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum())
            if n_out:
                out.append(f"{n_out:,} record(s) ({n_out / len(x) * 100:.1f}%) are outliers on {m} by the IQR rule.")
    return out or ["Not enough data to generate insights."]


# ----------------------------------------------------------------------------
# Static (matplotlib) rendering: used for PDF/Word export and as a fallback when interactive charts fail
# ----------------------------------------------------------------------------
def static_figure(plan: dict, palette: list):
    """A matplotlib Figure for the plan, or None when it has nothing to draw (or is a table)."""
    if plan["empty"] or plan["kind"] == "Table":
        return None
    kind, d, x, y, color = plan["kind"], plan["data"], plan["x"], plan["y"], plan["color"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    pal = palette
    if kind in ("Bar", "Treemap"):
        if color:
            piv = d.pivot_table(index=x, columns=color, values="value", aggfunc="sum", observed=True).reindex(d[x].cat.categories)
            piv.plot(kind="bar", ax=ax, color=pal[:piv.shape[1]], width=0.8)
            ax.legend(fontsize=7, title=color, title_fontsize=7)
        else:
            ax.bar(d[x].astype(str), d["value"], color=pal[0])
        ax.tick_params(axis="x", rotation=40, labelsize=7)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
        ax.set_ylabel(plan["value_label"])
    elif kind == "Horizontal bar":
        agg = d.groupby(x, observed=True)["value"].sum()
        ax.barh(agg.index.astype(str), agg.values, color=pal[0])
        ax.set_xlabel(plan["value_label"])
        ax.tick_params(axis="y", labelsize=7)
    elif kind == "Donut":
        ax.pie(d["value"], labels=d[x].astype(str), autopct="%1.0f%%", startangle=90, colors=(pal * 4)[:len(d)],
               wedgeprops={"width": 0.45}, textprops={"fontsize": 7})
        ax.set_aspect("equal")
    elif kind in ("Line", "Area"):
        series = d.groupby(color) if color else [(None, d)]
        for i, (name, g) in enumerate(series):
            g = g.sort_values(x)
            if kind == "Area":
                ax.fill_between(g[x], g["value"], alpha=0.45, color=pal[i % len(pal)], label=str(name) if name is not None else None)
            else:
                ax.plot(g[x], g["value"], marker="o", markersize=2.5, color=pal[i % len(pal)], label=str(name) if name is not None else None)
        ax.set_ylabel(plan["value_label"])
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        if color:
            ax.legend(fontsize=7, title=color, title_fontsize=7)
    elif kind == "Scatter":
        if color and color in d.columns:
            for i, (name, g) in enumerate(d.groupby(color)):
                ax.scatter(g[x], g[y], s=10, alpha=0.6, color=pal[i % len(pal)], label=str(name))
            ax.legend(fontsize=7, title=color, title_fontsize=7)
        else:
            ax.scatter(d[x], d[y], s=10, alpha=0.6, color=pal[0])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    elif kind == "Histogram":
        ax.hist(d[x], bins=plan["bins"], color=pal[0], edgecolor="white")
        ax.set_xlabel(x)
        ax.set_ylabel("Count")
    elif kind == "Box":
        names, groups = zip(*[(str(k)[:14], g[y].values) for k, g in d.groupby(x)])
        ax.boxplot(groups, patch_artist=True, boxprops={"facecolor": pal[0], "alpha": 0.6})  # no labels=: renamed across versions
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(y)
    elif kind in ("Heatmap", "Correlation"):
        data = d.astype(float)
        im = ax.imshow(data.values, cmap="RdBu_r" if kind == "Correlation" else "YlGnBu",
                       vmin=-1 if kind == "Correlation" else None, vmax=1 if kind == "Correlation" else None, aspect="auto")
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels([str(c)[:12] for c in data.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels([str(c)[:14] for c in data.index], fontsize=7)
        if data.size <= 100:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, fmt_num(data.values[i, j]) if kind == "Heatmap" else f"{data.values[i, j]:.2f}",
                            ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(plan["title"], fontsize=10)
    if kind not in ("Donut", "Heatmap", "Correlation"):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def static_png(plan: dict, palette: list):
    from report import png

    fig = static_figure(plan, palette)
    return None if fig is None else png(fig)


# ----------------------------------------------------------------------------
# Validation of AI-suggested visuals
# ----------------------------------------------------------------------------
def validate_spec(raw: dict, roles: dict, all_columns: list) -> dict:
    """Check a visual suggested by the AI against the dataset. Raises ValueError with a readable reason."""
    kind = str(raw.get("kind", "")).strip()
    match = next((k for k in KINDS if k.lower() == kind.lower()), None)
    if not match:
        raise ValueError(f"Unknown chart type '{kind}'. Choose from {', '.join(KINDS)}.")
    spec = {"kind": match, "title": str(raw.get("title") or match)[:80]}
    for field in ("x", "color", "size"):
        v = raw.get(field)
        if v not in (None, ""):
            if v not in all_columns:
                raise ValueError(f"The field '{v}' is not in this dataset.")
            spec[field] = v
    y = raw.get("y")
    if y not in (None, "", ROWS):
        if y not in all_columns:
            raise ValueError(f"The field '{y}' is not in this dataset.")
        spec["y"] = y
    else:
        spec["y"] = ROWS
    agg = str(raw.get("agg") or "Sum").title()
    spec["agg"] = agg if agg in AGGS else "Sum"
    try:
        spec["top_n"] = max(3, min(int(raw.get("top_n") or 10), 50))
    except (TypeError, ValueError):
        spec["top_n"] = 10
    if match in ("Bar", "Horizontal bar", "Donut", "Treemap", "Table", "Box", "Heatmap") and "x" not in spec:
        raise ValueError("This chart type needs a field to group by.")
    return spec


# ----------------------------------------------------------------------------
# Report blocks (same structure as the main report, so PDF/Word/HTML exporters can use them)
# ----------------------------------------------------------------------------
def build_blocks(df, full, params, roles, filters, specs, dataset_name: str) -> list:
    from report import Doc

    d = Doc()
    d.h1(f"Dashboard Report: {dataset_name}")
    d.p(f"Dataset: {dataset_name}. Measure: {value_label(params['measure'], params['agg'])}. "
        f"Grouped by {params.get('dim') or 'n/a'}"
        + (f", over time by {params['gran'].lower()} using {params['date']}" if params.get("date") else "") + ".")
    active = describe_filters(filters)
    d.p("Filters applied: " + ("; ".join(active) if active else "none (all records)") + f". {len(df):,} of {len(full):,} records.")
    cards = kpis(df, full, params)
    d.table(pd.DataFrame({"Indicator": [c["label"] for c in cards], "Value": [c["value"] for c in cards],
                          "Context": [c["delta"] or "" for c in cards]}), "Key indicators")
    d.h2("Insights")
    d.bullets(insights(df, full, params, roles))
    d.h2("Visuals")
    palette = PALETTES.get(params.get("theme"), PALETTES["Power BI"])
    for spec in specs:
        plan = make_plan(spec, df, params, roles)
        if plan["empty"]:
            d.p(f"{plan['title'] or spec.get('title', '')}: {plan['reason']}")
        elif plan["kind"] == "Table":
            d.table(plan["data"].rename(columns={"value": plan["value_label"]}), plan["title"], max_rows=30)
        else:
            try:
                img = static_png(plan, palette)
            except Exception as e:  # noqa: BLE001 - one chart must never break the report
                d.p(f"{plan['title']}: this chart could not be drawn ({type(e).__name__}: {e}).")
                continue
            if img:
                d.image(img, plan["title"])
    d.h2("Data sample")
    d.table(df.head(15), "First rows of the filtered data", max_rows=15)
    return d.blocks
