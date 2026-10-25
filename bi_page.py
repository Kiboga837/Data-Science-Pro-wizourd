"""Dashboards page: a Power BI style interface.

Layout: filter pane on the left (slicers), dashboard on the right with tabs Overview, Explore, Custom visuals, Data
and Export. A control bar at the top holds the dashboard *parameters* (measure, aggregation, dimension, second
dimension, date field, granularity, top N, colour theme). Change a parameter or a slicer and every visual, KPI card
and insight updates, so the same dashboard answers many questions.
"""
import importlib.util
import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import bi_core as B
import datahub as hub
import llm

HAS_PLOTLY = importlib.util.find_spec("plotly") is not None
NONE = "(none)"


# ----------------------------------------------------------------------------
# Interactive (Plotly) renderer with a static fallback
# ----------------------------------------------------------------------------
def plotly_figure(plan: dict, palette: list):
    import plotly.express as px

    kind, d, x, y, color = plan["kind"], plan["data"], plan["x"], plan["y"], plan["color"]
    labels = {"value": plan["value_label"]}
    if kind == "Bar":
        fig = px.bar(d, x=x, y="value", color=color, barmode="group", color_discrete_sequence=palette, labels=labels,
                     category_orders={x: list(d[x].cat.categories)})
    elif kind == "Horizontal bar":
        fig = px.bar(d, x="value", y=x, color=color, orientation="h", barmode="group", color_discrete_sequence=palette,
                     labels=labels, category_orders={x: list(d[x].cat.categories)})
    elif kind == "Donut":
        fig = px.pie(d, names=x, values="value", hole=0.5, color_discrete_sequence=palette)
    elif kind == "Treemap":
        fig = px.treemap(d, path=[x], values="value", color_discrete_sequence=palette)
    elif kind == "Line":
        fig = px.line(d, x=x, y="value", color=color, markers=True, color_discrete_sequence=palette, labels=labels)
    elif kind == "Area":
        fig = px.area(d, x=x, y="value", color=color, color_discrete_sequence=palette, labels=labels)
    elif kind == "Scatter":
        fig = px.scatter(d, x=x, y=y, color=color if color in d.columns else None, opacity=0.65, color_discrete_sequence=palette)
    elif kind == "Histogram":
        fig = px.histogram(d, x=x, nbins=plan["bins"], color_discrete_sequence=[palette[0]])
    elif kind == "Box":
        fig = px.box(d, x=x, y=y, color=x, color_discrete_sequence=palette)
    elif kind in ("Heatmap", "Correlation"):
        corr = kind == "Correlation"
        fig = px.imshow(d, text_auto=".2f" if corr else True, aspect="auto",
                        color_continuous_scale="RdBu_r" if corr else "YlGnBu",
                        zmin=-1 if corr else None, zmax=1 if corr else None)
    else:
        raise ValueError(f"No interactive renderer for {kind}")
    fig.update_layout(title=plan["title"], margin={"l": 10, "r": 10, "t": 45, "b": 10}, height=360,
                      template="plotly_white", legend_title_text=color or "")
    return fig


def draw_plan(plan: dict, palette: list, key: str):
    """Draw one visual: interactive when Plotly works, otherwise the static version (and say why)."""
    if plan["empty"]:
        st.info(f"{plan['title']}: {plan['reason']}")
        return
    if plan["kind"] == "Table":
        st.markdown(f"**{plan['title']}**")
        st.dataframe(plan["data"].rename(columns={"value": plan["value_label"]}), hide_index=True)
        return
    if HAS_PLOTLY:
        try:
            st.plotly_chart(plotly_figure(plan, palette), key=key)
            return
        except Exception as e:  # noqa: BLE001 - never let one chart break the dashboard
            st.session_state.setdefault("bi_fallbacks", []).append(f"{plan['kind']}: {type(e).__name__}: {str(e)[:160]}")
    try:
        fig = B.static_figure(plan, palette)
    except Exception as e:  # noqa: BLE001 - one chart must never break the dashboard
        plt.close("all")
        st.warning(f"{plan['title']}: this chart could not be drawn ({type(e).__name__}: {e}).")
        return
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)


def interactive_html(plans: list, palette: list, title: str):
    """One self-contained HTML page with every interactive chart (needs Plotly; loads plotly.js from a CDN)."""
    parts = []
    for p in plans:
        if p["empty"] or p["kind"] == "Table":
            continue
        try:
            parts.append(plotly_figure(p, palette).to_html(full_html=False, include_plotlyjs="cdn" if not parts else False))
        except Exception:  # noqa: BLE001
            continue
    if not parts:
        return None
    body = "".join(f"<div style='width:48%;display:inline-block;vertical-align:top'>{h}</div>" for h in parts)
    return f"<html><head><meta charset='utf-8'><title>{title}</title></head><body style='font-family:sans-serif'><h2>{title}</h2>{body}</body></html>".encode("utf-8")


# ----------------------------------------------------------------------------
# Small state helpers (set state BEFORE the widget is drawn, never after)
# ----------------------------------------------------------------------------
def _select(container, label, options, key, default=None):
    opts = list(options)
    if st.session_state.get(key) not in opts:
        st.session_state[key] = default if default in opts else opts[0]
    return container.selectbox(label, opts, key=key)


def _reset_filters(prefix: str):
    for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(prefix)]:
        del st.session_state[k]


def _none(v):
    return None if v in (NONE, None, "") else v


# ----------------------------------------------------------------------------
# Filter pane
# ----------------------------------------------------------------------------
def filters_pane(df: pd.DataFrame, roles: dict, ds: str) -> dict:
    st.markdown("**Filters**")
    fields = roles["dims"] + roles["dates"] + roles["nums"]
    default = (roles["dims"][:2] + roles["dates"][:1] + roles["nums"][:1])
    skey = f"biw_slicers_{ds}"
    if skey in st.session_state:
        st.session_state[skey] = [f for f in st.session_state[skey] if f in fields]
    else:
        st.session_state[skey] = default
    chosen = st.multiselect("Slicer fields", fields, key=skey)
    filters = {"cats": {}, "nums": {}, "dates": {}}
    prefix = f"biwf_{ds}_"
    for col in chosen:
        key = prefix + col
        if col in roles["dates"]:
            lo, hi = df[col].min(), df[col].max()
            if pd.isna(lo):
                continue
            lo, hi = lo.date(), hi.date()
            cur = st.session_state.get(key)
            if not (isinstance(cur, (tuple, list)) and len(cur) in (1, 2) and all(lo <= pd.Timestamp(v).date() <= hi for v in cur)):
                st.session_state[key] = (lo, hi)
            val = st.date_input(col, min_value=lo, max_value=hi, key=key)
            if isinstance(val, (tuple, list)) and len(val) == 2 and (val[0], val[1]) != (lo, hi):
                filters["dates"][col] = (val[0], val[1])
        elif col in roles["nums"] and col not in roles["dims"]:
            lo, hi = df[col].min(), df[col].max()
            if pd.isna(lo) or lo == hi:
                continue
            integer = pd.api.types.is_integer_dtype(df[col])
            lo, hi = (int(lo), int(hi)) if integer else (float(lo), float(hi))
            cur = st.session_state.get(key)
            if not (isinstance(cur, (tuple, list)) and len(cur) == 2 and lo <= cur[0] <= cur[1] <= hi
                    and type(cur[0]) is type(lo)):
                st.session_state[key] = (lo, hi)
            val = st.slider(col, min_value=lo, max_value=hi, key=key)
            if tuple(val) != (lo, hi):
                filters["nums"][col] = (val[0], val[1])
        else:
            opts = sorted(df[col].dropna().astype(str).unique().tolist())[:300]
            cur = st.session_state.get(key, [])
            st.session_state[key] = [v for v in cur if v in opts]
            val = st.multiselect(col, opts, key=key)
            if val:
                filters["cats"][col] = val
    st.button("Reset filters", on_click=_reset_filters, args=(prefix,))
    return filters


# ----------------------------------------------------------------------------
# Custom visuals
# ----------------------------------------------------------------------------
def _custom_store(ds_name: str) -> list:
    return st.session_state.setdefault("bi_custom", {}).setdefault(ds_name, [])


def _add_custom(ds_name: str):
    ss = st.session_state
    spec = {"id": f"c{len(_custom_store(ds_name)) + 1}_{ss.get('bi_custom_counter', 0)}", "kind": ss["biw_cv_kind"],
            "title": ss.get("biw_cv_title") or ss["biw_cv_kind"], "x": _none(ss.get("biw_cv_x")),
            "y": _none(ss.get("biw_cv_y")) or B.ROWS, "agg": ss.get("biw_cv_agg", "Sum"),
            "color": _none(ss.get("biw_cv_color")), "top_n": int(ss.get("biw_cv_topn", 10))}
    ss["bi_custom_counter"] = ss.get("bi_custom_counter", 0) + 1
    _custom_store(ds_name).append(spec)


def _remove_custom(ds_name: str, spec_id: str):
    store = _custom_store(ds_name)
    store[:] = [s for s in store if s["id"] != spec_id]


AI_VISUAL_SYSTEM = f"""You turn a request into ONE dashboard visual. Reply with ONLY a JSON object, no other text:
{{"kind": one of {B.KINDS}, "title": short title, "x": a field name or null, "y": a numeric field name or "(Row count)",
 "agg": one of {B.AGGS}, "color": a field name or null, "top_n": integer 3-50}}
Use only the field names listed. Bar/Horizontal bar/Donut/Treemap/Box/Heatmap need x (a category). Line/Area use a date
field as x. Scatter uses two numeric fields (x and y). Histogram uses a numeric x. Heatmap uses color as the second
category. Use "(Row count)" as y to count records."""


def _ai_visual(ds_name: str, columns_text: str, roles: dict, all_columns: list):
    ss = st.session_state
    request = (ss.get("biw_ai_text") or "").strip()
    creds = llm.get_credentials()
    if not request:
        return
    if not creds.key:
        ss["bi_ai_msg"] = ("error", "Enter an API key on the SQL Chat or Questions & Answers page first (a free Gemini key works).")
        return
    try:
        reply = llm.make_client(creds).generate(creds.model, AI_VISUAL_SYSTEM,
                                                [{"role": "user", "content": f"Fields:\n{columns_text}\n\nRequest: {request}"}], 600)
        m = re.search(r"\{.*\}", reply, re.S)
        if not m:
            raise ValueError("The AI did not return a visual definition.")
        spec = B.validate_spec(json.loads(m.group(0)), roles, all_columns)
        spec["id"] = f"ai{len(_custom_store(ds_name)) + 1}_{ss.get('bi_custom_counter', 0)}"
        ss["bi_custom_counter"] = ss.get("bi_custom_counter", 0) + 1
        _custom_store(ds_name).append(spec)
        ss["bi_ai_msg"] = ("success", f"Added '{spec['title']}' ({spec['kind']}).")
    except (llm.LLMError, ValueError, json.JSONDecodeError) as e:
        ss["bi_ai_msg"] = ("error", f"Could not build that visual: {e}")


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
def render_dashboard(df_raw: pd.DataFrame, df_clean, dataset_name: str):
    hub.sync_main(df_raw, df_clean, dataset_name)
    st.subheader("Dashboards")
    st.caption("A Power BI style workspace: pick a dataset, change the controls and slicers, and every KPI, chart and insight "
               "updates. Add your own visuals, then export the dashboard as PDF, Word or interactive HTML.")
    names = list(hub.tables())
    if not names:
        st.info("No tables yet. Load a dataset in the sidebar.")
        return
    mains = hub.main_tables()
    default_ds = mains[-1] if mains else names[0]
    if st.session_state.get("biw_dataset") not in names:
        st.session_state["biw_dataset"] = default_ds
    ds_name = st.selectbox("Dataset", names, key="biw_dataset")
    version = hub.versions().get(ds_name, 0)
    ds = f"{ds_name}-{version}"
    cached = st.session_state.get("bi_prep")
    if not cached or cached[0] != ds:
        st.session_state["bi_prep"] = (ds, B.prepare_dataset(hub.tables()[ds_name]))
    full = st.session_state["bi_prep"][1]
    roles = B.column_roles(full)
    if not (roles["nums"] or roles["dims"] or roles["dates"]):
        st.warning("This table has no numeric, categorical or date columns that can be charted.")
        return
    defaults = B.default_params(roles)

    # ---- control bar (parameters) -------------------------------------------------------------
    with st.expander("Dashboard controls: change these to see different insights", expanded=True):
        a, b, c, d = st.columns(4)
        measure = _select(a, "Measure", [B.ROWS] + roles["nums"], f"biw_measure_{ds}", defaults["measure"])
        agg = _select(b, "Aggregation", B.AGGS, f"biw_agg_{ds}", defaults["agg"])
        dim = _select(c, "Analyse by (dimension)", roles["dims"] or [NONE], f"biw_dim_{ds}", defaults["dim"])
        dim2 = _select(d, "Break down by (second dimension)", [NONE] + roles["dims"], f"biw_dim2_{ds}", defaults["dim2"])
        e, f, g, h = st.columns(4)
        date = _select(e, "Date field", [NONE] + roles["dates"], f"biw_date_{ds}", defaults["date"])
        gran = _select(f, "Time granularity", list(B.GRANS), f"biw_gran_{ds}", "Month")
        if f"biw_topn_{ds}" not in st.session_state:
            st.session_state[f"biw_topn_{ds}"] = 10
        top_n = g.slider("Top N categories", 3, 30, key=f"biw_topn_{ds}")
        theme = _select(h, "Colour theme", list(B.PALETTES), "biw_theme", "Power BI")
    params = {"measure": measure, "agg": agg, "dim": _none(dim), "dim2": _none(dim2), "date": _none(date),
              "gran": gran, "top_n": top_n, "theme": theme}
    palette = B.PALETTES[theme]

    # ---- filter pane + dashboard --------------------------------------------------------------
    left, right = st.columns([1, 4])
    with left:
        filters = filters_pane(full, roles, ds)
    df = B.apply_filters(full, filters)
    with right:
        st.caption(f"{len(df):,} of {len(full):,} records match the filters"
                   + (": " + "; ".join(B.describe_filters(filters)) if B.describe_filters(filters) else "."))
        t_over, t_explore, t_custom, t_data, t_export = st.tabs(["Overview", "Explore", "Custom visuals", "Data", "Export"])
        custom = _custom_store(ds_name)
        specs_all = B.auto_specs(roles) + custom

        with t_over:
            cards = B.kpis(df, full, params)
            for col, card in zip(st.columns(len(cards)), cards):
                col.metric(card["label"], card["value"], card["delta"], delta_color="off")
            with st.expander("Automatic insights for the current view", expanded=True):
                for line in B.insights(df, full, params, roles):
                    st.markdown(f"- {line}".replace("$", "\\$"))
            autos = B.auto_specs(roles)
            for i in range(0, len(autos), 2):
                for col, spec in zip(st.columns(2), autos[i:i + 2]):
                    with col:
                        draw_plan(B.make_plan(spec, df, params, roles), palette, f"biv_{ds}_{spec['id']}")
            with st.expander("Chart engine status"):
                st.caption("Interactive charts: " + ("Plotly is installed." if HAS_PLOTLY else
                           "Plotly is not installed, so static charts are shown. Add `plotly` to requirements.txt."))
                for line in st.session_state.get("bi_fallbacks", [])[-5:]:
                    st.caption("Fallback used -> " + line)

        with t_explore:
            _explore(df, roles, params, palette, ds)

        with t_custom:
            _custom_tab(df, full, roles, params, palette, ds, ds_name, custom)

        with t_data:
            q = st.text_input("Search the filtered rows", key="biw_search")
            view = df
            if q:
                text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
                mask = pd.Series(False, index=df.index)
                for c in text_cols:
                    mask |= df[c].astype(str).str.contains(q, case=False, regex=False, na=False)
                view = df[mask]
            st.dataframe(view.head(5000))
            st.caption(f"{len(view):,} row(s)" + ("; first 5,000 shown" if len(view) > 5000 else ""))
            st.download_button("Download filtered data (CSV)", view.to_csv(index=False).encode("utf-8"),
                               "dashboard_data.csv", "text/csv", key="bi_dl_data")

        with t_export:
            _export_tab(df, full, roles, params, filters, specs_all, palette, ds_name)


def _explore(df, roles, params, palette, ds):
    st.markdown("**Pivot table**")
    dims = roles["dims"]
    if not dims:
        st.info("A pivot table needs at least one category field.")
        return
    a, b = st.columns(2)
    rk = f"biw_pivrows_{ds}"
    st.session_state[rk] = [v for v in st.session_state.get(rk, dims[:1]) if v in dims] or dims[:1]
    rows = a.multiselect("Rows (up to 2)", dims, key=rk, max_selections=2)
    colchoice = _select(b, "Columns", [NONE] + dims, f"biw_pivcol_{ds}", NONE)
    measure, agg = params["measure"], params["agg"]
    work = df.copy()
    value = measure
    if measure == B.ROWS:
        work["_rows"], value = 1, "_rows"
        agg = "Sum"
    fn = {"Sum": "sum", "Average": "mean", "Median": "median", "Min": "min", "Max": "max", "Count": "count",
          "Distinct count": "nunique"}[B.effective_agg(work, value, agg) if value != "_rows" else "Sum"]
    for c in rows + ([colchoice] if colchoice != NONE else []):
        work[c] = B._label(work[c])
    if rows and len(work):
        pivot = pd.pivot_table(work, index=rows, columns=None if colchoice == NONE else colchoice, values=value,
                               aggfunc=fn, margins=True, margins_name="Total")
        st.dataframe(pivot.round(2))
        st.download_button("Download pivot (CSV)", pivot.round(4).to_csv().encode("utf-8"), "pivot.csv", "text/csv", key="bi_dl_pivot")

    st.markdown("**Drill-down**")
    st.caption("Pick a value in the first level to see what it is made of in the second level.")
    if len(dims) < 2:
        st.info("Drill-down needs two category fields.")
        return
    c1, c2, c3 = st.columns(3)
    lvl1 = _select(c1, "Level 1", dims, f"biw_dd1_{ds}", dims[0])
    lvl2 = _select(c2, "Level 2", [d for d in dims if d != lvl1], f"biw_dd2_{ds}", None)
    values = sorted(df[lvl1].astype(str).unique().tolist())[:200]
    chosen = _select(c3, f"Value of {lvl1}", values or [NONE], f"biw_ddv_{ds}", None)
    sub = df[df[lvl1].astype(str) == chosen]
    spec = {"id": "dd", "title": f"{B.value_label(measure, params['agg'])} by {lvl2} within {lvl1} = {chosen}", "kind": "Bar",
            "x": lvl2, "y": measure, "agg": params["agg"]}
    draw_plan(B.make_plan(spec, sub, params, roles), palette, f"biv_{ds}_drill")


def _custom_tab(df, full, roles, params, palette, ds, ds_name, custom):
    all_cols = list(full.columns)
    st.markdown("**Describe a visual in words**")
    st.text_input("e.g. 'average profit by region as a horizontal bar' or 'monthly sales trend by category'", key="biw_ai_text")
    types = {c: ("number" if c in roles["nums"] else "date" if c in roles["dates"] else "category" if c in roles["dims"] else "text")
             for c in all_cols}
    cols_text = "\n".join(f"- {c} ({t})" for c, t in types.items())
    st.button("Create with AI", on_click=_ai_visual, args=(ds_name, cols_text, roles, all_cols))
    msg = st.session_state.get("bi_ai_msg")
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])

    st.markdown("**Or build one yourself**")
    a, b, c = st.columns(3)
    _select(a, "Visual type", B.KINDS, "biw_cv_kind", "Bar")
    st.session_state.setdefault("biw_cv_title", "")
    b.text_input("Title (optional)", key="biw_cv_title")
    st.session_state.setdefault("biw_cv_topn", 10)
    c.number_input("Top N", 3, 50, key="biw_cv_topn")
    d, e, f = st.columns(3)
    x_opts = [NONE] + roles["dims"] + roles["dates"] + roles["nums"]
    _select(d, "X axis / category", x_opts, "biw_cv_x", roles["dims"][0] if roles["dims"] else NONE)
    _select(e, "Value (Y)", [B.ROWS] + roles["nums"], "biw_cv_y", params["measure"])
    _select(f, "Aggregation", B.AGGS, "biw_cv_agg", "Sum")
    _select(st, "Colour / second category", [NONE] + roles["dims"], "biw_cv_color", NONE)
    st.button("Add visual", on_click=_add_custom, args=(ds_name,))

    if not custom:
        st.info("No custom visuals yet. They respond to the filters but use the fields you chose.")
    for i in range(0, len(custom), 2):
        for col, spec in zip(st.columns(2), custom[i:i + 2]):
            with col:
                draw_plan(B.make_plan(spec, df, params, roles), palette, f"biv_{ds}_{spec['id']}")
                st.button("Remove", key=f"bi_rm_{spec['id']}", on_click=_remove_custom, args=(ds_name, spec["id"]))


def _export_tab(df, full, roles, params, filters, specs, palette, ds_name):
    st.caption("Exports use the current filters and controls. PDF and Word contain static charts; the HTML file keeps "
               "interactive charts (hover, zoom) when Plotly is installed.")
    if st.button("Build dashboard report", type="primary"):
        blocks = B.build_blocks(df, full, params, roles, filters, specs, ds_name)
        st.session_state["bi_blocks"] = blocks
        from export_page import make_files

        meta = {"title": f"Dashboard Report: {ds_name}", "author": st.session_state.get("report_author", ""),
                "dataset": ds_name, "date": pd.Timestamp.today().date().isoformat()}
        st.session_state["bi_report"] = make_files(blocks, meta, "dashboard_report")
        plans = [B.make_plan(s, df, params, roles) for s in specs]
        st.session_state["bi_html"] = interactive_html(plans, palette, meta["title"]) if HAS_PLOTLY else None
    files = st.session_state.get("bi_report")
    if files:
        cols = st.columns(len(files["files"]) + 1)
        for col, (label, data, name, mime) in zip(cols, files["files"]):
            col.download_button(label, data, name, mime, key=f"bi_rep_{name}")
        if st.session_state.get("bi_html"):
            cols[-1].download_button("Download interactive dashboard (HTML)", st.session_state["bi_html"],
                                     "dashboard_interactive.html", "text/html", key="bi_rep_html")
        for w in files["warnings"]:
            st.warning(w)
