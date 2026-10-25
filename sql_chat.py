"""SQL Chat page: ask for what you want in plain language; the app writes the SQL, runs it and shows the output.

Flow per message: schema + conversation -> AI writes SQLite SQL -> read-only execution -> if it fails, the error is
sent back and the AI corrects it (up to 2 retries) -> result table, chart and actions.
"""
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import datahub as hub
import llm
import sql_engine
from sql_engine import SQLRunError, SQLSession

SQL_SYSTEM = """You are an expert data analyst who writes SQLite SQL for the database described below.

Rules:
- SQLite dialect only. Read-only: write exactly one SELECT statement (a WITH ... SELECT is fine). Never modify data.
- Use the table and column names exactly as in the schema and quote them with double quotes, e.g. "Order Date".
  Text values use SINGLE quotes, e.g. WHERE "Region" = 'East'. Never use double quotes for values.
- Advanced SQL is welcome when it fits the request: window functions (ROW_NUMBER, RANK, LAG, SUM() OVER, NTILE),
  CTEs, subqueries, CASE, joins, UNION, HAVING.
- Date and time columns are TEXT in ISO format 'YYYY-MM-DD HH:MM:SS': use strftime('%Y-%m', col), date(col),
  julianday(). There is no DATE_TRUNC, ILIKE, QUALIFY or ::cast. Use CAST(x AS REAL) or multiply by 1.0 for
  decimal division, and LIKE for text matching (case-insensitive for ASCII).
- Extra functions available: STDEV, VARIANCE, MEDIAN, PERCENTILE(x, p) with p 0-100, CORR(x, y), SQRT, POWER, LN,
  LOG10, EXP, FLOOR, CEIL and the REGEXP operator.
- Give clear column aliases. ROUND money and percentages. Add ORDER BY when the order matters, and LIMIT 1000 unless
  the query aggregates to a small result.
- If the data may contain padded or inconsistent text (see the schema values), clean it with TRIM() in the query.

Reply in exactly this format:
<explanation>One to three sentences: what the query does and any assumption you made.</explanation>
<sql>
the query
</sql>
If the user only wants something explained and no new query is needed, put the answer in <explanation> and omit <sql>."""


# ----------------------------------------------------------------------------
# AI -> SQL -> result
# ----------------------------------------------------------------------------
def parse_reply(text: str) -> dict:
    def grab(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S | re.I)
        return m.group(1).strip() if m else ""

    sql = grab("sql")
    if not sql:
        fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
        sql = fence.group(1).strip() if fence else ""
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql).strip()
    explanation = grab("explanation") or (re.sub(r"<[^>]+>|```.*?```", "", text, flags=re.S).strip() if not sql else "")
    return {"explanation": explanation, "sql": sql}


def generate_and_run(client, model: str, question: str, session: SQLSession, history: list = None,
                     max_retries: int = 2) -> dict:
    """Turn a request into SQL and execute it. Raises llm.LLMError for provider problems."""
    system = SQL_SYSTEM + "\n\nDATABASE SCHEMA:\n" + session.schema_text()
    messages = []
    for h in (history or [])[-4:]:
        messages += [{"role": "user", "content": h["question"]},
                     {"role": "assistant", "content": f"<explanation>{h.get('explanation', '')}</explanation>\n<sql>\n{h['sql']}\n</sql>"}]
    messages.append({"role": "user", "content": question})
    out = {"question": question, "explanation": "", "sql": "", "df": None, "info": None, "error": None, "attempts": 0}
    for attempt in range(1, max_retries + 2):
        out["attempts"] = attempt
        reply = client.generate(model, system, messages, 2000)
        parsed = parse_reply(reply)
        out["explanation"] = parsed["explanation"]
        if not parsed["sql"]:
            out["kind"] = "answer"
            return out
        out["sql"], out["kind"] = parsed["sql"], "query"
        try:
            out["df"], out["info"] = session.run(parsed["sql"])
            out["error"] = None
            return out
        except SQLRunError as e:
            out["error"] = str(e)
            if attempt <= max_retries:
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": f"That query failed with this error:\n{e}\n"
                                                         "Return a corrected <explanation> and <sql> in the same format."}]
    return out


def example_prompts(info: dict) -> list:
    """Starter requests built from the user's own columns."""
    for t in info.values():
        cols = t["columns"]
        nums = [c["name"] for c in cols if c["kind"] == "num" and not c["is_id"]]
        cats = [c["name"] for c in cols if c["kind"] == "cat" and not c["is_id"]]
        dates = [c["name"] for c in cols if c["kind"] == "date"]
        if nums:
            ex = [f"How many rows are in {t['name']}, and what are the average, minimum and maximum of {nums[0]}?"]
            if cats:
                ex.append(f"Show the top 5 {cats[0]} values by total {nums[0]}, with each one's share of the overall total.")
                ex.append(f"For each {cats[0]}, find the rows where {nums[0]} is above that group's average.")
            if dates:
                ex.append(f"Show the monthly total of {nums[0]} with a running total and month-over-month growth.")
            return ex[:4]
    return ["List the tables and how many rows each has."]


# ----------------------------------------------------------------------------
# Charts and reports
# ----------------------------------------------------------------------------
def _temporal(s: pd.Series, name: str) -> bool:
    if re.search(r"date|month|year|week|day|period|quarter|time", str(name), re.I):
        return True
    return bool(s.astype(str).head(20).str.match(r"^\d{4}(-\d{2})?(-\d{2})?").all())


def auto_chart(df: pd.DataFrame):
    """(kind, x, [y...]) for a quick chart, or None when the result is not chartable."""
    if df is None or df.shape[1] < 2 or len(df) < 2:
        return None
    x = df.columns[0]
    ys = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])][:6]
    if not ys or df[x].nunique() < 2:
        return None
    return ("line" if _temporal(df[x], x) else "bar"), x, ys


def draw_chart(df: pd.DataFrame, kind: str, x: str, ys: list):
    data = df.head(60).copy()
    data[x] = data[x].astype(str)
    data = data.set_index(x)[ys]
    {"line": st.line_chart, "area": st.area_chart}.get(kind, st.bar_chart)(data)


def chart_png(df: pd.DataFrame):
    spec = auto_chart(df)
    if not spec:
        return None
    kind, x, ys = spec
    ys = ys[:1]  # one series: columns on different scales (e.g. a total and a count) would hide each other
    data = df.head(40)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = data[x].astype(str)
    if kind == "line":
        for y in ys:
            ax.plot(labels, data[y], marker="o", markersize=3, label=y)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        step = max(1, len(labels) // 12)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_xticklabels(labels.iloc[::step], rotation=60, fontsize=7)
    else:
        width = 0.8 / len(ys)
        for i, y in enumerate(ys):
            ax.bar([p + i * width for p in range(len(labels))], data[y], width=width, label=y)
        ax.set_xticks([p + 0.4 - width / 2 for p in range(len(labels))])
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    from report import png
    return png(fig)


def build_sql_blocks(msgs: list, tables_used: list) -> list:
    """Report blocks (same format as the main report) for the SQL session."""
    from report import Doc

    d = Doc()
    answered = [m for m in msgs if m["role"] == "assistant" and (m.get("sql") or m.get("explanation"))]
    d.h1("SQL Analysis Session")
    d.p(f"{len(answered)} request(s) were written in plain language, converted to SQL, executed read-only on the tables "
        f"{', '.join(tables_used) or '(none)'} and are recorded below with their results.")
    for i, m in enumerate(answered, 1):
        d.h2(f"Request {i}: {m['question']}")
        if m.get("explanation"):
            d.p(m["explanation"])
        if m.get("sql"):
            d.h3("SQL query")
            d.code(m["sql"])
        if m.get("error"):
            d.callout(f"The query failed: {m['error']}")
        if m.get("df") is not None:
            info = m.get("info") or {}
            d.table(m["df"], f"Result: {len(m['df']):,} row(s) in {info.get('seconds', 0)}s", max_rows=25)
            img = chart_png(m["df"])
            if img:
                d.image(img, "Chart of the result")
        if m.get("explain"):
            d.h3("Interpretation")
            d.p(m["explain"])
    return d.blocks


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
def _ask(text: str):
    st.session_state["sql_pending"] = text


def _to_editor(sql: str):
    st.session_state["sqlw_editor"] = sql
    st.session_state["sqlw_mode"] = "SQL editor"


def _save_result(df: pd.DataFrame, name_key: str):
    name = hub.add_table(st.session_state.get(name_key) or "query_result", df, "query")
    st.session_state["sql_saved_msg"] = f"Saved as table '{name}'. It is available to new queries and to Dashboards."


def _explain(index: int, model: str, creds):
    m = st.session_state["sql_msgs"][index]
    try:
        client = llm.make_client(creds)
        text = client.generate(
            model, "You explain query results for business readers in 2 to 5 sentences. Use only the numbers shown.",
            [{"role": "user", "content": f"Request: {m['question']}\nSQL:\n{m['sql']}\nResult (first rows):\n{m['df'].head(30).to_string()}"}], 600)
        m["explain"] = text.strip()
    except llm.LLMError as e:
        m["explain"] = f"Could not explain: {e}"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def get_session():
    t = hub.tables()
    key = tuple(sorted(hub.versions().items()))
    if st.session_state.get("sql_session_key") != key or "sql_session" not in st.session_state:
        st.session_state["sql_session"] = SQLSession(dict(t)) if t else None
        st.session_state["sql_session_key"] = key
    return st.session_state["sql_session"]


def _show_result(df: pd.DataFrame, info: dict, key: str, sql: str = None):
    if df is None:
        return
    st.dataframe(df.head(1000))
    note = f"{len(df):,} row(s) in {info['seconds']}s" if info else f"{len(df):,} row(s)"
    if info and info.get("truncated"):
        note += f" (stopped at the {sql_engine.MAX_ROWS:,}-row limit)"
    st.caption(note + ("; first 1,000 shown" if len(df) > 1000 else ""))
    a, b = st.columns(2)
    a.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "query_result.csv", "text/csv", key=f"sql_dl_{key}")
    spec = auto_chart(df)
    with st.expander("Chart"):
        if not spec:
            st.caption("This result has no category/number layout that can be charted.")
        else:
            kind0, x, ys = spec
            st.session_state.setdefault(f"sqlw_ck_{key}", kind0)
            ck = f"sqlw_cy_{key}"
            st.session_state[ck] = [v for v in st.session_state.get(ck, ys[:1]) if v in ys] or ys[:1]
            c1, c2 = st.columns(2)
            kind = c1.selectbox("Chart type", ["bar", "line", "area"], key=f"sqlw_ck_{key}")
            chosen = c2.multiselect("Series (columns on different scales hide each other)", ys, key=ck)
            draw_chart(df, kind, x, chosen or ys[:1])
    with st.expander("Save as a table (for dashboards and further queries)"):
        st.session_state.setdefault(f"sqlw_name_{key}", f"query_{key}")
        st.text_input("Table name", key=f"sqlw_name_{key}")
        st.button("Save table", key=f"sql_savebtn_{key}", on_click=_save_result, args=(df, f"sqlw_name_{key}"))


def render_sql_chat(df_raw: pd.DataFrame, df_clean, dataset_name: str):
    hub.sync_main(df_raw, df_clean, dataset_name)
    st.subheader("SQL Chat")
    st.caption("Describe what you want in plain language. The app writes the SQL, runs it read-only on your data and "
               "shows the output. You can also write SQL yourself, browse a cookbook of advanced techniques, and add "
               "more tables to join.")

    with st.expander("AI provider and key (a free Gemini key works)", expanded=not llm.get_credentials().key):
        creds = llm.render_llm_settings()

    session = get_session()
    if session is None:
        st.warning("No tables available yet.")
        return
    if st.session_state.get("sql_saved_msg"):
        st.success(st.session_state.pop("sql_saved_msg"))

    st.session_state.setdefault("sqlw_mode", "Chat")
    mode = st.radio("View", ["Chat", "SQL editor", "Cookbook", "Tables and schema"], horizontal=True, key="sqlw_mode")

    if mode == "Chat":
        _chat(session, creds)
    elif mode == "SQL editor":
        _editor(session)
    elif mode == "Cookbook":
        _cookbook(session)
    else:
        st.markdown("**Tables available** (the main dataset, your uploads and saved results)")
        hub.render_table_manager()
        for name, t in session.info.items():
            with st.expander(f"{name}  ({t['rows']:,} rows)"):
                st.dataframe(pd.DataFrame([{"Column": c["name"], "SQL type": c["type"], "Details": c["info"],
                                            "Nulls": c["nulls"]} for c in t["columns"]]).set_index("Column"))


def _chat(session: SQLSession, creds):
    msgs = st.session_state.setdefault("sql_msgs", [])
    prompt = st.chat_input("Ask about your data, e.g. 'Top 5 regions by revenue with their share of total'")
    pending = st.session_state.pop("sql_pending", None)
    prompt = prompt or pending
    if not msgs and not prompt:
        st.markdown("**Try one of these:**")
        for i, ex in enumerate(example_prompts(session.info)):
            st.button(ex, key=f"sql_ex_{i}", on_click=_ask, args=(ex,))

    if prompt:
        msgs.append({"role": "user", "content": prompt})
        history = [{"question": m["question"], "explanation": m.get("explanation", ""), "sql": m["sql"]}
                   for m in msgs if m["role"] == "assistant" and m.get("sql") and not m.get("error")]
        if not creds.key:
            msgs.append({"role": "assistant", "question": prompt, "error": "Enter an API key above first (a free Gemini key works).",
                         "explanation": "", "sql": "", "df": None})
        else:
            with st.spinner("Writing and running the query..."):
                try:
                    res = generate_and_run(llm.make_client(creds), creds.model, prompt, session, history)
                except llm.LLMError as e:
                    res = {"question": prompt, "explanation": "", "sql": "", "df": None, "info": None, "error": str(e)}
            msgs.append({"role": "assistant", **res})

    for i, m in enumerate(msgs):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
                continue
            if m.get("explanation"):
                st.markdown(m["explanation"])
            if m.get("sql"):
                st.code(m["sql"], language="sql")
                if m.get("attempts", 1) > 1 and not m.get("error"):
                    st.caption(f"The first query failed, so it was corrected automatically ({m['attempts']} attempts).")
            if m.get("error"):
                st.error(m["error"])
            if m.get("df") is not None:
                _show_result(m["df"], m.get("info"), str(i // 2 + 1))
                c1, c2 = st.columns(2)
                c1.button("Open in SQL editor", key=f"sql_edit_{i}", on_click=_to_editor, args=(m["sql"],))
                c2.button("Explain these results", key=f"sql_exp_{i}", on_click=_explain, args=(i, creds.model, creds))
                if m.get("explain"):
                    st.info(m["explain"])

    if msgs:
        a, b = st.columns(2)
        if a.button("Clear conversation"):
            st.session_state["sql_msgs"] = []
            st.session_state.pop("sql_report", None)
            st.rerun()
        if b.button("Prepare session report (PDF and Word)"):
            _prepare_report(session, msgs)
        _report_downloads()


def _prepare_report(session, msgs):
    blocks = build_sql_blocks(msgs, list(session.info))
    st.session_state["sql_blocks"] = blocks
    from export_page import make_files

    meta = {"title": "SQL Analysis Session", "author": st.session_state.get("report_author", ""),
            "dataset": ", ".join(session.info), "date": pd.Timestamp.today().date().isoformat()}
    st.session_state["sql_report"] = make_files(blocks, meta, "sql_session")


def _report_downloads():
    files = st.session_state.get("sql_report")
    if not files:
        return
    cols = st.columns(len(files["files"]) or 1)
    for col, (label, data, name, mime) in zip(cols, files["files"]):
        col.download_button(label, data, name, mime, key=f"sql_rep_{name}")
    for w in files["warnings"]:
        st.warning(w)


def _editor(session: SQLSession):
    first = next(iter(session.info))
    st.session_state.setdefault("sqlw_editor", f'SELECT *\nFROM {sql_engine.qi(first)}\nLIMIT 100')
    st.text_area("SQL (read-only: SELECT or WITH ... SELECT)", key="sqlw_editor", height=220)
    if st.button("Run query", type="primary"):
        try:
            df, info = session.run(st.session_state["sqlw_editor"])
            st.session_state["sql_editor_result"] = {"df": df, "info": info, "error": None}
        except SQLRunError as e:
            st.session_state["sql_editor_result"] = {"df": None, "info": None, "error": str(e)}
    res = st.session_state.get("sql_editor_result")
    if res:
        if res["error"]:
            st.error(res["error"])
        else:
            _show_result(res["df"], res["info"], "editor")


def _cookbook(session: SQLSession):
    st.caption("Ready-made advanced techniques, filled in with your own tables and columns. Run them as they are or "
               "send them to the editor and adapt them.")
    results = st.session_state.setdefault("sql_cook_results", {})
    for i, t in enumerate(sql_engine.cookbook(session.info)):
        with st.expander(t["title"]):
            st.caption(t["tip"])
            st.code(t["sql"], language="sql")
            a, b = st.columns(2)
            a.button("Use in editor", key=f"sql_cb_use_{i}", on_click=_to_editor, args=(t["sql"],))
            if b.button("Run here", key=f"sql_cb_run_{i}"):
                try:
                    results[i] = session.run(t["sql"])
                except SQLRunError as e:
                    results[i] = (None, str(e))
            if i in results:
                df, info = results[i]
                st.error(info) if df is None else st.dataframe(df.head(200))
