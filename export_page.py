"""Report Export: PDF, Word and HTML downloads for the main report, the dashboard and the SQL session."""
import pandas as pd
import streamlit as st

import pdf_export


def make_files(blocks: list, meta: dict, stem: str) -> dict:
    """Build the downloadable files. Returns {'files': [(label, bytes, filename, mime)], 'warnings': [str]}.
    A failure in one format never blocks the others."""
    from report import to_docx, to_html

    meta = {"title": "Report", "author": "", "dataset": "", "date": pd.Timestamp.today().date().isoformat(), **meta}
    files, warnings = [], []
    try:
        files.append(("Download PDF", pdf_export.to_pdf(blocks, meta), f"{stem}.pdf", "application/pdf"))
    except ImportError:
        warnings.append("PDF export needs the 'reportlab' package. Add `reportlab` to requirements.txt and redeploy.")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"PDF export failed: {type(e).__name__}: {e}")
    try:
        files.append(("Download Word (.docx)", to_docx(blocks, meta), f"{stem}.docx",
                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Word export failed: {type(e).__name__}: {e}")
    try:
        files.append(("Download HTML", to_html(blocks, meta), f"{stem}.html", "text/html"))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"HTML export failed: {type(e).__name__}: {e}")
    return {"files": files, "warnings": warnings}


def _sql_blocks():
    msgs = st.session_state.get("sql_msgs", [])
    if not any(m["role"] == "assistant" and (m.get("sql") or m.get("explanation")) for m in msgs):
        return None
    from sql_chat import build_sql_blocks
    import datahub as hub

    return build_sql_blocks(msgs, list(hub.tables()))


def render_export(df_raw: pd.DataFrame, df: pd.DataFrame, label: str, dataset_name: str):
    st.subheader("Report Export (PDF and Word)")
    st.caption("Combine your work into one document and download it as PDF, Word or HTML. Each part appears here once "
               "you have created it on its own page.")

    main = st.session_state.get("report_blocks")
    dash = st.session_state.get("bi_blocks")
    sql = _sql_blocks()
    found = {"main": bool(main), "dash": bool(dash), "sql": bool(sql)}

    st.markdown("**What can be included**")
    st.markdown(f"- Main analysis report: {'ready' if main else 'not created yet (use the **Report** page and press Generate report)'}\n"
                f"- Dashboard snapshot: {'ready' if dash else 'not created yet (use **Dashboards** > Export > Build dashboard report)'}\n"
                f"- SQL Chat session: {'ready' if sql else 'not created yet (ask something on the **SQL Chat** page)'}")
    if not any(found.values()):
        st.info("Nothing to export yet. Create at least one of the items above.")
        return

    st.session_state.setdefault("expw_title", st.session_state.get("report_title") or "Analysis Report")
    st.session_state.setdefault("expw_author", st.session_state.get("report_author", ""))
    for key, ok in (("expw_main", found["main"]), ("expw_dash", found["dash"]), ("expw_sql", found["sql"])):
        st.session_state.setdefault(key, ok)
    a, b = st.columns(2)
    a.text_input("Report title", key="expw_title")
    b.text_input("Author", key="expw_author")
    c1, c2, c3 = st.columns(3)
    c1.checkbox("Main analysis report", key="expw_main", disabled=not found["main"])
    c2.checkbox("Dashboard snapshot", key="expw_dash", disabled=not found["dash"])
    c3.checkbox("SQL Chat session", key="expw_sql", disabled=not found["sql"])

    if st.button("Build combined report", type="primary"):
        blocks = []
        if found["main"] and st.session_state["expw_main"]:
            blocks += main
        if found["dash"] and st.session_state["expw_dash"]:
            blocks += dash
        if found["sql"] and st.session_state["expw_sql"]:
            blocks += sql
        if not blocks:
            st.error("Select at least one part to include.")
        else:
            meta = {"title": st.session_state["expw_title"] or "Analysis Report", "author": st.session_state["expw_author"],
                    "dataset": label, "date": pd.Timestamp.today().date().isoformat()}
            st.session_state["exp_files"] = make_files(blocks, meta, "analysis_report")
            st.session_state["exp_count"] = (sum(b[0] == "h1" for b in blocks), sum(b[0] == "image" for b in blocks),
                                             sum(b[0] == "table" for b in blocks))

    out = st.session_state.get("exp_files")
    if out:
        sections, images, tables = st.session_state["exp_count"]
        st.success(f"Report built: {sections} section(s), {tables} table(s), {images} chart(s).")
        cols = st.columns(len(out["files"]) or 1)
        for col, (lbl, data, name, mime) in zip(cols, out["files"]):
            col.download_button(lbl, data, name, mime, key=f"exp_dl_{name}")
        for w in out["warnings"]:
            st.warning(w)
