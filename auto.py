"""One-click full analysis: inspect, clean, analyse, model, answer questions and build the report.

Works on any tabular dataset: the target column, problem type and features are detected from the
data itself, and every step is isolated so one failure never stops the others.
"""
import re
import time

import pandas as pd
import streamlit as st

import cleaning
import llm
import ml
import qa
import report
from common import categorical_cols, id_like_columns, numeric_cols
from inspection import quality_issues
from stats_analysis import key_findings

# Words that usually mark an outcome column, most specific first.
TARGET_HINTS = ("target", "label", "outcome", "churn", "survived", "default", "diagnosis", "species", "class",
                "response", "status", "final", "result", "price", "sales", "revenue", "salary", "income",
                "score", "grade", "rating", "cost", "y")


# ----------------------------------------------------------------------------
# Detection helpers
# ----------------------------------------------------------------------------
def _usable_target(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return False
    n = s.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return n >= 2
    return 2 <= n <= 20


def suggest_target(df: pd.DataFrame):
    """Return (column or None, reason). Uses column names, then the 'last column' convention."""
    ids = set(id_like_columns(df))
    cols = [c for c in df.columns if c not in ids]
    cand = [c for c in cols if _usable_target(df[c])]
    if not cand:
        return None, "No column looks suitable as a prediction target."
    for hint in TARGET_HINTS:
        hits = [c for c in cand if hint in re.split(r"[^a-z0-9]+", str(c).lower())]
        if hits:
            return hits[-1], f"The name '{hits[-1]}' looks like an outcome ('{hint}')."
    if cols[-1] in cand:
        return cols[-1], "Using the last column, a common convention for the outcome."
    return cand[-1], "Using the last suitable column."


def infer_problem(s: pd.Series):
    """'Regression', 'Classification', or None when the target has too many text categories."""
    n = s.nunique(dropna=True)
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        return "Classification" if 2 <= n <= 20 else None
    if n == 2 or (pd.api.types.is_integer_dtype(s) and n <= 10):
        return "Classification"
    return "Regression"


def choose_features(df: pd.DataFrame, target: str, cap: int = 50) -> list:
    ids = set(id_like_columns(df))
    feats = [c for c in df.columns if c != target and c not in ids and df[c].nunique(dropna=True) > 1
             and df[c].isna().mean() < 0.6]
    return feats[:cap]


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run_full_analysis(df_raw: pd.DataFrame, opts: dict, progress=None) -> dict:
    """Run every step. Returns {'steps': [...], 'findings': [...], 'label': str, 'target': str, 'problem': str}."""
    steps, out = [], {"findings": [], "target": None, "problem": None}
    state = {"df": df_raw, "label": "Original data"}
    names = ["Inspect the data", "Clean the data", "Statistical analysis", "Train and compare models",
             "Answer questions", "Executive summary", "Build the report"]

    def step(i, fn):
        if progress:
            progress(i / len(names), names[i])
        start = time.time()
        try:
            status, detail = fn()
        except Exception as e:  # noqa: BLE001 - never let one step stop the rest
            status, detail = "Failed", f"{type(e).__name__}: {e}"
        steps.append({"Step": names[i], "Status": status, "Details": detail, "Seconds": round(time.time() - start, 1)})

    # 1. inspect ---------------------------------------------------------------
    def inspect():
        miss = int(df_raw.isna().sum().sum())
        cols = int(df_raw.isna().any().sum())
        issues = quality_issues(df_raw)
        return "Done", (f"{miss:,} missing cells in {cols} column(s); {len(issues)} data-quality finding(s); "
                        f"{int(df_raw.duplicated().sum())} duplicate row(s).")
    step(0, inspect)

    # 2. clean -----------------------------------------------------------------
    def clean():
        if not opts.get("clean", True):
            existing = st.session_state.get("df_clean")
            if existing is not None:  # respect cleaning the user already did by hand
                state["df"], state["label"] = existing, "Cleaned data"
                return "Done", "Automatic cleaning is off; using the cleaned data you prepared earlier."
            return "Skipped", "Cleaning switched off; the original data is analysed."
        df_clean, log, before = cleaning.auto_clean(df_raw)
        st.session_state["df_clean"], st.session_state["clean_log"] = df_clean, log
        st.session_state["clean_before"] = before
        state["df"], state["label"] = df_clean, "Cleaned data"
        return "Done", (f"{len(log)} action(s): rows {before['rows']:,} to {len(df_clean):,}, missing cells "
                        f"{before['missing']:,} to {int(df_clean.isna().sum().sum()):,}.")
    step(1, clean)
    df, label = state["df"], state["label"]

    # 3. statistics ------------------------------------------------------------
    def stats_step():
        out["findings"] = key_findings(df)
        return "Done", f"{len(out['findings'])} key finding(s). Full tables and charts are built in the report."
    step(2, stats_step)

    # 4. models ----------------------------------------------------------------
    def model_step():
        if not opts.get("model", True):
            return "Skipped", "Model training switched off."
        target = opts.get("target")
        reason = "Chosen by you."
        if not target or target == "Auto-detect":
            target, reason = suggest_target(df)
        if target is None or target not in df.columns:
            return "Skipped", reason
        problem = opts.get("problem")
        if not problem or problem == "Auto-detect":
            problem = infer_problem(df[target])
        if problem is None:
            return "Skipped", f"'{target}' has more than 20 text categories, so it is not a usable target."
        out["target"], out["problem"] = target, problem

        usable = int(df[target].notna().sum())
        if usable < 20:
            return "Skipped", f"Only {usable} rows have a value for '{target}'; at least 20 are needed to train and test models."
        if not choose_features(df, target):
            return "Skipped", f"No usable feature columns besides '{target}' (IDs, constants and mostly-empty columns are excluded)."
        if problem == "Classification" and df[target].nunique(dropna=True) < 2:
            return "Skipped", f"'{target}' has only one class."

        max_rows = int(opts.get("max_rows", 50000))
        df_model = df.sample(max_rows, random_state=42) if len(df) > max_rows else df
        big = len(df_model) > 20000
        y_raw = ml.make_target(df_model, target, problem)
        feats = choose_features(df_model, target)
        result, msgs = ml.train_compare(
            df_model, y_raw, target, problem, feats, label, n_estimators=50 if big else 100,
            cv_folds=3 if big else 5,
            progress=lambda f, t: progress and progress((3 + f) / len(names), t))
        if result is None:
            return "Failed", " ".join(t for _, t in msgs) or "Training failed."
        result["sig"] = (tuple(df.columns), df.shape, label)
        if len(df) > max_rows:
            result["note"] = f"Trained on a random sample of {max_rows:,} of {len(df):,} rows."
        st.session_state.pop("ml_tuned_params", None)
        st.session_state["ml_result"] = result
        best = result["metrics"]["CV score (mean)"].idxmax()
        extra = ""
        if opts.get("tune") and best in ml.PARAM_GRIDS:
            st.session_state["ml_tuned_params"] = ml.tune_and_add(result, best, 15)
            best = result["metrics"]["CV score (mean)"].idxmax()
            extra = " Best model was tuned."
        prim = ml.primary_metric(problem)
        score = result["metrics"].loc[best, "CV score (mean)"]
        warn = f" Notes: {' '.join(t for _, t in msgs)}" if msgs else ""
        return "Done", (f"{problem} for '{target}' ({reason}) with {len(feats)} feature(s). Best: {best}, "
                        f"cross-validated {prim} {score:.3f}.{extra}{warn}")
    step(3, model_step)

    # 5. questions -------------------------------------------------------------
    def question_step():
        text = st.session_state.get("q_text_main", "")
        questions = [qa.clean_question(x) for x in text.splitlines() if len(qa.clean_question(x)) > 3]
        if not opts.get("answer", True):
            return "Skipped", "Switched off."
        if not questions:
            return "Skipped", "No questions entered (add some on the Questions & Answers page)."
        creds = llm.get_credentials()
        if not creds.key:
            return "Skipped", "No AI key available (a free Gemini key works; enter it on the Questions & Answers page)."
        results = st.session_state.setdefault("qa_results", {})
        try:
            qa.run_questions(llm.make_client(creds), creds.model, df, label, questions, results, economy=creds.economy)
        except llm.LLMError as e:
            done = sum(1 for q in questions if "answer" in results.get((label, df.shape, q), {}))
            return "Partly done", f"{done} of {len(questions)} answered. Stopped: {e}"
        done = sum(1 for q in questions if "answer" in results.get((label, df.shape, q), {}))
        return "Done", f"{done} of {len(questions)} question(s) answered with code and procedure."
    step(4, question_step)

    # 6. executive summary -----------------------------------------------------
    def summary_step():
        if not opts.get("ai_summary"):
            return "Skipped", "The report still contains an automatic 'Key facts' summary."
        msg = report.draft_exec_summary(report.collect_facts(report._context(df_raw, df, label)))
        return ("Done" if msg.startswith("Draft") else "Skipped"), msg
    step(5, summary_step)

    # 7. report ----------------------------------------------------------------
    def report_step():
        warns = report.generate_report(df_raw, df, label)
        blocks = st.session_state["report_blocks"]
        failed = [b[1] for b in blocks if b[0] == "callout" and "could not be generated" in b[1]]
        n_tab, n_img = sum(b[0] == "table" for b in blocks), sum(b[0] == "image" for b in blocks)
        detail = f"{sum(b[0] == 'h1' for b in blocks)} section(s), {n_tab} table(s), {n_img} chart(s)."
        if failed or warns:
            detail += f" Problems: {len(failed)} section(s) failed. {' '.join(warns)}"
        return ("Done" if not failed else "Partly done"), detail
    step(6, report_step)

    if progress:
        progress(1.0, "Finished")
    out.update({"steps": steps, "label": label})
    return out


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def _goto(page: str):
    st.session_state["_goto"] = page


def _init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


def render_auto(df_raw: pd.DataFrame, dataset_name: str):
    st.subheader("Full Analysis (one click)")
    st.caption("Runs the whole workflow on your data: inspection, cleaning, statistics, model comparison, your "
               "questions and a finished report. Your original data is never modified, and every step can be "
               "revisited and refined afterwards.")

    cols = list(df_raw.columns)
    target, why = suggest_target(df_raw)
    problem = infer_problem(df_raw[target]) if target else None
    if target:
        st.info(f"Detected prediction target: **{target}** ({problem or 'not usable'}). {why} You can override this below.")
    else:
        st.info(f"{why} Modelling will be skipped unless you choose a target below.")

    _init("auto_clean", True)
    _init("auto_model", True)
    _init("auto_target", "Auto-detect")
    _init("auto_problem", "Auto-detect")
    _init("auto_tune", False)
    _init("auto_answer", True)
    _init("auto_ai_summary", False)
    _init("auto_max_rows", 50000)
    if st.session_state["auto_target"] not in ["Auto-detect"] + cols:
        st.session_state["auto_target"] = "Auto-detect"

    a, b = st.columns(2)
    a.checkbox("Clean the data using the recommended decisions (off = use your own earlier cleaning, if any)", key="auto_clean")
    a.checkbox("Train and compare predictive models", key="auto_model")
    a.checkbox("Tune the best model (slower)", key="auto_tune")
    has_key = bool(llm.get_credentials().key)
    b.checkbox("Answer my questions with AI" + ("" if has_key else " (needs a free Gemini key)"), key="auto_answer")
    b.checkbox("Draft the executive summary with AI" + ("" if has_key else " (needs a free Gemini key)"), key="auto_ai_summary")
    c, d = st.columns(2)
    c.selectbox("Prediction target", ["Auto-detect"] + cols, key="auto_target")
    d.selectbox("Problem type", ["Auto-detect", "Regression", "Classification"], key="auto_problem")
    with st.expander("Advanced"):
        st.number_input("Maximum rows used to train models (larger data is sampled)", 1000, 1000000, key="auto_max_rows")
        st.caption("Questions come from the Questions & Answers page; use its 'Insert suggested questions' button "
                   "to start from sensible ones for this dataset.")

    if st.button("Run full analysis", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        opts = {"clean": st.session_state["auto_clean"], "model": st.session_state["auto_model"],
                "tune": st.session_state["auto_tune"], "answer": st.session_state["auto_answer"],
                "ai_summary": st.session_state["auto_ai_summary"], "target": st.session_state["auto_target"],
                "problem": st.session_state["auto_problem"], "max_rows": st.session_state["auto_max_rows"]}
        result = run_full_analysis(df_raw, opts, lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
        bar.empty()
        st.session_state["auto_run"] = result
        st.session_state["auto_run"]["dataset"] = dataset_name
        if "df_clean" in st.session_state:  # make the sidebar match the data the results were built on
            st.session_state["_data_source"] = result["label"]

    run = st.session_state.get("auto_run")
    if run:
        st.markdown("---")
        st.markdown("#### Results")
        st.dataframe(pd.DataFrame(run["steps"]).set_index("Step"))
        if run["findings"]:
            st.markdown("**Key findings**")
            st.markdown("\n".join(f"- {f}".replace("$", "\\$") for f in run["findings"]))
        if st.session_state.get("report_html"):
            n = st.session_state.get("report_name", "report")
            e, f = st.columns(2)
            e.download_button("Download report (HTML)", st.session_state["report_html"], f"{n}.html", "text/html")
            if st.session_state.get("report_docx"):
                f.download_button("Download report (Word)", st.session_state["report_docx"], f"{n}.docx",
                                  "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        st.markdown("**Continue**")
        g = st.columns(4)
        g[0].button("Open the report", on_click=_goto, args=("Report",), type="primary")
        g[1].button("Review model results", on_click=_goto, args=("Predictive Model",))
        g[2].button("Review cleaning", on_click=_goto, args=("Data Cleaning",))
        g[3].button("Statistical analysis", on_click=_goto, args=("Statistical Analysis",))
