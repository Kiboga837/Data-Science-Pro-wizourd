"""Questions & Answers: upload questions about the data, get answers with the
full procedure (steps, Python code, libraries and why) and the real output.

How it works
1. The questions are read from a file (txt, md, csv, xlsx, docx) or typed.
2. The AI model (Google Gemini free tier by default, or Claude) receives the question plus a SUMMARY of the data
   (column names, types, a few sample rows, basic stats). It writes pandas code.
3. The code runs HERE, on the full dataset, inside a restricted environment.
4. The AI model writes a plain-language answer from the real output.
The full dataset is never sent to the AI provider, only the summary.
"""
import ast
import builtins
import io
import re
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

import llm
from common import analysis_numeric_cols, categorical_cols, id_like_columns
from llm import LLMError


ALLOWED_ROOTS = {"pandas", "numpy", "matplotlib", "seaborn", "scipy", "sklearn", "math", "statistics",
                 "datetime", "itertools", "collections", "re", "warnings", "functools", "string", "textwrap"}
BLOCKED_NAMES = {"os", "sys", "subprocess", "open", "exec", "eval", "compile", "__import__", "input", "globals",
                 "locals", "getattr", "setattr", "delattr", "vars", "breakpoint", "exit", "quit", "importlib",
                 "builtins", "memoryview"}
BLOCKED_ATTRS = {"read_csv", "read_excel", "read_pickle", "read_parquet", "read_sql", "read_html", "read_json",
                 "read_table", "read_clipboard", "read_feather", "read_fwf", "to_csv", "to_excel", "to_pickle",
                 "to_parquet", "to_sql", "to_json", "to_clipboard", "to_html", "to_feather", "savefig", "load",
                 "save", "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile", "system", "popen", "environ",
                 "eval", "io", "sys", "os", "subprocess", "builtins", "importlib", "urlopen", "urlretrieve"}
SAFE_BUILTINS = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "len", "list", "map",
                 "max", "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip", "isinstance",
                 "reversed", "divmod", "pow", "repr", "slice", "frozenset", "format", "iter", "next", "callable",
                 "hasattr", "type", "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
                 "ZeroDivisionError", "ArithmeticError", "StopIteration", "AssertionError", "NameError"]

SYSTEM_PROMPT = """You are a careful senior data analyst. You answer questions about a pandas DataFrame named `df`
by writing Python code that is executed for you.

Reply in EXACTLY this tag format and nothing else:

<approach>
1. First analytical step, in plain English
2. Next step ...
</approach>
<libraries>
pandas | why this library is needed for this question
scipy.stats | why ...
</libraries>
<code>
# python code only, no markdown fences
</code>

Code rules:
- `df` already exists (the active dataset). Do not load files. Work on `data = df.copy()`.
- Available without importing: pd, np, plt, sns, stats (scipy.stats). You may import pandas, numpy, matplotlib,
  seaborn, scipy, sklearn, math, statistics, datetime, itertools, collections, re.
- No file, network, OS access. Do not call plt.show() or save figures; just create them with matplotlib/seaborn.
- Use print() to show each intermediate result (counts, groups, test statistics), so the procedure is visible.
- Handle missing values explicitly and say so in a comment. Use the exact column names given.
- Store the final answer in a variable named `result` (a number, string, Series or DataFrame).
- Only list in <libraries> the libraries your code actually uses.
- If the question cannot be answered from the data, set result to a short string explaining why."""


# ----------------------------------------------------------------------------
# Reading questions
# ----------------------------------------------------------------------------
_NUMBERING = re.compile(r"^\s*(?:Q(?:uestion)?\s*\d+\s*[\.\):\-]?|\d+\s*[\.\):\-]|[-*\u2022])\s*", re.I)


def clean_question(line: str) -> str:
    return _NUMBERING.sub("", line).strip()


def parse_questions(name: str, raw: bytes) -> list:
    ext = name.lower().rsplit(".", 1)[-1]
    lines = []
    if ext in ("txt", "md"):
        lines = raw.decode("utf-8", errors="ignore").splitlines()
    elif ext in ("csv", "xlsx"):
        tbl = pd.read_csv(BytesIO(raw), header=None) if ext == "csv" else pd.read_excel(BytesIO(raw), header=None)
        col = tbl.iloc[:, 0].dropna().astype(str).tolist()
        if col and col[0].strip().lower() in ("question", "questions", "q", "query"):
            col = col[1:]
        lines = col
    elif ext == "docx":
        try:
            from docx import Document
        except ImportError as e:
            raise RuntimeError("Reading .docx needs `pip install python-docx`.") from e
        lines = [p.text for p in Document(BytesIO(raw)).paragraphs]
    else:
        raise ValueError(f"Unsupported question file type: .{ext}")
    return [q for q in (clean_question(x) for x in lines) if len(q) > 3]


def suggest_questions(df: pd.DataFrame, max_q: int = 8) -> list:
    """Sensible starter questions built from the dataset's own columns (works for any dataset)."""
    nums = analysis_numeric_cols(df)
    ids = set(id_like_columns(df))
    cats = [c for c in categorical_cols(df, 15) if c not in ids]
    groups = [c for c in cats if 2 <= df[c].nunique() <= 10]
    out = ["How many rows and columns does the dataset have, and what are the data types?"]
    if df.isna().any().any():
        out.append("Which columns have missing values, and what percentage of each column is missing?")
    if nums:
        out.append(f"What are the mean, median and standard deviation of {nums[0]}?")
    if cats and nums:
        out.append(f"What is the average {nums[0]} for each {cats[0]}, and which {cats[0]} has the highest?")
    if groups and nums:
        out.append(f"Is the difference in {nums[min(1, len(nums) - 1)]} between the groups of {groups[0]} "
                   "statistically significant?")
    if len(nums) >= 2:
        out.append("Which two numeric columns are most strongly correlated, and is the relationship significant?")
    if nums:
        out.append(f"Are there outliers in {nums[-1]}? Count them using the IQR rule.")
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dates and nums:
        out.append(f"How does the total of {nums[0]} change over time (by month or year)?")
    if cats:
        out.append(f"What is the distribution of {cats[-1]}, as counts and percentages?")
    seen, unique = set(), []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique[:max_q]


def quick_answer(display) -> str:
    """Answer used in economy mode, built from the computed result without a second AI request."""
    if display is None:
        body = "No `result` value was produced; see the printed output below."
    elif isinstance(display, pd.DataFrame):
        body = f"The result is a table with {len(display)} row(s); see below."
    else:
        body = f"Result: {display}"
    return body + " (Written explanation skipped to save quota.)"


def run_questions(client, model, df, label, questions, results, progress=None, economy=False):
    """Answer every question not yet answered, storing each result under (label, shape, question).
    Provider problems (bad key, quota, rate limit) raise LLMError and stop the batch; answers already
    stored are kept, so pressing the button again continues where it stopped."""
    context = build_context(df)
    for i, question in enumerate(questions):
        rkey = (label, df.shape, question)
        if rkey in results:
            continue
        if progress:
            progress(i / max(len(questions), 1), f"Answering {i + 1}/{len(questions)}: {question[:60]}")
        try:
            results[rkey] = answer_question(client, model, question, df, context, economy=economy)
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            results[rkey] = {"question": question, "error": f"Unexpected error: {e}", "attempts": 0}


# ----------------------------------------------------------------------------
# Context sent to the model (a summary, never the whole dataset)
# ----------------------------------------------------------------------------
def build_context(df: pd.DataFrame) -> str:
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append(f"- {c!r}: dtype={s.dtype}, missing={int(s.isna().sum())}, unique={s.nunique()}")
    desc = df.describe(include="all").T.head(40).to_string(max_colwidth=30) if len(df) else ""
    return (f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\nColumns:\n" + "\n".join(cols)
            + "\n\nFirst 5 rows:\n" + df.head(5).to_string(max_colwidth=30)
            + "\n\nSummary statistics:\n" + desc)


# ----------------------------------------------------------------------------
# Guarded execution of model-written code
# ----------------------------------------------------------------------------
def validate_code(code: str) -> ast.AST:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"SyntaxError: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_ROOTS:
                    raise ValueError(f"Import of '{a.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_ROOTS:
                raise ValueError(f"Import from '{node.module}' is not allowed.")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"Use of '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in BLOCKED_ATTRS:
                raise ValueError(f"Access to '.{node.attr}' is not allowed.")
    return tree


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_ROOTS:
        raise ImportError(f"Import of '{name}' is not allowed.")
    return builtins.__import__(name, globals, locals, fromlist, level)


def execute_code(code: str, df: pd.DataFrame) -> dict:
    """Run validated code against a copy of df. Returns stdout, result, images and error."""
    out = {"stdout": "", "result": None, "images": [], "error": None}
    try:
        tree = validate_code(code)
    except ValueError as e:
        out["error"] = str(e)
        return out

    buffer = io.StringIO()

    def _print(*args, **kwargs):
        kwargs.pop("file", None)
        print(*args, file=buffer, **kwargs)

    safe = {n: getattr(builtins, n) for n in SAFE_BUILTINS}
    safe.update({"print": _print, "__import__": _guarded_import})
    import seaborn as sns

    env = {"__builtins__": safe, "pd": pd, "np": np, "plt": plt, "sns": sns, "stats": stats, "df": df.copy()}
    plt.close("all")
    try:
        exec(compile(tree, "<question>", "exec"), env)
        out["result"] = env.get("result")
    except Exception as e:  # noqa: BLE001 - we report any failure back to the model
        out["error"] = f"{type(e).__name__}: {e}"
    out["stdout"] = buffer.getvalue()[:8000]
    for num in plt.get_fignums():
        fig = plt.figure(num)
        img = BytesIO()
        fig.savefig(img, format="png", dpi=110, bbox_inches="tight")
        out["images"].append(img.getvalue())
    plt.close("all")
    return out


def result_for_display(result):
    """Convert the `result` variable into something safe to store and show."""
    if result is None:
        return None
    if isinstance(result, pd.Series):
        return result.head(500).to_frame()
    if isinstance(result, pd.DataFrame):
        return result.head(500)
    if isinstance(result, (np.generic,)):
        return result.item()
    if isinstance(result, (int, float, str, bool)):
        return result
    return str(result)[:3000]


def result_as_text(display) -> str:
    if display is None:
        return "(no `result` variable was set)"
    if isinstance(display, pd.DataFrame):
        return display.head(40).to_string()
    return str(display)


# ----------------------------------------------------------------------------
# Talking to the model
# ----------------------------------------------------------------------------
def parse_reply(text: str) -> dict:
    def grab(tag):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
        return m.group(1).strip() if m else ""

    code = grab("code")
    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code).strip()
    if not code:
        raise ValueError("The reply did not contain a <code> block.")
    steps = [re.sub(r"^\s*\d+[\.\)]\s*", "", ln).strip() for ln in grab("approach").splitlines() if ln.strip()]
    libs = []
    for ln in grab("libraries").splitlines():
        if "|" in ln:
            name, why = ln.split("|", 1)
            libs.append((name.strip(" -*"), why.strip()))
    return {"approach": steps, "libraries": libs, "code": code}


def ask_model(client, model: str, system: str, messages: list, max_tokens: int = 3000) -> str:
    return client.generate(model, system, messages, max_tokens)


def answer_question(client, model: str, question: str, df: pd.DataFrame, context: str, max_attempts: int = 3,
                    economy: bool = False) -> dict:
    messages = [{"role": "user", "content": f"Dataset summary:\n{context}\n\nQuestion: {question}"}]
    parsed, run, attempts = None, None, 0
    for attempts in range(1, max_attempts + 1):
        reply = ask_model(client, model, SYSTEM_PROMPT, messages)
        try:
            parsed = parse_reply(reply)
        except ValueError as e:
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"{e} Please answer again using the exact tag format."}]
            continue
        run = execute_code(parsed["code"], df)
        if run["error"] is None:
            break
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"The code failed with:\n{run['error']}\n"
                                                 "Return the corrected full reply in the same tag format."}]
    if parsed is None or run is None:
        return {"question": question, "error": "The model did not return usable code.", "attempts": attempts}

    display = result_for_display(run["result"])
    answer = ""
    if run["error"] is None and economy:
        answer = quick_answer(display)
    elif run["error"] is None:
        evidence = f"Printed output:\n{run['stdout'][:5000]}\n\nFinal result:\n{result_as_text(display)[:4000]}"
        answer = ask_model(
            client, model,
            "You explain data analysis results. Using ONLY the numbers in the execution output, answer the "
            "question directly in 2 to 6 sentences, then list any assumptions or caveats (for example how missing "
            "values were treated). Never invent numbers.",
            [{"role": "user", "content": f"Question: {question}\n\nSteps taken:\n" + "\n".join(parsed["approach"])
              + f"\n\n{evidence}"}], max_tokens=900)
    return {"question": question, "approach": parsed["approach"], "libraries": parsed["libraries"],
            "code": parsed["code"], "stdout": run["stdout"], "result": display, "images": run["images"],
            "error": run["error"], "answer": answer, "attempts": attempts}


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def build_report(results: list, label: str) -> str:
    md = [f"# Question and Answer Report\n\nDataset used: {label}\n"]
    for i, r in enumerate(results, 1):
        md.append(f"\n## Q{i}. {r['question']}\n")
        if r.get("error"):
            md.append(f"**Could not be answered:** {r['error']}\n")
        md.append(f"**Answer**\n\n{r.get('answer', '')}\n")
        if r.get("approach"):
            md.append("**Procedure**\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(r["approach"], 1)) + "\n")
        if r.get("libraries"):
            md.append("**Libraries used and why**\n" + "\n".join(f"- `{n}`: {w}" for n, w in r["libraries"]) + "\n")
        if r.get("code"):
            md.append("**Python code**\n\n```python\n" + r["code"] + "\n```\n")
        if r.get("stdout"):
            md.append("**Output**\n\n```\n" + r["stdout"] + "\n```\n")
    return "\n".join(md)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def _show_result(r: dict):
    if r.get("error") and not r.get("code"):
        st.error(r["error"])
        return
    if r.get("error"):
        st.error(f"The generated code still failed after {r['attempts']} attempt(s): {r['error']}")
    elif r.get("answer"):
        st.markdown("#### Answer")
        st.markdown(r["answer"])
    if r.get("attempts", 1) > 1 and not r.get("error"):
        st.caption(f"The first code failed, so it was automatically corrected ({r['attempts']} attempts).")
    st.markdown("#### How the answer was reached")
    for n, step in enumerate(r["approach"], 1):
        st.markdown(f"{n}. {step}")
    st.markdown("**Libraries used and why**")
    if r["libraries"]:
        st.table(pd.DataFrame(r["libraries"], columns=["Library", "Why it is needed"]).set_index("Library"))
    st.markdown("**Python code that was executed**")
    st.code(r["code"], language="python")
    if r.get("stdout"):
        st.markdown("**Printed output**")
        st.code(r["stdout"], language="text")
    if r.get("result") is not None:
        st.markdown("**Final result**")
        if isinstance(r["result"], pd.DataFrame):
            st.dataframe(r["result"])
        else:
            st.write(r["result"])
    for img in r.get("images", []):
        st.image(img)


def _insert_suggestions(df: pd.DataFrame):
    existing = st.session_state.get("q_text_main", "")
    fresh = [s for s in suggest_questions(df) if s not in existing]
    st.session_state["q_text_main"] = (existing.rstrip("\n") + "\n" if existing.strip() else "") + "\n".join(fresh)


def render_qa(df: pd.DataFrame, label: str):
    st.subheader("Questions & Answers")
    st.caption("Upload questions about your data. Each one is answered by running real Python on your dataset, "
               "and the procedure, code and libraries are shown.")
    with st.expander("How this works and what is shared", expanded=False):
        st.markdown(
            "- An AI model writes the pandas code; **the code runs on this app's server** against your full data.\n"
            "- Only a **summary** (column names, types, five sample rows, basic statistics) and your question are "
            "sent to the AI provider. The full dataset is not sent.\n"
            "- Code runs in a restricted environment (no file, network or OS access), but review the code shown "
            "and verify important answers.\n"
            "- **Free option:** Google Gemini's free tier needs only a free key from Google AI Studio, no billing.")
    creds = llm.render_llm_settings()

    up = st.file_uploader("Upload questions (txt, md, csv, xlsx, docx)", type=["txt", "md", "csv", "xlsx", "docx"],
                          key="q_file")
    if up is not None and st.session_state.get("q_loaded_id") != f"{up.name}-{up.size}":
        try:
            st.session_state["q_text_main"] = "\n".join(parse_questions(up.name, up.getvalue()))
            st.session_state["q_loaded_id"] = f"{up.name}-{up.size}"
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the questions file: {e}")
    st.session_state.setdefault("q_text_main", "")
    text = st.text_area("Questions (one per line, edit freely; kept while you move between pages)", height=160,
                        key="q_text_main",
                        placeholder="\n".join(suggest_questions(df)[:2]))
    st.button("Insert suggested questions for this dataset", on_click=_insert_suggestions, args=(df,))
    questions = [clean_question(x) for x in text.splitlines() if len(clean_question(x)) > 3]
    st.caption(f"{len(questions)} question(s) ready.")

    results = st.session_state.setdefault("qa_results", {})
    ctx_key = label
    if st.button("Answer questions", type="primary", disabled=not questions):
        if not creds.key:
            st.error("Enter an API key first (a free Gemini key works).")
        else:
            status = st.empty()
            client = llm.make_client(creds, notify=lambda m: status.info(m))
            bar = st.progress(0.0, text="Starting...")
            try:
                run_questions(client, creds.model, df, ctx_key, questions, results,
                              lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t), creds.economy)
            except LLMError as e:
                done = sum(1 for qq in questions if (ctx_key, df.shape, qq) in results)
                st.error(f"Stopped after {done} of {len(questions)} answers: {e}")
            bar.empty()
            status.empty()

    shown = [results[(ctx_key, df.shape, q)] for q in questions if (ctx_key, df.shape, q) in results]
    if shown:
        st.markdown("---")
        for i, r in enumerate(shown, 1):
            with st.expander(f"Q{i}. {r['question']}", expanded=(i == 1)):
                _show_result(r)
        st.download_button("Download full Q&A report (Markdown)", build_report(shown, label).encode("utf-8"),
                           "qa_report.md", "text/markdown")
        if st.button("Clear answers"):
            results.clear()
            st.rerun()
