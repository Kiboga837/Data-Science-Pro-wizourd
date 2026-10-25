"""Read-only SQL engine for SQL Chat (SQLite, in memory).

Safety model
- Tables are loaded into a private in-memory database, so the user's real files are never touched.
- After loading, a SQLite *authorizer* is installed that only permits SELECT-style operations. INSERT, UPDATE,
  DELETE, DROP, CREATE, ATTACH, PRAGMA and friends are refused by SQLite itself, not by text matching.
- A progress handler stops any query that runs longer than a time limit.

Advanced SQL available: joins, subqueries, CTEs (including recursive), window functions, CASE, set operations,
plus extra functions SQLite lacks: STDEV, VARIANCE, MEDIAN, PERCENTILE(x, p), CORR(x, y), SQRT, POWER, LN, LOG10,
EXP, FLOOR, CEIL and the REGEXP operator.
"""
import math
import os
import re
import sqlite3
import tempfile
import time

import numpy as np
import pandas as pd

from common import id_like_columns, is_text

MAX_ROWS = 100_000
TIME_LIMIT = 20.0
_ALLOWED_ACTIONS = {21, 20, 31, 33}  # SQLITE_SELECT, SQLITE_READ, SQLITE_FUNCTION, SQLITE_RECURSIVE


class SQLRunError(Exception):
    """A query could not be run; the message is written for end users."""


def qi(name) -> str:
    """Quote an identifier for SQLite."""
    return '"' + str(name).replace('"', '""') + '"'


# ----------------------------------------------------------------------------
# Extra aggregates and functions
# ----------------------------------------------------------------------------
class Stdev:
    """Sample standard deviation (Welford's streaming algorithm)."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def step(self, value):
        try:
            x = float(value)
        except (TypeError, ValueError):
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def finalize(self):
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else None


class Variance(Stdev):
    def finalize(self):
        return self.m2 / (self.n - 1) if self.n > 1 else None


class Median:
    def __init__(self):
        self.values = []

    def step(self, value):
        try:
            self.values.append(float(value))
        except (TypeError, ValueError):
            pass

    def finalize(self):
        return float(np.median(self.values)) if self.values else None


class Percentile:
    """PERCENTILE(x, p) with p between 0 and 100."""

    def __init__(self):
        self.values, self.p = [], 50.0

    def step(self, value, p):
        try:
            self.values.append(float(value))
            self.p = float(p)
        except (TypeError, ValueError):
            pass

    def finalize(self):
        return float(np.percentile(self.values, min(max(self.p, 0.0), 100.0))) if self.values else None


class Corr:
    """Pearson correlation CORR(x, y)."""

    def __init__(self):
        self.x, self.y = [], []

    def step(self, a, b):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return
        self.x.append(fa)
        self.y.append(fb)

    def finalize(self):
        if len(self.x) < 3:
            return None
        x, y = np.array(self.x), np.array(self.y)
        if x.std() == 0 or y.std() == 0:
            return None
        return float(np.corrcoef(x, y)[0, 1])


def _num(fn):
    def wrapper(value):
        try:
            return fn(float(value))
        except (TypeError, ValueError, OverflowError):
            return None
    return wrapper


def _register(conn: sqlite3.Connection):
    conn.create_aggregate("stdev", 1, Stdev)
    conn.create_aggregate("variance", 1, Variance)
    conn.create_aggregate("median", 1, Median)
    conn.create_aggregate("percentile", 2, Percentile)
    conn.create_aggregate("corr", 2, Corr)
    for name, fn in {"sqrt": math.sqrt, "ln": math.log, "log10": math.log10, "exp": math.exp,
                     "floor": math.floor, "ceil": math.ceil}.items():
        conn.create_function(name, 1, _num(fn), deterministic=True)
    conn.create_function("power", 2, lambda a, b: _pow(a, b), deterministic=True)
    conn.create_function("regexp", 2, lambda pattern, text: 1 if text is not None and re.search(pattern, str(text)) else 0,
                         deterministic=True)


def _pow(a, b):
    try:
        return math.pow(float(a), float(b))
    except (TypeError, ValueError, OverflowError):
        return None


def _disable_dqs(conn: sqlite3.Connection) -> bool:
    """SQLite quirk: "Nope" with no such column silently becomes the TEXT 'Nope' on every row, so a typo returns
    wrong results instead of an error. Turn that off (Python 3.12+). Returns False when it is not possible."""
    try:
        conn.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
        conn.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL, False)
        return True
    except (AttributeError, sqlite3.Error):
        return False


def _check_quoted(sql: str, known: set):
    """Fallback for Python < 3.12: every "double-quoted" name must be a known table/column or an alias defined in
    the query; otherwise it would silently be treated as a text value."""
    body = re.sub(r"'(?:[^']|'')*'", "''", sql)
    defined = set(re.findall(r'\bas\s+"((?:[^"]|"")+)"', body, re.I))
    defined |= set(re.findall(r'"((?:[^"]|"")+)"\s+as\s*\(', body, re.I))
    defined |= set(re.findall(r'\bwith\s+(?:recursive\s+)?"((?:[^"]|"")+)"', body, re.I))
    defined |= set(re.findall(r'\bas\s+([A-Za-z_]\w*)', body, re.I))                                  # unquoted aliases
    defined |= set(re.findall(r'\b(?:from|join)\s+(?:"[^"]+"|\w+)\s+(?:as\s+)?([A-Za-z_]\w*)', body, re.I))  # table aliases
    for token in re.findall(r'"((?:[^"]|"")+)"', body):
        name = token.replace('""', '"')
        if name not in known and name not in defined:
            raise SQLRunError(f'no such column or table: "{name}". Text values need single quotes (\'value\'); '
                              "double quotes are only for column and table names.")


def _authorizer(action, arg1, arg2, dbname, source):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


# ----------------------------------------------------------------------------
# Schema description (used for the AI prompt, the schema browser and the cookbook)
# ----------------------------------------------------------------------------
def describe_table(name: str, df: pd.DataFrame) -> dict:
    ids = set(id_like_columns(df))
    cols = []
    for c in df.columns:
        s = df[c]
        entry = {"name": c, "nulls": int(s.isna().sum()), "is_id": c in ids, "values": []}
        if pd.api.types.is_datetime64_any_dtype(s):
            lo, hi = s.min(), s.max()
            entry.update(kind="date", type="TEXT",
                         info=f"date/time stored as ISO text; range {lo:%Y-%m-%d} to {hi:%Y-%m-%d}" if pd.notna(lo) else "date/time")
        elif pd.api.types.is_bool_dtype(s):
            entry.update(kind="bool", type="INTEGER", info="0/1 flag")
        elif pd.api.types.is_numeric_dtype(s):
            integer = pd.api.types.is_integer_dtype(s)
            entry.update(kind="num", type="INTEGER" if integer else "REAL",
                         info=f"range {s.min():.6g} to {s.max():.6g}" if s.notna().any() else "all null")
        else:
            nun = s.nunique(dropna=True)
            vals = [str(v) for v in s.dropna().value_counts().head(8).index] if nun <= 30 else []
            sample = [str(v)[:25] for v in s.dropna().head(3)]
            entry.update(kind="cat" if nun <= 30 else "text", type="TEXT", values=vals,
                         info=("values: " + ", ".join(repr(v) for v in vals)) if vals else
                         f"{nun} distinct; e.g. " + ", ".join(repr(v) for v in sample))
        cols.append(entry)
    return {"name": name, "rows": len(df), "columns": cols}


def schema_text(info: dict, max_tables: int = 15, max_cols: int = 60) -> str:
    lines = []
    for name in list(info)[:max_tables]:
        t = info[name]
        lines.append(f'Table {qi(name)} ({t["rows"]:,} rows)')
        for c in t["columns"][:max_cols]:
            extra = f'; {c["nulls"]} nulls' if c["nulls"] else ""
            lines.append(f'  {qi(c["name"])} {c["type"]}  -- {c["info"]}{extra}')
        if len(t["columns"]) > max_cols:
            lines.append(f"  ... {len(t['columns']) - max_cols} more columns")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------------
def _dedupe(names):
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def clean_sql(sql: str) -> str:
    sql = (sql or "").strip()
    while sql.endswith(";"):
        sql = sql[:-1].rstrip()
    if not sql:
        raise SQLRunError("The query is empty.")
    return sql


def _first_keyword(sql: str) -> str:
    text = re.sub(r"/\*.*?\*/|--[^\n]*", " ", sql, flags=re.S).strip().lower()
    m = re.match(r"[a-z_]+", text)
    return m.group(0) if m else ""


def _friendly(err: Exception, limit: float) -> str:
    msg = str(err)
    low = msg.lower()
    if "not authorized" in low or "authoriz" in low:
        return "Only read-only queries (SELECT, or WITH ... SELECT) are allowed. Nothing was changed."
    if "interrupt" in low:
        return f"The query ran for more than {limit:.0f} seconds and was stopped. Add filters, a LIMIT, or simplify the joins."
    if "only execute one statement" in low:
        return "Run one statement at a time."
    if "no such table" in low:
        return msg + ". Check the table names in the 'Tables and schema' view."
    if "no such column" in low:
        return (msg + ". Column names are case-sensitive; quote names containing spaces with double quotes. "
                "Text values need single quotes ('value').")
    return msg


class SQLSession:
    def __init__(self, tables: dict):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.strict_names = _disable_dqs(self.conn)
        self.info = {}
        for name, df in tables.items():
            df.to_sql(name, self.conn, index=False, if_exists="replace")
            self.info[name] = describe_table(name, df)
        _register(self.conn)
        self.conn.set_authorizer(_authorizer)  # from here on the database is read-only

    def schema_text(self) -> str:
        return schema_text(self.info)

    def run(self, sql: str, max_rows: int = MAX_ROWS, time_limit: float = TIME_LIMIT):
        """Run one read-only query. Returns (DataFrame, info dict)."""
        sql = clean_sql(sql)
        if _first_keyword(sql) not in ("select", "with", "values"):
            raise SQLRunError("Only read-only queries (SELECT, or WITH ... SELECT) are allowed. Nothing was changed.")
        if not self.strict_names:
            _check_quoted(sql, {n for n in self.info} | {c["name"] for t in self.info.values() for c in t["columns"]})
        start = time.time()
        self.conn.set_progress_handler(lambda: 1 if time.time() - start > time_limit else 0, 20000)
        try:
            cur = self.conn.execute(sql)
            names = _dedupe([d[0] for d in (cur.description or [])])
            rows = cur.fetchmany(max_rows + 1)
        except sqlite3.Error as e:
            raise SQLRunError(_friendly(e, time_limit)) from e
        finally:
            self.conn.set_progress_handler(None, 0)
        truncated = len(rows) > max_rows
        df = pd.DataFrame(rows[:max_rows], columns=names)
        return df, {"seconds": round(time.time() - start, 3), "truncated": truncated, "rows": len(df)}


def read_sqlite_file(blob: bytes, max_tables: int = 50, max_rows: int = 1_000_000) -> dict:
    """Read every table/view of an uploaded SQLite database file (read-only) into DataFrames."""
    if not blob.startswith(b"SQLite format 3"):
        raise ValueError("This is not a SQLite database file.")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        f.write(blob)
        path = f.name
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'")]
        return {n: pd.read_sql_query(f"SELECT * FROM {qi(n)} LIMIT {int(max_rows)}", conn) for n in names[:max_tables]}
    finally:
        if conn is not None:
            conn.close()
        os.unlink(path)


# ----------------------------------------------------------------------------
# Cookbook of advanced techniques, filled in with the user's own tables and columns
# ----------------------------------------------------------------------------
def cookbook(info: dict) -> list:
    """Return [{'title','tip','sql'}...] built from the schema. Every template is runnable as is."""
    if not info:
        return []
    best = None
    for name, t in info.items():
        cols = t["columns"]
        nums = [c for c in cols if c["kind"] == "num" and not c["is_id"]]
        cats = [c for c in cols if c["kind"] == "cat" and not c["is_id"] and 2 <= len(c["values"]) <= 30]
        dates = [c for c in cols if c["kind"] == "date"]
        score = 4 * bool(nums) + 2 * bool(cats) + bool(dates) + min(t["rows"], 10 ** 6) / 10 ** 9
        if best is None or score > best[0]:
            best = (score, name, cols, nums, cats, dates)
    _, tname, cols, nums, cats, dates = best
    T = qi(tname)
    out = []

    def add(title, tip, sql):
        out.append({"title": title, "tip": tip, "sql": sql.strip()})

    add("Preview the first rows", "The quickest way to see what a table contains.", f"SELECT *\nFROM {T}\nLIMIT 20")
    add("Row count of every table", "UNION ALL stacks results from different queries.",
        "\nUNION ALL\n".join(f"SELECT '{n.replace(chr(39), chr(39) * 2)}' AS table_name, COUNT(*) AS row_count FROM {qi(n)}" for n in list(info)[:12]))
    add("Missing values per column", "CASE inside SUM counts rows meeting a condition.",
        "\nUNION ALL\n".join(
            f"SELECT '{c['name'].replace(chr(39), chr(39) * 2)}' AS column_name, "
            f"SUM(CASE WHEN {qi(c['name'])} IS NULL THEN 1 ELSE 0 END) AS missing, COUNT(*) AS total FROM {T}"
            for c in cols[:15]))
    key_cols = ", ".join(qi(c["name"]) for c in cols[:3])
    add("Find duplicate records", "GROUP BY + HAVING finds repeated combinations.",
        f"SELECT {key_cols}, COUNT(*) AS copies\nFROM {T}\nGROUP BY {key_cols}\nHAVING COUNT(*) > 1\nORDER BY copies DESC")

    if nums and cats:
        m, c = qi(nums[0]["name"]), qi(cats[0]["name"])
        add("Top 3 rows per group (ROW_NUMBER)", "A window function with PARTITION BY ranks inside each group.",
            f"WITH ranked AS (\n  SELECT *, ROW_NUMBER() OVER (PARTITION BY {c} ORDER BY {m} DESC) AS rn\n  FROM {T}\n)\nSELECT *\nFROM ranked\nWHERE rn <= 3\nORDER BY {c}, rn")
        add("Share of total by group", "SUM(SUM(x)) OVER () divides each group by the grand total.",
            f"SELECT {c} AS grp, SUM({m}) AS total,\n       ROUND(100.0 * SUM({m}) / SUM(SUM({m})) OVER (), 2) AS pct_of_total\nFROM {T}\nGROUP BY {c}\nORDER BY total DESC")
        add("Rank groups with ties (RANK)", "RANK leaves gaps after ties; DENSE_RANK does not.",
            f"SELECT {c} AS grp, SUM({m}) AS total,\n       RANK() OVER (ORDER BY SUM({m}) DESC) AS rnk\nFROM {T}\nGROUP BY {c}")
        add("Median and 90th percentile by group", "STDEV, MEDIAN and PERCENTILE are extra functions added by this app.",
            f"SELECT {c} AS grp, COUNT(*) AS n, ROUND(AVG({m}), 2) AS mean,\n       MEDIAN({m}) AS median, PERCENTILE({m}, 90) AS p90, ROUND(STDEV({m}), 2) AS sd\nFROM {T}\nGROUP BY {c}")
        vals = cats[0]["values"][:4]
        pivots = ",\n       ".join(f"SUM(CASE WHEN {c} = '{v.replace(chr(39), chr(39) * 2)}' THEN {m} END) AS {qi(v)}" for v in vals)
        lead = qi(dates[0]["name"]) if dates else None
        group = f"strftime('%Y', {lead})" if lead else None
        if group:
            add("Pivot with CASE (years as rows)", "SQLite has no PIVOT; conditional aggregation does the same.",
                f"SELECT {group} AS year,\n       {pivots}\nFROM {T}\nGROUP BY year\nORDER BY year")
        else:
            add("Pivot with CASE", "SQLite has no PIVOT; conditional aggregation does the same.",
                f"SELECT 'all rows' AS scope,\n       {pivots}\nFROM {T}")
    if nums:
        m = qi(nums[0]["name"])
        add("Outliers beyond 3 standard deviations (CTE)", "A CTE computes statistics once; the main query uses them.",
            f"WITH stats AS (\n  SELECT AVG({m}) AS mu, STDEV({m}) AS sd FROM {T}\n)\nSELECT t.*\nFROM {T} t, stats\nWHERE ABS(t.{m} - stats.mu) > 3 * stats.sd")
        add("Quartiles with NTILE", "NTILE(4) splits ordered rows into four equal buckets.",
            f"SELECT quartile, COUNT(*) AS n, MIN({m}) AS lo, MAX({m}) AS hi\nFROM (SELECT {m}, NTILE(4) OVER (ORDER BY {m}) AS quartile FROM {T} WHERE {m} IS NOT NULL)\nGROUP BY quartile")
    if len(nums) >= 2:
        a, b = qi(nums[0]["name"]), qi(nums[1]["name"])
        add("Correlation between two numeric columns", "CORR is an extra aggregate added by this app.",
            f"SELECT ROUND(CORR({a}, {b}), 4) AS correlation, COUNT(*) AS n\nFROM {T}")
    if nums and dates:
        m, d = qi(nums[0]["name"]), qi(dates[0]["name"])
        monthly = f"SELECT strftime('%Y-%m', {d}) AS month, SUM({m}) AS total\n  FROM {T}\n  WHERE {d} IS NOT NULL\n  GROUP BY month"
        add("Running total by month", "SUM() OVER (ORDER BY ...) accumulates as it goes.",
            f"WITH monthly AS (\n  {monthly}\n)\nSELECT month, total, SUM(total) OVER (ORDER BY month) AS running_total\nFROM monthly\nORDER BY month")
        add("3-month moving average", "ROWS BETWEEN defines the sliding window.",
            f"WITH monthly AS (\n  {monthly}\n)\nSELECT month, total,\n       ROUND(AVG(total) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS moving_avg_3m\nFROM monthly\nORDER BY month")
        add("Month-over-month growth (LAG)", "LAG looks at the previous row of the ordered window.",
            f"WITH monthly AS (\n  {monthly}\n)\nSELECT month, total,\n       LAG(total) OVER (ORDER BY month) AS previous_month,\n       ROUND(100.0 * (total - LAG(total) OVER (ORDER BY month)) / LAG(total) OVER (ORDER BY month), 2) AS growth_pct\nFROM monthly\nORDER BY month")
    if len(info) >= 2:
        names = list(info)
        for i, a in enumerate(names):
            hit = None
            for b in names[i + 1:]:
                shared = [c["name"] for c in info[a]["columns"] if c["name"] in {x["name"] for x in info[b]["columns"]} and not c["name"].startswith("Unnamed")]
                if shared:
                    hit = (a, b, shared[0])
                    break
            if hit:
                a, b, k = hit
                add(f"Join {a} with {b}", "A JOIN combines rows of two tables on a shared column.",
                    f"SELECT x.*, y.*\nFROM {qi(a)} x\nJOIN {qi(b)} y ON x.{qi(k)} = y.{qi(k)}\nLIMIT 100")
                break
    return out
