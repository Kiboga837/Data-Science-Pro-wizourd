# Smart Data Science Studio Pro: SQL Chat, Dashboards, PDF/Word export

These features were added **without editing any existing file**. Your original app is untouched.
`app.py` still runs exactly as before. The new capabilities live in a second entry file, `pro_app.py`,
which includes every original page plus three new ones.

## Run
    streamlit run pro_app.py          # new: everything, including the pages below
    streamlit run app.py              # original, unchanged

On Streamlit Community Cloud, set the app's **Main file path** to `pro_app.py`
(or deploy a second app from the same repo and keep the original on `app.py`).

## Two optional libraries
Add the lines in `requirements_pro_addon.txt` to your `requirements.txt`:
`plotly` (interactive charts) and `reportlab` (PDF export). Without them the app still works:
charts fall back to static images and PDF export shows an install hint (Word and HTML still download).

## New pages
### SQL Chat
Chat in plain language ("top 5 regions by revenue with their share of the total"). The app writes the SQL, runs it
and shows the table, a chart and the query. If a query fails, the error goes back to the AI and it corrects itself
(up to 2 retries). Follow-ups work ("now only for 2023") because earlier queries are sent as context.
- Uses the same AI settings as Questions & Answers: free Google Gemini key by default, Claude optional.
- **Views:** Chat, SQL editor (write your own), Cookbook, Tables and schema.
- **Cookbook:** ~15 advanced techniques filled in with *your* tables and columns: top N per group (ROW_NUMBER),
  running totals, moving averages, month-over-month growth (LAG), share of total, RANK, NTILE quartiles, pivots with
  CASE, outliers with CTEs, duplicates, missing values per column, correlation, joins. Run as is or edit.
- **Tables:** your main dataset (original and cleaned) are tables automatically. Add more with CSV, Excel (each sheet
  becomes a table) or a SQLite database file, then JOIN them. Save any result as a new table to query it again or chart it.
- **Extra SQL functions:** STDEV, VARIANCE, MEDIAN, PERCENTILE(x, p), CORR(x, y), SQRT, POWER, LN, LOG10, EXP, FLOOR,
  CEIL and the REGEXP operator, on top of SQLite's window functions and CTEs (including recursive).
- **Safety:** read-only. SQLite itself refuses INSERT/UPDATE/DELETE/DROP/CREATE/ATTACH/PRAGMA (enforced by an
  authorizer, not text matching), queries stop after 20 s, results are capped at 100,000 rows. It runs on an in-memory
  copy, so your files are never touched. Typos in column names are errors (SQLite's "silent text value" quirk is disabled).
- Only the table/column names, types, a few sample values and your request go to the AI, never the full data.

### Dashboards (Power BI style)
Pick any table, then change the **controls** (measure, aggregation, dimension, second dimension, date field, time
granularity, top N, colour theme) and the **slicers** in the left filter pane (categories, numeric ranges, date
ranges). Every KPI card, chart and automatic insight updates.
- Tabs: **Overview** (KPI cards, insights, 8 visuals that follow the controls), **Explore** (pivot table, drill-down),
  **Custom visuals** (build your own, or describe one in words and the AI creates it), **Data**, **Export**.
- 11 visual types: bar, horizontal bar, line, area, donut, scatter, histogram, box, heatmap, treemap, correlation (+ table).
- Export: PDF, Word, HTML report (static charts) and an interactive HTML dashboard (needs plotly).

### Report Export (PDF / Word)
Combines the main analysis report, the dashboard snapshot and the SQL session into one document: title page,
contents, tables with repeating headers, charts, SQL code, page numbers. Download as **PDF**, **Word (.docx)** or HTML.
Each part is created on its own page first (Report > Generate; Dashboards > Export; SQL Chat), then combined here.
PDF uses Unicode fonts (Greek letters, math symbols); Chinese/Japanese/emoji are not supported in the PDF.

## Files added (nothing existing was modified)
pro_app.py (entry point) | datahub.py (shared tables) | sql_engine.py | sql_chat.py | bi_core.py (dashboard logic) |
bi_page.py (dashboard UI) | pdf_export.py | export_page.py | requirements_pro_addon.txt |
tests/pro_fake.py, test_pro_sql.py, test_pro_bi.py, test_pro_export.py, test_pro_app.py

## Tests (no Streamlit needed)
    python tests/test_pro_sql.py       # SQL engine, AI->SQL flow, chat page
    python tests/test_pro_bi.py        # dashboard logic, every visual, slicers, AI visuals, plotly paths
    python tests/test_pro_export.py    # PDF/Word/HTML export
    python tests/test_pro_app.py       # runs app.py and pro_app.py end to end through every page
