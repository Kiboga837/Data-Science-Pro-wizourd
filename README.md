# Smart Data Science Studio

Works on **any tabular dataset** (CSV or Excel, any domain). Nothing is hard-coded to a particular column:
the prediction target, problem type, features, ID columns and date columns are detected from the data.

## Start here: Full Analysis (one click)
Upload a file, open **Full Analysis (one click)**, press **Run full analysis**. It will:
1. inspect the data (missing values, duplicates, quality issues)
2. clean it with the recommended decisions (original data is never modified)
3. compute the statistical findings
4. detect the target and problem type, then train, cross-validate and compare 11-12 models (optionally tune the best)
5. answer your questions (Questions & Answers page list; needs a free Gemini key)
6. build the comprehensive report (HTML + Word)

Every step is isolated: if one fails or does not apply, the others still run, and the results table says why.
Then refine any step on its own page.

## Step by step
Data Inspection -> Data Cleaning -> Exploratory Analysis -> Statistical Analysis -> Time Trends ->
Questions & Answers -> Predictive Model -> Report.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Keeping your analysis
Report page -> **Save your analysis** -> `.zip` project; reload from the sidebar. Format is ZIP + JSON + Parquet
(no pickle, so loading is safe). Fitted model objects are not stored; comparison tables and importances are.

## Questions & Answers with a FREE Gemini key
The AI features (answering your questions with visible code, and the optional executive summary) use
**Google Gemini's free tier by default**. No billing or credit card.
1. Open https://aistudio.google.com/app/apikey, sign in with a Google account, click **Create API key**.
2. In the app (Questions & Answers page) paste the key, press **Test connection**. Default model: `gemini-2.5-flash`.
   Models change over time: press **List available models** to see what your key can use.
3. Optional: save the key so you do not paste it each session. Locally create `.streamlit/secrets.toml`:
       GEMINI_API_KEY = "your-key"
   On Streamlit Community Cloud: App settings -> Secrets, same line. (An environment variable `GEMINI_API_KEY` works too.)

Free-tier tips (Google does not publish exact limits in its docs; they are in AI Studio -> Rate limits):
- The app paces requests (default 4 s apart) and waits and retries when Google says "slow down".
- If you hit a **daily** limit, answered questions are kept; continue tomorrow or try `gemini-2.5-flash-lite`.
- Turn on **Save quota** to use 1 request per question instead of 2 (skips the written explanation; you still get the
  computed result, code and procedure).
- Privacy: Google's pricing page says free-tier requests may be used to improve its products. Only a data summary
  (columns, types, 5 sample rows, basic stats) and your question are sent, never the full dataset, but avoid
  confidential data on the free tier.

Prefer Claude? Choose "Anthropic Claude (paid)" as the provider and `pip install anthropic`.

## Built-in samples (sidebar)
Sales (messy), Customer churn (classification), Student performance (regression): each has missing values,
placeholders, stray spaces and numbers stored as text, to try every feature.

## Tests (no Streamlit needed)
    python tests/test_generic.py   # one-click analysis + every page on 19 different datasets
    python tests/test_report.py    # report, Word/HTML export, project save/restore
    python tests/test_logic.py     # cleaning, statistics, models, guarded Q&A execution
    python tests/test_llm.py       # Gemini client: retries, quotas, errors (simulated, no network)
    python tests/test_ui.py        # every page's render code

## Files
app.py (routing, state) | auto.py (one-click pipeline) | inspection.py | cleaning.py | stats_analysis.py | eda.py |
qa.py | llm.py (Gemini/Claude client) | ml.py | report.py | project.py | common.py (helpers, ID detection, samples)
