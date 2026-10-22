"""Comprehensive statistical analysis: descriptive, normality, correlation,
categorical association, hypothesis tests and outlier detection."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression

from common import analysis_numeric_cols, categorical_cols, id_like_columns, numeric_cols, show_method


# ----------------------------------------------------------------------------
# Computation (these functions are also shown to the user as "the code used")
# ----------------------------------------------------------------------------
def descriptive_table(df, cols, ci=0.95):
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        n = len(s)
        if n == 0:
            continue
        mean, sd = s.mean(), s.std()
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        tcrit = stats.t.ppf((1 + ci) / 2, n - 1) if n > 1 else np.nan
        mode = s.mode()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append({
            "column": c, "count": n, "missing": len(df) - n, "mean": mean, "median": s.median(),
            "mode": mode.iloc[0] if len(mode) else np.nan, "std": sd, "variance": s.var(), "SEM": sem,
            "min": s.min(), "Q1": q1, "Q3": q3, "max": s.max(), "range": s.max() - s.min(), "IQR": q3 - q1,
            "skewness": s.skew(), "kurtosis (excess)": s.kurt(),
            "CV %": sd / mean * 100 if mean != 0 else np.nan,
            f"{int(ci * 100)}% CI low": mean - tcrit * sem, f"{int(ci * 100)}% CI high": mean + tcrit * sem,
            "sum": s.sum(),
        })
    return pd.DataFrame(rows).set_index("column")


def percentile_table(df, cols):
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    out = df[cols].quantile(qs).T
    out.columns = [f"P{int(q * 100)}" for q in qs]
    return out


def categorical_summary(df, cols):
    rows = []
    for c in cols:
        vc = df[c].value_counts(dropna=True)
        if vc.empty:
            continue
        rows.append({"column": c, "count": int(vc.sum()), "missing": int(df[c].isna().sum()),
                     "unique": len(vc), "mode": vc.index[0], "mode count": int(vc.iloc[0]),
                     "mode %": vc.iloc[0] / vc.sum() * 100,
                     "entropy (bits)": float(stats.entropy(vc.values, base=2))})
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def normality_tests(s, alpha=0.05):
    s = s.dropna().astype(float)
    n = len(s)
    rows = []
    if n >= 3 and s.std() > 0:
        sample = s.sample(5000, random_state=42) if n > 5000 else s
        w, p = stats.shapiro(sample)
        rows.append(("Shapiro-Wilk", w, p, p > alpha))
    if n >= 8 and s.std() > 0:
        k2, p = stats.normaltest(s)
        rows.append(("D'Agostino-Pearson K2", k2, p, p > alpha))
    if n >= 3 and s.std() > 0:
        jb, p = stats.jarque_bera(s)
        rows.append(("Jarque-Bera", jb, p, p > alpha))
        try:
            ad = stats.anderson(s, dist="norm")
            crit = ad.critical_values[2]  # 5% level
            rows.append(("Anderson-Darling (5% critical value)", ad.statistic, np.nan, ad.statistic < crit))
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["Test", "Statistic", "p-value", f"Looks normal at alpha={alpha}?"])


def bh_adjust(p):
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1)
    return out


def strength_label(r):
    a = abs(r)
    return ("negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5
            else "strong" if a < 0.7 else "very strong")


def correlation_pairs(df, cols, method="pearson"):
    funcs = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}
    data = df[cols]
    if method == "kendall" and len(data) > 3000:
        data = data.sample(3000, random_state=42)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pair = data[[a, b]].dropna()
            if len(pair) < 3 or pair[a].std() == 0 or pair[b].std() == 0:
                continue
            r, p = funcs[method](pair[a], pair[b])
            rows.append({"column A": a, "column B": b, "n": len(pair), "r": r, "p-value": p,
                         "strength": strength_label(r), "direction": "positive" if r > 0 else "negative"})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p (BH-adjusted)"] = bh_adjust(out["p-value"].values)
        out = out.reindex(out["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return out


def fisher_ci(r, n, conf=0.95):
    if n <= 3 or abs(r) >= 1:
        return np.nan, np.nan
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    crit = stats.norm.ppf((1 + conf) / 2)
    return np.tanh(z - crit * se), np.tanh(z + crit * se)


def vif_table(df, cols):
    """Variance Inflation Factor: how well each column is predicted by the others."""
    X = df[cols].dropna()
    rows = []
    if len(X) <= len(cols) + 1 or len(cols) < 2:
        return pd.DataFrame()
    for c in cols:
        others = X.drop(columns=c)
        r2 = LinearRegression().fit(others, X[c]).score(others, X[c])
        vif = np.inf if r2 >= 0.999999 else 1 / (1 - r2)
        rows.append({"column": c, "R2 vs others": r2, "VIF": vif,
                     "verdict": "severe" if vif > 10 else "moderate" if vif > 5 else "ok"})
    return pd.DataFrame(rows).set_index("column")


def chi_square_test(df, a, b):
    ct = pd.crosstab(df[a], df[b])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else np.nan
    return {"table": ct, "chi2": chi2, "p": p, "dof": dof, "cramers_v": v,
            "low_expected": float((expected < 5).mean())}


def compare_groups(df, num, grp):
    data = df[[num, grp]].dropna()
    groups = {str(k): g[num].values.astype(float) for k, g in data.groupby(grp) if len(g) >= 2}
    k = len(groups)
    if k < 2:
        return None
    out = {"k": k, "groups": groups}
    out["summary"] = pd.DataFrame({
        "n": {g: len(v) for g, v in groups.items()},
        "mean": {g: v.mean() for g, v in groups.items()},
        "std": {g: v.std(ddof=1) for g, v in groups.items()},
        "median": {g: np.median(v) for g, v in groups.items()},
    })
    out["levene"] = stats.levene(*groups.values())
    out["shapiro"] = {g: (stats.shapiro(v[:5000])[1] if len(v) >= 3 and v.std() > 0 else np.nan)
                      for g, v in groups.items()}
    tests = []
    names = list(groups)
    if k == 2:
        a, b = groups[names[0]], groups[names[1]]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
        tests.append(("Welch t-test (means)", t, p, f"Cohen's d = {d:.3f}"))
        u, p2 = stats.mannwhitneyu(a, b, alternative="two-sided")
        tests.append(("Mann-Whitney U (distributions)", u, p2, f"rank-biserial |r| = {abs(1 - 2 * u / (len(a) * len(b))):.3f}"))
    else:
        f, p = stats.f_oneway(*groups.values())
        allv = np.concatenate(list(groups.values()))
        ssb = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in groups.values())
        sst = ((allv - allv.mean()) ** 2).sum()
        tests.append(("One-way ANOVA (means)", f, p, f"eta squared = {ssb / sst:.3f}"))
        h, p2 = stats.kruskal(*groups.values())
        n = len(allv)
        tests.append(("Kruskal-Wallis H (distributions)", h, p2, f"epsilon squared = {h * (n + 1) / (n ** 2 - 1):.3f}"))
        if k <= 15:
            try:
                tk = stats.tukey_hsd(*groups.values())
                out["posthoc"] = pd.DataFrame(tk.pvalue, index=names, columns=names)
            except Exception:
                pass
    out["tests"] = pd.DataFrame(tests, columns=["Test", "Statistic", "p-value", "Effect size"])
    return out


def outlier_summary(df, cols):
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        if len(s) < 5:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = (s - s.mean()) / s.std() if s.std() > 0 else s * 0
        mad = (s - s.median()).abs().median()
        mz = 0.6745 * (s - s.median()) / mad if mad > 0 else s * 0
        rows.append({"column": c, "IQR outliers": int(((s < lo) | (s > hi)).sum()),
                     "|z| > 3": int((z.abs() > 3).sum()), "modified z > 3.5": int((mz.abs() > 3.5).sum()),
                     "lower fence": lo, "upper fence": hi, "min": s.min(), "max": s.max()})
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def key_findings(df):
    out, nums = [], analysis_numeric_cols(df)
    if nums:
        d = descriptive_table(df, nums)
        skewed = d[d["skewness"].abs() > 1].index.tolist()
        if skewed:
            out.append(f"Strongly skewed columns (|skew| > 1): {', '.join(skewed)}. Medians describe them better "
                       "than means, and a log transform may help models.")
        var = d[d["CV %"].abs() > 100].index.tolist()
        if var:
            out.append(f"Very high variability (coefficient of variation above 100%): {', '.join(var)}.")
        if len(nums) > 1:
            pairs = correlation_pairs(df, nums[:25])
            strong = pairs[pairs["r"].abs() >= 0.7] if not pairs.empty else pairs
            for _, r in strong.head(5).iterrows():
                out.append(f"{r['column A']} and {r['column B']} are {r['strength']}ly {r['direction']}ly "
                           f"correlated (r = {r['r']:.2f}).")
        nonnormal = []
        for c in nums[:25]:
            t = normality_tests(df[c])
            if not t.empty and not bool(t.iloc[0, 3]):
                nonnormal.append(c)
        if nonnormal:
            out.append(f"Columns failing the Shapiro-Wilk normality test: {', '.join(nonnormal)}. "
                       "Prefer rank-based (non-parametric) tests for these.")
    ids = set(id_like_columns(df))
    for c in [x for x in categorical_cols(df, max_unique=30) if x not in ids][:15]:
        vc = df[c].value_counts(normalize=True)
        if len(vc) and vc.iloc[0] > 0.8:
            out.append(f"'{c}' is dominated by one category ({vc.index[0]}, {vc.iloc[0] * 100:.0f}%): "
                       "imbalanced for classification.")
    miss = df.isna().mean()
    miss = miss[miss > 0]
    if len(miss):
        out.append(f"{len(miss)} column(s) still contain missing values, worst: "
                   f"{miss.idxmax()} ({miss.max() * 100:.1f}%).")
    return out or ["No notable issues detected."]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def _verdict(p, alpha):
    return (f"p = {p:.4g} < {alpha}: statistically significant (reject H0)." if p < alpha
            else f"p = {p:.4g} >= {alpha}: not significant (cannot reject H0).")


def render_statistics(df: pd.DataFrame):
    st.subheader("Statistical Analysis")
    nums = numeric_cols(df)
    cats = categorical_cols(df, max_unique=30)
    if not nums and not cats:
        st.warning("No numeric or categorical columns to analyse.")
        return

    tabs = st.tabs(["Key Findings", "Descriptive", "Distribution & Normality", "Correlation",
                    "Categorical", "Hypothesis Tests", "Outliers"])

    with tabs[0]:
        st.write("**Automatic findings**")
        for line in key_findings(df):
            st.markdown(f"- {line}")

    with tabs[1]:
        if nums:
            chosen = st.multiselect("Numeric columns", nums, default=nums[:15])
            if chosen:
                st.write("**Descriptive statistics**")
                st.dataframe(descriptive_table(df, chosen))
                st.write("**Percentiles**")
                st.dataframe(percentile_table(df, chosen))
        if cats:
            st.write("**Categorical summary**")
            st.dataframe(categorical_summary(df, cats))
        show_method("descriptive statistics", [descriptive_table, percentile_table, categorical_summary],
                    {"pandas": "mean, median, quantile, skew and kurt computed per column.",
                     "scipy.stats": "t distribution for confidence intervals; entropy for categories.",
                     "numpy": "Square roots and vectorised arithmetic."})

    with tabs[2]:
        if not nums:
            st.info("No numeric columns.")
        else:
            col = st.selectbox("Column", nums, key="norm_col")
            s = df[col].dropna()
            alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01, key="norm_alpha")
            res = normality_tests(s, alpha)
            st.dataframe(res.set_index("Test"))
            if len(s) > 5000:
                st.caption("Shapiro-Wilk uses a random sample of 5,000 rows. Very large samples reject normality "
                           "for tiny deviations, so also inspect the plots.")
            g1, g2 = st.columns(2)
            with g1:
                fig, ax = plt.subplots()
                sns.histplot(s, kde=True, ax=ax, color="steelblue")
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)
                plt.close(fig)
            with g2:
                fig, ax = plt.subplots()
                stats.probplot(s, dist="norm", plot=ax)
                ax.set_title("Normal Q-Q plot")
                st.pyplot(fig)
                plt.close(fig)
            sk = s.skew()
            if abs(sk) > 1 and s.min() > 0:
                st.info(f"Skewness is {sk:.2f}. A log transform (np.log1p) often makes such data more symmetric.")
            st.write("**All numeric columns at a glance**")
            glance = []
            for c in nums:
                t = normality_tests(df[c], alpha)
                glance.append({"column": c, "skewness": df[c].skew(), "kurtosis": df[c].kurt(),
                               "Shapiro p": t.iloc[0, 2] if not t.empty else np.nan,
                               "normal?": bool(t.iloc[0, 3]) if not t.empty else None})
            st.dataframe(pd.DataFrame(glance).set_index("column"))
            show_method("normality", [normality_tests],
                        {"scipy.stats": "shapiro, normaltest, jarque_bera, anderson and probplot (Q-Q plot).",
                         "seaborn / matplotlib": "Histogram with density curve and Q-Q plot."})

    with tabs[3]:
        if len(nums) < 2:
            st.info("Need at least two numeric columns.")
        else:
            method = st.selectbox("Method", ["pearson", "spearman", "kendall"],
                                  help="Pearson: linear. Spearman/Kendall: monotonic, robust to outliers.")
            cols = st.multiselect("Columns", nums, default=nums[:15], key="corr_cols")
            if len(cols) >= 2:
                fig, ax = plt.subplots(figsize=(8, 5.5))
                sns.heatmap(df[cols].corr(method=method), annot=True, fmt=".2f", cmap="coolwarm",
                            vmin=-1, vmax=1, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
                pairs = correlation_pairs(df, cols, method)
                st.write("**Ranked pairs with significance** (BH adjusts for testing many pairs at once)")
                st.dataframe(pairs)
                if not pairs.empty:
                    a, b = st.columns(2)
                    x = a.selectbox("X", cols, key="cx")
                    y = b.selectbox("Y", cols, index=min(1, len(cols) - 1), key="cy")
                    if x != y:
                        pair = df[[x, y]].dropna()
                        fig, ax = plt.subplots()
                        sns.regplot(data=pair, x=x, y=y, ax=ax, scatter_kws={"alpha": 0.4})
                        st.pyplot(fig)
                        plt.close(fig)
                        if len(pair) > 3:
                            r, p = stats.pearsonr(pair[x], pair[y])
                            lo, hi = fisher_ci(r, len(pair))
                            st.write(f"Pearson r = {r:.3f}, 95% CI [{lo:.3f}, {hi:.3f}], n = {len(pair)}. "
                                     + _verdict(p, 0.05))
                st.write("**Multicollinearity (Variance Inflation Factor)**")
                v = vif_table(df, cols)
                if v.empty:
                    st.info("Not enough complete rows to compute VIF.")
                else:
                    st.dataframe(v)
                    st.caption("VIF above 5 suggests overlap with other columns; above 10 is severe.")
            show_method("correlation", [correlation_pairs, fisher_ci, vif_table, bh_adjust, strength_label],
                        {"scipy.stats": "pearsonr, spearmanr and kendalltau return the coefficient and p-value.",
                         "scikit-learn": "LinearRegression gives the R-squared needed for VIF.",
                         "numpy": "arctanh/tanh for the Fisher confidence interval; BH adjustment.",
                         "seaborn": "Heatmap and regression scatter."})

    with tabs[4]:
        if not cats:
            st.info("No categorical columns with 30 or fewer categories.")
        else:
            c = st.selectbox("Column", cats, key="cat_freq")
            vc = df[c].value_counts(dropna=False)
            freq = pd.DataFrame({"count": vc, "percent": (vc / vc.sum() * 100).round(2)})
            st.dataframe(freq)
            st.bar_chart(vc.head(20))
            if len(cats) >= 2:
                st.write("**Chi-square test of independence**")
                a, b = st.columns(2)
                c1 = a.selectbox("Variable 1", cats, key="chi1")
                c2 = b.selectbox("Variable 2", cats, index=1, key="chi2")
                if c1 != c2:
                    r = chi_square_test(df[[c1, c2]].dropna(), c1, c2)
                    st.dataframe(r["table"])
                    st.write(f"chi2 = {r['chi2']:.3f}, dof = {r['dof']}, Cramer's V = {r['cramers_v']:.3f}. "
                             + _verdict(r["p"], 0.05))
                    if r["low_expected"] > 0.2:
                        st.warning(f"{r['low_expected'] * 100:.0f}% of cells have expected counts below 5; "
                                   "the chi-square approximation may be unreliable.")
            show_method("categorical association", [chi_square_test, categorical_summary],
                        {"scipy.stats": "chi2_contingency tests whether two categories are independent.",
                         "pandas": "crosstab builds the contingency table."})

    with tabs[5]:
        kind = st.radio("Test", ["Compare groups", "One-sample t-test"], horizontal=True)
        alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01, key="ht_alpha")
        if kind == "Compare groups":
            if not nums or not cats:
                st.info("Needs one numeric and one categorical column.")
            else:
                a, b = st.columns(2)
                num = a.selectbox("Numeric outcome", nums, key="cg_num")
                grp = b.selectbox("Grouping column", cats, key="cg_grp")
                r = compare_groups(df, num, grp)
                if r is None:
                    st.info("Need at least two groups with two or more values.")
                else:
                    st.write("**Group summary**")
                    st.dataframe(r["summary"])
                    lev_stat, lev_p = r["levene"]
                    normal_ok = all((p > alpha) for p in r["shapiro"].values() if not np.isnan(p))
                    big = min(len(v) for v in r["groups"].values()) >= 30
                    st.write(f"Assumptions: Levene equal-variance p = {lev_p:.4g}; "
                             f"normality in every group: {'yes' if normal_ok else 'no'}"
                             f"{' (but every group has 30+ rows, so the CLT helps)' if big and not normal_ok else ''}.")
                    st.dataframe(r["tests"].set_index("Test"))
                    use_param = normal_ok or big
                    pick = 0 if use_param else 1
                    row = r["tests"].iloc[pick]
                    st.success(f"Recommended: **{row['Test']}**. " + _verdict(row["p-value"], alpha))
                    if "posthoc" in r:
                        st.write("**Tukey HSD post-hoc p-values** (which groups differ)")
                        st.dataframe(r["posthoc"])
                    fig, ax = plt.subplots()
                    order = r["summary"].index.tolist()
                    plot_df = df[[num, grp]].dropna()
                    plot_df[grp] = plot_df[grp].astype(str)
                    sns.boxplot(data=plot_df, x=grp, y=num, order=order, ax=ax)
                    plt.xticks(rotation=30)
                    st.pyplot(fig)
                    plt.close(fig)
                show_method("group comparison", [compare_groups],
                            {"scipy.stats": "levene, ttest_ind, mannwhitneyu, f_oneway, kruskal, tukey_hsd, shapiro.",
                             "numpy": "Effect sizes (Cohen's d, eta squared)."})
        else:
            if not nums:
                st.info("No numeric columns.")
            else:
                col = st.selectbox("Column", nums, key="t1_col")
                s = df[col].dropna()
                mu0 = st.number_input("Hypothesised mean (H0)", value=float(s.mean()) if len(s) else 0.0)
                if len(s) > 2:
                    t, p = stats.ttest_1samp(s, mu0)
                    ci = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s))
                    st.write(f"mean = {s.mean():.4f}, t = {t:.3f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]. "
                             + _verdict(p, alpha))

    with tabs[6]:
        if not nums:
            st.info("No numeric columns.")
        else:
            st.dataframe(outlier_summary(df, nums))
            col = st.selectbox("Column", nums, key="out_col")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col].dropna(), ax=ax, color="lightcoral")
            st.pyplot(fig)
            plt.close(fig)
            show_method("outlier detection", [outlier_summary],
                        {"pandas": "Quantiles for the IQR rule; medians for the MAD-based modified z-score.",
                         "numpy": "Vectorised z-scores and fences."})
