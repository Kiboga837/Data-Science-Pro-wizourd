"""Predictive modelling: train many models, compare with cross-validation,
optionally tune the best one, and choose the winner."""
import time
import warnings
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from common import id_like_columns, is_text, numeric_cols, show_method

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SLOW_MODELS = {"SVR", "Support Vector Machine", "KNN"}


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def prepare_frame(X: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """Expand dates into year/month/day/weekday and make categoricals plain objects."""
    X = X.copy()
    for c in date_cols:
        d = pd.to_datetime(X[c], errors="coerce")
        X[f"{c}_year"], X[f"{c}_month"] = d.dt.year, d.dt.month
        X[f"{c}_day"], X[f"{c}_dayofweek"] = d.dt.day, d.dt.dayofweek
        X = X.drop(columns=c)
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])):
            # text/bool/category/mixed columns become plain strings (gaps stay missing) so imputing and
            # one-hot encoding never trip over mixed Python types such as 1, "two" and 3.0 in one column
            X[c] = X[c].astype(object).map(lambda v: v if pd.isna(v) else str(v))
    return X


def make_preprocessor() -> ColumnTransformer:
    """Impute and scale numbers; impute and one-hot encode categories. Lives INSIDE the
    pipeline, so it is fitted on training folds only (no data leakage)."""
    numeric = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer(
        [("num", numeric, make_column_selector(dtype_include=np.number)),
         ("cat", categorical, make_column_selector(dtype_exclude=np.number))],
        sparse_threshold=0)


def get_models(problem: str, n_estimators: int, max_depth: int, balance: bool = False) -> dict:
    depth = None if max_depth == 0 else max_depth
    cw = "balanced" if balance else None
    if problem == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1, max_iter=10000),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
            "Decision Tree": DecisionTreeRegressor(max_depth=depth, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=n_estimators, max_depth=depth, random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(n_estimators=n_estimators, max_depth=depth, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=n_estimators, max_depth=depth or 3, random_state=42),
            "HistGradient Boosting": HistGradientBoostingRegressor(max_iter=n_estimators, max_depth=depth, random_state=42),
            "AdaBoost": AdaBoostRegressor(n_estimators=n_estimators, random_state=42),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "SVR": SVR(),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(n_estimators=n_estimators, max_depth=depth or 6, random_state=42, n_jobs=-1)
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=cw),
            "Decision Tree": DecisionTreeClassifier(max_depth=depth, random_state=42, class_weight=cw),
            "Random Forest": RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, random_state=42,
                                                    n_jobs=-1, class_weight=cw),
            "Extra Trees": ExtraTreesClassifier(n_estimators=n_estimators, max_depth=depth, random_state=42,
                                                n_jobs=-1, class_weight=cw),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_estimators, max_depth=depth or 3, random_state=42),
            "HistGradient Boosting": HistGradientBoostingClassifier(max_iter=n_estimators, max_depth=depth, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=n_estimators, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Machine": SVC(class_weight=cw),
            "Naive Bayes": GaussianNB(),
            "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(n_estimators=n_estimators, max_depth=depth or 6, random_state=42,
                                              n_jobs=-1, eval_metric="logloss")
    return models


PARAM_GRIDS = {
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]},
    "ElasticNet": {"alpha": [0.001, 0.01, 0.1, 1], "l1_ratio": [0.1, 0.5, 0.9]},
    "Decision Tree": {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 2, 5, 10]},
    "Random Forest": {"n_estimators": [100, 200, 300, 500], "max_depth": [None, 5, 10, 20],
                      "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5, 1.0]},
    "Extra Trees": {"n_estimators": [100, 200, 300, 500], "max_depth": [None, 5, 10, 20],
                    "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5, 1.0]},
    "Gradient Boosting": {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2],
                          "max_depth": [2, 3, 4, 5], "subsample": [0.7, 0.85, 1.0]},
    "HistGradient Boosting": {"learning_rate": [0.02, 0.05, 0.1, 0.2], "max_iter": [100, 200, 400],
                              "max_depth": [None, 3, 5, 8], "l2_regularization": [0, 0.1, 1.0]},
    "AdaBoost": {"n_estimators": [50, 100, 200, 400], "learning_rate": [0.01, 0.1, 0.5, 1.0]},
    "KNN": {"n_neighbors": [3, 5, 7, 9, 15], "weights": ["uniform", "distance"]},
    "SVR": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    "Support Vector Machine": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "Naive Bayes": {"var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]},
    "XGBoost": {"n_estimators": [100, 200, 400], "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 6, 8], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.7, 0.85, 1.0]},
}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def scoring_for(problem: str):
    if problem == "Regression":
        return {"r2": "r2"}, "r2"
    return {"f1_weighted": "f1_weighted"}, "f1_weighted"


def evaluate(problem: str, pipe, X, y_true, y_pred, n_classes: int = 2) -> dict:
    if problem == "Regression":
        return {"R2": r2_score(y_true, y_pred), "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))}
    out = {"Accuracy": accuracy_score(y_true, y_pred),
           "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
           "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
           "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0), "ROC AUC": np.nan}
    try:
        if n_classes == 2:
            scores = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)
            out["ROC AUC"] = roc_auc_score(y_true, scores)
        elif hasattr(pipe, "predict_proba"):
            out["ROC AUC"] = roc_auc_score(y_true, pipe.predict_proba(X), multi_class="ovr", average="weighted")
    except Exception:
        pass
    return out


def primary_metric(problem: str) -> str:
    return "R2" if problem == "Regression" else "F1"


def build_row(name, problem, pipe, cv_mean, cv_std, X_train, y_train, X_test, y_test, seconds, n_classes):
    preds = pipe.predict(X_test)
    test = evaluate(problem, pipe, X_test, y_test, preds, n_classes)
    prim = primary_metric(problem)
    train_score = (r2_score(y_train, pipe.predict(X_train)) if problem == "Regression"
                   else f1_score(y_train, pipe.predict(X_train), average="weighted", zero_division=0))
    row = {"Model": name, "CV score (mean)": cv_mean, "CV score (std)": cv_std, **test,
           "Train score": train_score, "Overfit gap": train_score - test[prim], "Train time (s)": seconds}
    return row, preds


def explain_choice(metrics: pd.DataFrame, problem: str) -> str:
    prim = primary_metric(problem)
    ranked = metrics.sort_values("CV score (mean)", ascending=False)
    best = ranked.iloc[0]
    name = ranked.index[0]
    text = (f"**{name}** is the best model: it has the highest cross-validated {prim} "
            f"({best['CV score (mean)']:.3f} +/- {best['CV score (std)']:.3f}), and scores {best[prim]:.3f} "
            f"on the untouched test set.")
    if len(ranked) > 1:
        second = ranked.iloc[1]
        gap = best["CV score (mean)"] - second["CV score (mean)"]
        if gap < best["CV score (std)"]:
            text += (f" The runner-up, {ranked.index[1]}, is within one standard deviation, so the two are "
                     "statistically hard to separate; prefer the simpler or faster one if you value that.")
    if best["Overfit gap"] > 0.15:
        text += (f" Warning: training {prim} exceeds test {prim} by {best['Overfit gap']:.2f}, which signals "
                 "overfitting. Try fewer trees, a smaller depth or more data.")
    if abs(best["CV score (mean)"] - best[prim]) > 0.1:
        text += " CV and test scores differ noticeably, so treat the numbers as approximate."
    return text


def tune_model(name, base_pipe, X_train, y_train, scoring, n_iter, cv):
    grid = {f"model__{k}": v for k, v in PARAM_GRIDS[name].items()}
    size = int(np.prod([len(v) for v in grid.values()]))
    search = RandomizedSearchCV(clone(base_pipe), grid, n_iter=min(n_iter, size), cv=cv, scoring=scoring,
                                random_state=42, n_jobs=2, refit=True)
    search.fit(X_train, y_train)
    idx = search.best_index_
    return search, float(search.cv_results_["mean_test_score"][idx]), float(search.cv_results_["std_test_score"][idx])


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def make_target(df: pd.DataFrame, target: str, problem: str, threshold=None) -> pd.Series:
    """The target series. A many-valued numeric target used for classification is split into High/Low."""
    y = df[target]
    if problem == "Classification" and pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        thr = float(y.median()) if threshold is None else float(threshold)
        y = pd.Series(np.where(y > thr, "High", "Low"), index=df.index).where(y.notna())
    return y


def train_compare(df, y_raw, target, problem, features, label="", test_size=0.2, n_estimators=100, max_depth=0,
                  cv_folds=5, balance=False, chosen=None, progress=None):
    """Train and cross-validate the chosen models.
    Returns (result_dict_or_None, messages) where messages is a list of (kind, text) with kind in
    {"warning", "error", "info"}. Used by both the Predictive Model page and the one-click analysis."""
    msgs = []
    catalog = get_models(problem, n_estimators, max_depth, balance)
    chosen = chosen or [m for m in catalog if not (len(df) > 10000 and m in SLOW_MODELS)]
    if not features or not chosen:
        return None, [("error", "Select at least one feature and one model.")]
    mask = y_raw.notna()
    data, y = df.loc[mask], y_raw.loc[mask]
    if len(data) < 20:
        return None, [("error", "Fewer than 20 usable rows. Clean or load more data.")]

    date_cols = [c for c in features if pd.api.types.is_datetime64_any_dtype(data[c])]
    dropped = [c for c in features if is_text(data[c]) and data[c].nunique() > 20]
    kept = [c for c in features if c not in dropped]
    if dropped:
        msgs.append(("warning", f"Dropped high-cardinality text columns (IDs, names): {', '.join(dropped)}"))
    if not kept:
        return None, msgs + [("error", "No usable feature columns remain.")]
    X = prepare_frame(data[kept], [c for c in date_cols if c in kept])

    if problem == "Regression":
        nums = numeric_cols(data)
        leaks = [c for c in kept if c in nums and data[c].nunique() > 1 and abs(data[c].corr(data[target])) > 0.98]
        if leaks:
            msgs.append(("warning", f"Possible data leakage: {', '.join(leaks)} is almost perfectly correlated with the target."))

    classes, stratify = None, None
    y_fit = y.astype(float) if problem == "Regression" else y
    if problem == "Classification":
        enc = LabelEncoder()
        y_fit = pd.Series(enc.fit_transform(y.astype(str)), index=y.index)
        classes = list(enc.classes_)
        if len(classes) < 2:
            return None, msgs + [("error", "The target needs at least two classes.")]
        stratify = y_fit if y_fit.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(X, y_fit, test_size=test_size, random_state=42, stratify=stratify)
    scoring, _ = scoring_for(problem)
    splitter = StratifiedKFold if problem == "Classification" and stratify is not None else KFold
    cv = splitter(n_splits=cv_folds, shuffle=True, random_state=42)

    fitted, preds_all, rows = {}, {}, []
    for i, name in enumerate(chosen):
        if progress:
            progress(i / len(chosen), f"Training {name} ({i + 1}/{len(chosen)})")
        try:
            pipe = Pipeline([("prep", make_preprocessor()), ("model", catalog[name])])
            start = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
                pipe.fit(X_train, y_train)
            key = "test_" + list(scoring)[0]
            row, preds = build_row(name, problem, pipe, cvres[key].mean(), cvres[key].std(), X_train, y_train,
                                   X_test, y_test, time.time() - start, len(classes) if classes else 2)
            rows.append(row)
            fitted[name], preds_all[name] = pipe, preds
        except Exception as ex:  # noqa: BLE001
            msgs.append(("warning", f"{name} failed and was skipped: {ex}"))
    if not fitted:
        return None, msgs + [("error", "No model could be trained with these settings.")]

    meta = {}
    for col in kept:
        s = data[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            meta[col] = {"type": "date", "default": s.dropna().median().date() if s.notna().any() else None}
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            med = 0.0 if s.dropna().empty else float(s.median())
            is_int = pd.api.types.is_integer_dtype(s)
            meta[col] = {"type": "numeric", "is_int": is_int, "default": int(round(med)) if is_int else med}
        else:
            meta[col] = {"type": "cat", "options": sorted(s.dropna().astype(str).unique().tolist())}

    result = {"sig": (tuple(df.columns), df.shape, label), "problem": problem, "target": target,
              "metrics": pd.DataFrame(rows).set_index("Model"), "models": fitted, "preds": preds_all,
              "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "classes": classes,
              "kept": kept, "date_cols": [c for c in date_cols if c in kept], "meta": meta, "cv": cv,
              "scoring": scoring, "perm": {}}
    return result, msgs


def tune_and_add(res: dict, base: str, iters: int) -> dict:
    """Tune `base` with RandomizedSearchCV and add it to the results as '<base> (tuned)'."""
    problem = res["problem"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.time()
        search, cvm, cvs = tune_model(base, res["models"][base], res["X_train"], res["y_train"],
                                      res["scoring"]["f1_weighted" if problem != "Regression" else "r2"],
                                      iters, res["cv"])
        row, preds = build_row(f"{base} (tuned)", problem, search.best_estimator_, cvm, cvs, res["X_train"],
                               res["y_train"], res["X_test"], res["y_test"], time.time() - start,
                               len(res["classes"]) if res["classes"] else 2)
    res["metrics"] = pd.concat([res["metrics"].drop(index=f"{base} (tuned)", errors="ignore"),
                                pd.DataFrame([row]).set_index("Model")])
    res["models"][f"{base} (tuned)"], res["preds"][f"{base} (tuned)"] = search.best_estimator_, preds
    return search.best_params_


def render_ml(df: pd.DataFrame, label: str):
    st.subheader("Predictive Modeling: train, compare and choose the best model")
    st.caption(f"Training data: {label} ({len(df):,} rows)")
    nums = numeric_cols(df)
    problem = st.radio("Problem type", ["Regression", "Classification"], horizontal=True)

    if problem == "Regression":
        if not nums:
            st.error("You need at least one numeric column to use as the target.")
            return
        target = st.selectbox("Select Target Column", nums)
        y_raw = df[target]
    else:
        candidates = [c for c in df.columns if c in nums or df[c].nunique() <= 20]
        if not candidates:
            st.error("No suitable target column found for classification.")
            return
        target = st.selectbox("Select Target Column", candidates)
        y_raw = df[target]
        if target in nums and df[target].nunique() > 20:
            threshold = st.number_input(f"Label rows 'High' when {target} is above (otherwise 'Low')",
                                        value=float(df[target].median()))
            y_raw = pd.Series(np.where(df[target] > threshold, "High", "Low"), index=df.index).where(df[target].notna())

    available = [c for c in df.columns if c != target]
    ids = set(id_like_columns(df))
    features = st.multiselect("Select Feature Columns", available,
                              default=[c for c in nums if c != target and c not in ids])

    a, b, c = st.columns(3)
    test_size = a.slider("Test Size", 0.1, 0.5, 0.2)
    n_estimators = b.slider("Number of Trees / Rounds", 50, 500, 100, step=50)
    max_depth = c.slider("Max Depth (0 = default)", 0, 30, 0)
    d, e = st.columns(2)
    cv_folds = d.slider("Cross-validation folds", 3, 10, 5)
    balance = e.checkbox("Balance classes (where supported)", False) if problem == "Classification" else False

    all_models = list(get_models(problem, n_estimators, max_depth, balance).keys())
    big = len(df) > 10000
    default_models = [m for m in all_models if not (big and m in SLOW_MODELS)]
    chosen = st.multiselect("Models to train and compare", all_models, default=default_models)
    if big:
        st.caption("Slow models (SVM, KNN) are off by default for datasets above 10,000 rows.")
    if not HAS_XGB:
        st.caption("Tip: `pip install xgboost` adds XGBoost to the comparison.")

    if st.button("Train & Compare Models", type="primary"):
        bar = st.progress(0.0, text="Training...")
        result, msgs = train_compare(df, y_raw, target, problem, features, label, test_size, n_estimators,
                                     max_depth, cv_folds, balance, chosen,
                                     progress=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
        bar.empty()
        for kind, text in msgs:
            getattr(st, kind)(text)
        if result is None:
            return
        st.session_state["ml_result"] = result

    res = st.session_state.get("ml_result")
    if not res or res["sig"] != (tuple(df.columns), df.shape, label):
        if st.session_state.get("ml_saved"):
            st.info("A saved model comparison from your project is included in the Report. Retrain here to inspect models or predict.")
        return
    st.caption(f"Showing the latest results: {res['problem']} model for target \"{res['target']}\". Press Train & Compare to refresh after changing settings.")
    if res.get("note"):
        st.caption(res["note"])

    problem, prim = res["problem"], primary_metric(res["problem"])
    metrics = res["metrics"].sort_values("CV score (mean)", ascending=False)
    best_name = metrics.index[0]

    st.success("Training complete")
    st.subheader("Model Comparison")
    show = metrics.copy()
    show.insert(0, "Rank", range(1, len(show) + 1))
    st.dataframe(show.round(4))
    st.markdown(explain_choice(metrics, problem))
    st.bar_chart(metrics["CV score (mean)"].rename(f"Cross-validated {prim}"))

    # ---- tuning -----------------------------------------------------------
    base = best_name.replace(" (tuned)", "")
    with st.expander(f"Tune the best model ({best_name}) with RandomizedSearchCV"):
        if base not in PARAM_GRIDS:
            st.info(f"{base} has no hyperparameters worth tuning here.")
        else:
            iters = st.slider("Search iterations", 5, 60, 20)
            st.caption("Searches: " + ", ".join(f"{k} in {v}" for k, v in PARAM_GRIDS[base].items()))
            if st.button("Run tuning"):
                with st.spinner("Searching hyperparameters..."):
                    st.session_state["ml_tuned_params"] = tune_and_add(res, base, iters)
                st.rerun()
            if "ml_tuned_params" in st.session_state and f"{base} (tuned)" in res["models"]:
                st.write("Best parameters found:", st.session_state["ml_tuned_params"])
                st.caption("The tuned CV score comes from the search itself and is slightly optimistic; "
                           "the test score is the fair comparison.")

    # ---- inspect ----------------------------------------------------------
    names = list(metrics.index)
    inspect = st.selectbox("Inspect model", names, index=0)
    pipe, preds = res["models"][inspect], res["preds"][inspect]
    y_test, X_test = res["y_test"], res["X_test"]

    g1, g2 = st.columns(2)
    with g1:
        if problem == "Regression":
            st.write("**Actual vs Predicted**")
            fig, ax = plt.subplots()
            ax.scatter(y_test, preds, alpha=0.5)
            lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
            ax.plot(lims, lims, "r--")
            ax.set_xlabel(f"Actual {res['target']}")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, preds, labels=range(len(res["classes"])))
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=res["classes"],
                        yticklabels=res["classes"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
    with g2:
        if problem == "Regression":
            st.write("**Residuals**")
            fig, ax = plt.subplots()
            ax.scatter(preds, y_test - preds, alpha=0.5)
            ax.axhline(0, color="r", linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual (actual - predicted)")
            st.pyplot(fig)
            plt.close(fig)
        elif len(res["classes"]) == 2:
            st.write("**ROC Curve**")
            try:
                sc = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, sc)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr)
                ax.plot([0, 1], [0, 1], "k--")
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
                st.pyplot(fig)
                plt.close(fig)
            except Exception:
                st.info("ROC curve unavailable for this model.")

    st.write("**Feature importance** (permutation: how much the score drops when a feature is shuffled)")
    if inspect not in res["perm"]:
        with st.spinner("Computing permutation importance..."):
            Xs = X_test.sample(min(1000, len(X_test)), random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pi = permutation_importance(pipe, Xs, y_test.loc[Xs.index],
                                            scoring="r2" if problem == "Regression" else "f1_weighted",
                                            n_repeats=5, random_state=42, n_jobs=1)
            res["perm"][inspect] = pd.DataFrame({"feature": Xs.columns, "importance": pi.importances_mean,
                                                 "std": pi.importances_std}).sort_values("importance", ascending=False)
    imp = res["perm"][inspect].head(12)
    fig, ax = plt.subplots()
    sns.barplot(data=imp, x="importance", y="feature", ax=ax, color="seagreen")
    st.pyplot(fig)
    plt.close(fig)

    if problem == "Classification":
        out = pd.DataFrame({"Actual": [res["classes"][i] for i in y_test], "Predicted": [res["classes"][int(i)] for i in preds]})
    else:
        out = pd.DataFrame({"Actual": y_test.values, "Predicted": preds}).round(3)
    d1, d2 = st.columns(2)
    d1.download_button("Download test predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                       f"predictions_{inspect.replace(' ', '_').lower()}.csv", "text/csv")
    buf = BytesIO()
    joblib.dump({"pipeline": pipe, "features": res["kept"], "date_cols": res["date_cols"],
                 "classes": res["classes"]}, buf)
    d2.download_button(f"Download {inspect} (.joblib)", buf.getvalue(), "model.joblib")

    # ---- single-row prediction -------------------------------------------
    st.subheader(f"Predict a Single Row ({inspect})")
    with st.form("single_prediction"):
        inputs, cols = {}, st.columns(3)
        for i, col in enumerate(res["kept"]):
            m = res["meta"][col]
            with cols[i % 3]:
                if m["type"] == "numeric":
                    inputs[col] = st.number_input(col, value=m["default"], step=1 if m["is_int"] else None)
                elif m["type"] == "date":
                    inputs[col] = st.date_input(col, value=m["default"])
                else:
                    inputs[col] = st.selectbox(col, m["options"])
        submitted = st.form_submit_button("Predict")
    if submitted:
        row = pd.DataFrame([{k: (pd.Timestamp(v) if res["meta"][k]["type"] == "date" else v) for k, v in inputs.items()}])
        row = prepare_frame(row, res["date_cols"])
        pred = pipe.predict(row)[0]
        if problem == "Regression":
            st.metric(f"Predicted {res['target']}", f"{float(pred):,.2f}")
        else:
            st.metric(f"Predicted {res['target']}", res["classes"][int(pred)])
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(row)[0]
                st.dataframe(pd.DataFrame({"Class": res["classes"], "Probability": proba.round(3)}).set_index("Class"))

    show_method(
        "model training and selection",
        [make_preprocessor, prepare_frame, get_models, evaluate, build_row, explain_choice, tune_model],
        {"scikit-learn": "Models, Pipeline, ColumnTransformer, cross_validate, RandomizedSearchCV, metrics, permutation_importance.",
         "pandas": "Holds the data and expands dates into year/month/weekday features.",
         "numpy": "Numeric arrays and error calculations.",
         "xgboost (optional)": "Gradient boosting library that is often among the strongest models.",
         "joblib": "Saves the trained pipeline so it can be reused later.",
         "seaborn / matplotlib": "Confusion matrix, residual, ROC and importance charts."},
    )
