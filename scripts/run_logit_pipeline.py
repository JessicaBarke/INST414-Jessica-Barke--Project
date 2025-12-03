"""
Sprint 3 logistic regression pipeline for QSSR track.

This script is the main modeling pipeline for the project:
"Analysis of Social Media Use and Academic Outcomes in College Students"

Why this script exists:
    1. Load the cleaned modeling dataset
    2. Build a design matrix with a clear outcome and predictors
    3. Fit logistic regression models (full, reduced, and train/test)
    4. Compute baseline and model performance metrics
    5. Run cross validation to check generalization
    6. Save model summaries, metrics tables, and figures for Sprint 3

Dataset:
    data/processed/social_media_addiction_vs_relationships.csv

Outcome:
    Affects_Academic_Performance (Yes/No -> 1/0)
    This is the survey question that directly asks if social media affects academic performance.

Predictors:
    Addicted_Score
    Avg_Daily_Usage_Hours
    Sleep_Hours_Per_Night
    Mental_Health_Score
    Conflicts_Over_Social_Media

    These were chosen because they capture intensity of use, well-being,
    and conflict around social media, which are all plausible risk factors
    for academic strain.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

import statsmodels.api as sm

# Fixed random seed so results are reproducible across runs
RANDOM_STATE = 123


def get_project_paths():
    """
    Locate the main project directories using relative paths.

    Why:
        Graders and future users should be able to clone the repo
        and run this script without editing any hard-coded file paths.
    """
    # This file lives in scripts/, so the project root is two levels up
    project_root = Path(__file__).resolve().parents[1]

    # Main modeling dataset for Sprint 3
    data_path = (
        project_root
        / "data"
        / "processed"
        / "social_media_addiction_vs_relationships.csv"
    )

    if not data_path.exists():
        # If this happens, the user either renamed or moved the file
        raise FileNotFoundError(
            f"Expected modeling data at {data_path}. Make sure the file exists there."
        )

    # Where to save tables with metrics and summaries
    results_dir = project_root / "reports" / "models"
    # Where to save all Sprint 3 figures
    figures_dir = project_root / "reports" / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Using data file:", data_path.relative_to(project_root))

    return data_path, results_dir, figures_dir, project_root


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Read the cleaned modeling dataset.

    Why:
        Keep the read step separate so it is easy to swap datasets
        or check the shape of the data during debugging.
    """
    df = pd.read_csv(data_path)
    print("Loaded data with shape", df.shape)
    return df


def build_design_matrix(df: pd.DataFrame):
    """
    Build X (predictors) and y (binary outcome) for logistic regression.

    Why:
        It keeps all variable choices in one place and makes it easy
        to report to the reader which predictors were used and how
        the outcome was coded.
    """
    target_col = "Affects_Academic_Performance"

    if target_col not in df.columns:
        raise ValueError(
            f"{target_col} not found in columns. Available columns: {df.columns.tolist()}"
        )

    # Clean and recode the Yes/No responses into 1/0 for logistic regression
    y_raw = df[target_col].astype(str).str.strip().str.lower()
    y = y_raw.map({"yes": 1, "no": 0})

    if y.isna().any():
        raise ValueError(
            "Found values in Affects_Academic_Performance that are not Yes/No. "
            "Clean or recode them before modeling."
        )

    # Core behavioral and well-being predictors for the model
    feature_cols = [
        "Addicted_Score",
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Conflicts_Over_Social_Media",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing predictor columns: {missing}")

    X = df[feature_cols].copy()

    # Add intercept term for logistic regression
    X = sm.add_constant(X)

    print("Using outcome column:", target_col)
    print("Using predictors:")
    for c in feature_cols:
        print("  ", c)

    return X, y, feature_cols, target_col


def compute_baseline_metrics(y: pd.Series) -> pd.DataFrame:
    """
    Majority-class baseline.

    Why:
        This tells us how well a trivial model would do just by always
        predicting the most common class. Our real model needs to beat this
        to be considered useful.
    """
    majority_class = int(y.mode()[0])
    baseline_preds = np.full_like(y, fill_value=majority_class)
    acc = accuracy_score(y, baseline_preds)

    metrics = pd.DataFrame(
        {
            "model": ["baseline_majority_class"],
            "accuracy": [acc],
            "majority_class": [majority_class],
        }
    )
    return metrics


def fit_logit_model(X: pd.DataFrame, y: pd.Series):
    """
    Fit a logistic regression model with a safety fallback.

    Why:
        The dataset shows near-perfect separation for some predictors.
        Regular Logit can fail in that case, so we try the standard fit
        first, and if it fails, we fall back to a lightly regularized fit.
    """
    model = sm.Logit(y, X)
    try:
        result = model.fit(disp=False)
        return result
    except Exception as e:
        print("Standard Logit fit failed with:", repr(e))
        print("Falling back to regularized Logit fit.")
        # L1 regularization stabilizes the fit when there is separation
        result = model.fit_regularized(method="l1", alpha=1e-4, maxiter=200)
        return result


def logit_metrics(result, X: pd.DataFrame, y: pd.Series, model_name: str):
    """
    Compute performance metrics and ROC points for a logistic model.

    Why:
        Sprint 3 requires accuracy, precision, recall, F1, ROC AUC,
        and a confusion matrix so we can tell a full story about model quality.
    """
    pred_probs = result.predict(X)
    preds = (pred_probs >= 0.5).astype(int)

    acc = accuracy_score(y, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    fpr, tpr, _ = roc_curve(y, pred_probs)
    auc = roc_auc_score(y, pred_probs)

    metrics = pd.DataFrame(
        {
            "model": [model_name],
            "accuracy": [acc],
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
            "roc_auc": [auc],
            "tn": [tn],
            "fp": [fp],
            "fn": [fn],
            "tp": [tp],
        }
    )

    return metrics, fpr, tpr, pred_probs


def save_confusion_matrix_plot(tn, fp, fn, tp, figures_dir: Path, suffix: str):
    """
    Save a labeled confusion matrix heatmap.

    Why:
        This gives a visual way to explain what kinds of mistakes the model
        is making in the Results and Diagnostics sections.
    """
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred no impact", "Pred impact"])
    ax.set_yticklabels(["True no impact", "True impact"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_title("Confusion matrix " + suffix)
    fig.colorbar(im, ax=ax)

    fig.savefig(figures_dir / f"confusion_matrix_{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def save_roc_plot(fpr, tpr, auc, figures_dir: Path, suffix: str):
    """
    Save ROC curve plot.

    Why:
        ROC AUC is required for Sprint 3 evaluation and is a standard way
        to show the trade-off between sensitivity and specificity.
    """
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve " + suffix)
    ax.legend()

    fig.savefig(figures_dir / f"roc_curve_{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def save_probability_plot(pred_probs, y, figures_dir: Path, suffix: str):
    """
    Plot predicted probability distributions by true class.

    Why:
        This helps show separation: if the model is working well,
        the "impact" group should cluster near high probabilities and
        the "no impact" group near low probabilities.
    """
    fig, ax = plt.subplots()
    ax.hist(pred_probs[y == 0], bins=20, alpha=0.6, label="No impact")
    ax.hist(pred_probs[y == 1], bins=20, alpha=0.6, label="Impact")

    ax.set_xlabel("Predicted probability of impact")
    ax.set_ylabel("Count")
    ax.set_title("Predicted probability by outcome " + suffix)
    ax.legend()

    fig.savefig(
        figures_dir / f"predicted_probabilities_{suffix}.png", bbox_inches="tight"
    )
    plt.close(fig)


def save_odds_ratios(result, results_dir: Path, figures_dir: Path):
    """
    Save odds ratios and CIs from the full model and plot them.

    Why:
        Odds ratios translate coefficients into something easier to read.
        This is what you will cite in the Results and Interpretation section.
    """
    params = result.params
    conf_int = result.conf_int()
    conf_int.columns = ["lower", "upper"]

    or_df = pd.DataFrame(
        {
            "feature": params.index,
            "odds_ratio": np.exp(params),
            "or_lower": np.exp(conf_int["lower"]),
            "or_upper": np.exp(conf_int["upper"]),
        }
    )
    # Drop intercept from the table
    or_df = or_df[or_df["feature"] != "const"]

    or_df.to_csv(results_dir / "odds_ratios.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(or_df)))
    ax.errorbar(
        or_df["odds_ratio"],
        or_df["feature"],
        xerr=[
            or_df["odds_ratio"] - or_df["or_lower"],
            or_df["or_upper"] - or_df["odds_ratio"],
        ],
        fmt="o",
    )
    ax.axvline(1.0, linestyle=":")
    ax.set_xlabel("Odds ratio")
    ax.set_title("Odds ratios with 95% CIs")

    fig.savefig(figures_dir / "odds_ratios_plot.png", bbox_inches="tight")
    plt.close(fig)


def cross_validated_accuracy(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Run stratified k-fold cross validation for accuracy.

    Why:
        This checks how stable the model is across different train/validation
        splits and supports the generalization claims in Sprint 3.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = sm.Logit(y_train, X_train)
        try:
            result = model.fit(disp=False)
        except Exception:
            # Same separation issue can appear in folds,
            # so we use the same regularized fallback.
            result = model.fit_regularized(method="l1", alpha=1e-4, maxiter=200)

        val_probs = result.predict(X_val)
        val_preds = (val_probs >= 0.5).astype(int)
        acc = accuracy_score(y_val, val_preds)

        scores.append({"fold": fold, "accuracy": acc})

    cv_df = pd.DataFrame(scores)
    cv_df["accuracy_mean"] = cv_df["accuracy"].mean()
    cv_df["accuracy_std"] = cv_df["accuracy"].std()

    return cv_df


def train_test_split_model(X: pd.DataFrame, y: pd.Series):
    """
    Train and evaluate a model on a single train/test split.

    Why:
        The train/test split gives you a clean story:
        train the model on 80 percent of the data and test generalization
        on the remaining 20 percent.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = sm.Logit(y_train, X_train)
    try:
        result = model.fit(disp=False)
    except Exception:
        result = model.fit_regularized(method="l1", alpha=1e-4, maxiter=200)

    test_probs = result.predict(X_test)
    test_preds = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_preds, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    auc = roc_auc_score(y_test, test_probs)

    metrics = pd.DataFrame(
        {
            "model": ["logit_train_test_split"],
            "accuracy": [acc],
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
            "roc_auc": [auc],
            "tn": [tn],
            "fp": [fp],
            "fn": [fn],
            "tp": [tp],
        }
    )

    return result, metrics, test_probs, y_test


def run_pipeline():
    """
    Orchestrate the full Sprint 3 pipeline.

    Why:
        This function is the single entry point for the script and
        ties all steps together so anyone can reproduce the analysis
        by running one command.
    """
    data_path, results_dir, figures_dir, project_root = get_project_paths()
    df = load_data(data_path)

    X, y, feature_cols, target_col = build_design_matrix(df)

    # 1. Baseline model (majority class)
    baseline_df = compute_baseline_metrics(y)

    # 2. Full model using all predictors
    full_result = fit_logit_model(X, y)
    full_metrics, fpr, tpr, full_probs = logit_metrics(
        full_result, X, y, "logit_full"
    )

    tn, fp, fn, tp = (
        int(full_metrics["tn"].iloc[0]),
        int(full_metrics["fp"].iloc[0]),
        int(full_metrics["fn"].iloc[0]),
        int(full_metrics["tp"].iloc[0]),
    )
    save_confusion_matrix_plot(tn, fp, fn, tp, figures_dir, suffix="full")
    save_roc_plot(
        fpr,
        tpr,
        float(full_metrics["roc_auc"].iloc[0]),
        figures_dir,
        suffix="full",
    )
    save_probability_plot(full_probs, y, figures_dir, suffix="full")
    save_odds_ratios(full_result, results_dir, figures_dir)

    # 3. Reduced model dropping one predictor to check robustness
    reduced_cols = [c for c in X.columns if c not in ["Conflicts_Over_Social_Media"]]
    X_reduced = X[reduced_cols]
    reduced_result = fit_logit_model(X_reduced, y)
    reduced_metrics, _, _, _ = logit_metrics(
        reduced_result, X_reduced, y, "logit_reduced"
    )

    # 4. Cross validation to assess generalization
    cv_df = cross_validated_accuracy(X, y, n_splits=5)

    # 5. Train/test split model for a clean out-of-sample evaluation
    tts_result, tts_metrics, test_probs, y_test = train_test_split_model(X, y)

    tn2, fp2, fn2, tp2 = (
        int(tts_metrics["tn"].iloc[0]),
        int(tts_metrics["fp"].iloc[0]),
        int(tts_metrics["fn"].iloc[0]),
        int(tts_metrics["tp"].iloc[0]),
    )
    save_confusion_matrix_plot(
        tn2, fp2, fn2, tp2, figures_dir, suffix="train_test"
    )
    save_probability_plot(test_probs, y_test, figures_dir, suffix="train_test")

    # 6. Save misclassified cases for qualitative error analysis
    full_preds = (full_probs >= 0.5).astype(int)
    mis_idx = df.index[full_preds != y]
    df.loc[mis_idx].to_csv(
        results_dir / "misclassified_cases_logit_full.csv", index=False
    )

    # Combine all metric tables into one CSV
    metrics_all = pd.concat(
        [baseline_df, full_metrics, reduced_metrics, tts_metrics],
        ignore_index=True,
    )
    metrics_all.to_csv(
        results_dir / "classification_metrics_logit.csv", index=False
    )

    # Save cross validation results
    cv_df.to_csv(results_dir / "cross_validation_logit.csv", index=False)

    # Save model summaries so you can quote them in the report
    with open(results_dir / "logit_full_summary.txt", "w") as f:
        f.write(str(full_result.summary()))

    with open(results_dir / "logit_reduced_summary.txt", "w") as f:
        f.write(str(reduced_result.summary()))

    with open(results_dir / "logit_train_test_summary.txt", "w") as f:
        f.write(str(tts_result.summary()))

    print("Pipeline finished.")
    print("Results saved in:", results_dir)
    print("Figures saved in:", figures_dir)


if __name__ == "__main__":
    # Single entry point for the pipeline.
    # Why:
    #   This makes the script behave well both when run directly
    #   and if imported from another module.
    run_pipeline()
