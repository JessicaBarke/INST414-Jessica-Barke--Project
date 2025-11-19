"""
04_models_qssr.py

Simple Sprint-3 logistic regression model.
Outcome: acad_impact (0/1)
Predictors: hours_social_media, addiction_score, sleep_hours,
            heavy_user, sleep_ok, platform_group (one-hot)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1. Load cleaned dataset
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "clean" / "primary_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = REPORTS_DIR / "models"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

# -------------------------
# 2. Outcome + predictors
# -------------------------
y = df["acad_impact"].astype(int)

predictors = [
    "hours_social_media",
    "addiction_score",
    "sleep_hours",
    "heavy_user",
    "sleep_ok",
]

# One-hot encode platform_group
df = pd.get_dummies(df, columns=["platform_group"], drop_first=True)

# Add the platform_group dummies to predictors list
platform_cols = [c for c in df.columns if c.startswith("platform_group_")]
predictors.extend(platform_cols)

X = df[predictors].copy()

# Force everything to float (converts bool/int to float)
X = X.astype(float)

# Add intercept
X = sm.add_constant(X)

print("Final X shape:", X.shape)
print("X dtypes:\n", X.dtypes)

# -------------------------
# 3. Baseline model
# -------------------------
baseline_rate = y.mean()
baseline_class = 1 if baseline_rate >= 0.5 else 0
baseline_accuracy = (y == baseline_class).mean()

print("\nBaseline accuracy:", baseline_accuracy)

baseline_df = pd.DataFrame({
    "baseline_positive_rate": [baseline_rate],
    "baseline_accuracy": [baseline_accuracy],
})
baseline_df.to_csv(MODELS_DIR / "baseline_metrics.csv", index=False)

# -------------------------
# 4. Logistic Regression
# -------------------------
logit_model = sm.Logit(y, X)
result = logit_model.fit()

with open(MODELS_DIR / "model_summary.txt", "w") as f:
    f.write(result.summary().as_text())

print("\nModel fit complete.")
print(result.summary())

# Odds ratios
params = result.params
odds_ratios = np.exp(params)
odds_ratios.to_csv(MODELS_DIR / "odds_ratios.csv")

# -------------------------
# 5. Classification metrics
# -------------------------
y_pred_prob = result.predict(X)
y_pred = (y_pred_prob >= 0.5).astype(int)

cm = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

metrics_df = pd.DataFrame({
    "accuracy": [acc],
    "precision": [prec],
    "recall": [rec],
    "f1": [f1],
    "tn": [cm[0,0]],
    "fp": [cm[0,1]],
    "fn": [cm[1,0]],
    "tp": [cm[1,1]],
})
metrics_df.to_csv(MODELS_DIR / "classification_metrics.csv", index=False)

print("\nConfusion matrix:")
print(cm)
print("\nMetrics:")
print(metrics_df.T)

# -------------------------
# 6. Odds ratio bar plot
# -------------------------
odds_plot_df = odds_ratios.drop("const")
odds_plot_df.sort_values().plot(kind="barh", figsize=(8,6))
plt.axvline(1.0, color="black", linestyle="--")
plt.title("Odds Ratios for Predictors")
plt.xlabel("Odds Ratio")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "odds_ratio_plot.png", dpi=300)
plt.close()

print("\nSaved: odds_ratio_plot.png")
print("Saved model files in:", MODELS_DIR)
