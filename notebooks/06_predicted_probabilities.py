"""
06_predicted_probabilities.py

What this notebook does:
    1. Loads the fitted logistic regression model and the test set.
    2. Computes predicted probabilities for each student.
    3. Plots the distribution of probabilities for impact vs no impact.
    4. Saves all figures so they can go directly into the Sprint 3 report.

Why this notebook exists:
    Predicted probabilities help show how well the model separates the groups.
    This is required for discussing model performance and evidence of separation.
    It also helps identify misclassified students for error analysis.
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "clean" / "primary_clean.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load cleaned data
df = pd.read_csv(CLEAN_PATH)

# 2. Set up outcome and predictors to match 04_models_qssr.py
y = df["acad_impact"]

X = df[[
    "hours_social_media",
    "addiction_score",
    "sleep_hours",
    "heavy_user",
    "sleep_ok",
    "platform_group"
]].copy()

# 3. One hot encode platform_group (drop one category as reference)
X = pd.get_dummies(
    X,
    columns=["platform_group"],
    drop_first=True
)

# 4. Add intercept
X = sm.add_constant(X)

# Make sure everything is numeric
X = X.astype(float)

print("Design matrix shape:", X.shape)
print("Outcome positive rate:", y.mean())

# 5. Fit logistic regression (same model as before)
logit_model = sm.Logit(y, X)
result = logit_model.fit(maxiter=35, disp=True)

# 6. Get predicted probabilities
df["pred_prob"] = result.predict(X)

print("Predicted probability summary:")
print(df["pred_prob"].describe())

# 7. Plot distribution of predicted probabilities by actual outcome
plt.figure(figsize=(8, 6))

# Students who reported academic impact = 1
df.loc[df["acad_impact"] == 1, "pred_prob"].plot.hist(
    bins=20,
    alpha=0.6,
    label="Reported academic impact = 1"
)

# Students who did not report academic impact = 0
df.loc[df["acad_impact"] == 0, "pred_prob"].plot.hist(
    bins=20,
    alpha=0.6,
    label="Reported academic impact = 0"
)

plt.xlabel("Predicted probability of academic impact")
plt.ylabel("Number of students")
plt.title("Distribution of Predicted Probabilities by Outcome")
plt.legend()

out_path = FIG_DIR / "fig_predicted_probabilities.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved predicted probability figure to: {out_path}")
