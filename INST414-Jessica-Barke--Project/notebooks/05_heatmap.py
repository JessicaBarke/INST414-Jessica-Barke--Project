"""
Generate confusion matrix heatmap for Sprint 3
Uses the confusion matrix counts saved in reports/models/classification_metrics.csv
and saves a figure in reports/figures/confusion_matrix.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROJECT_ROOT = Path(".")
MODELS_DIR = PROJECT_ROOT / "reports" / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Load confusion matrix counts from metrics file
metrics_path = MODELS_DIR / "classification_metrics.csv"
metrics = pd.read_csv(metrics_path)

# Expect columns: tn, fp, fn, tp (from 04_models_qssr.py)
tn = int(metrics.loc[0, "tn"])
fp = int(metrics.loc[0, "fp"])
fn = int(metrics.loc[0, "fn"])
tp = int(metrics.loc[0, "tp"])

cm = np.array([[tn, fp],
               [fn, tp]])

print("Confusion matrix from metrics file:")
print(cm)

# Make sure figures directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No impact", "Impact"],
    yticklabels=["No impact", "Impact"],
)

plt.title("Confusion Matrix for Logistic Regression Model")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()

out_path = FIGURES_DIR / "confusion_matrix.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved confusion matrix heatmap to {out_path}")
