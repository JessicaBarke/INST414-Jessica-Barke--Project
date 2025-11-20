# INST414-Jessica-Barke--Project
INST414 – Sprint 3 Progress (QSSR)
# INST414 – Sprint 3 Progress (QSSR)
**Author:** Jessica Barke • **Course:** INST414 (QSSR) • **Date:** Nov 2025

## Summary
- Completed full modeling workflow for Sprint 3 using logistic regression to predict academic impact.
- Implemented behavioral predictors including `hours_social_media`, `addiction_score`, `sleep_hours`, `heavy_user`, `sleep_ok`, and one-hot encoded `platform_group`.
- Generated model diagnostics and visualizations, including confusion matrix heatmap, odds ratio plot, platform group comparison, hours-versus-impact scatterplot, and predicted probability plot.
- Key finding: behavioral variables strongly predict academic impact, with addiction-related features showing the sharpest separation; however, quasi-separation and multicollinearity create unstable coefficients that will be addressed in Sprint 4.

## What to open
- Modeling and diagnostics: `notebooks/04_models_qssr.py`
- Confusion matrix and probability results: `reports/models/classification_metrics.csv` and `reports/models/logit_sample_size_tracking.csv`
- Model summary and odds ratios: `reports/models/model_summary.txt` and `reports/models/odds_ratios.csv`
- Figures: all visualizations, including `confusion_matrix.png` and `odds_ratio_plot.png`, under `reports/figures/`
- Sprint 3 report: `reports/sprint3_report.pdf` (technical write-up)

## Next (Sprint 4)
- Add regularized logistic regression (L1/L2) to stabilize extreme coefficients and reduce quasi-separation.
- Introduce a train–test split or k-fold cross-validation to evaluate generalization.
- Simplify overlapping predictors (for example, revisiting `hours_social_media` and `heavy_user`) and test interaction terms between usage and sleep quality.
- Refine coefficient plots and domain-focused visualizations for the final presentation and non-technical summary.

## Sources
- Kaggle (Shamim, 2023) and (Mashayekhi, 2024)
- Repo: github.com/JessicaBarke/INST414-Jessica-Barke--Project
