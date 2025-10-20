# INST414-Jessica-Barke--Project
INST414 – Sprint 2 Progress (QSSR)

Author: Jessica Barke • Course: INST414 (QSSR) • Date: Oct 2025

Summary

Cleaned and standardized the primary and secondary Kaggle datasets.
Engineered variables: heavy_user, sleep_ok, platform_group, study_intensity.
EDA outputs saved to reports/, figures in reports/figures/.
Key finding: negative relationship between daily social-media hours and self-reported academic impact; heavy users with <7h sleep report the lowest outcomes.
What to open

Cleaning: notebooks/02_cleaning_pipeline.py
EDA: notebooks/03_eda.py
Outputs: reports/summary_stats.csv, reports/correlations.csv, reports/cleaning_log.md, figures under reports/figures/.
Next (Sprint 3)

Logistic (academic impact) + OLS (productivity) models in notebooks/04_models_qssr.py.
Test interaction heavy_user × sleep_ok, run robustness checks, and add coefficient plots.
Sources

Kaggle (Shamim, 2023) and (Mashayekhi, 2024)
Repo: github.com/JessicaBarke/INST414-Jessica-Barke--Project
