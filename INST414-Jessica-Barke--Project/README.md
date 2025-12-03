# INST414 – Sprint 3 Progress (QSSR)

**Author:** Jessica Barke  
**Course:** INST414 (QSSR)  
**Last Updated:** December 2025  

---

## Project Overview

This project investigates how social media usage influences college students’ academic performance. The goal is to understand whether behaviors like addiction, conflict, sleep disruption, or heavy use meaningfully predict academic strain. This matters because a lot of college students feel stressed or distracted by social media but rarely see clear data on which specific habits are actually linked to academic problems.

**Research Question:**  
How does social media usage affect college students’ academic performance?

The modeling dataset (`social_media_addiction_vs_relationships.csv`) includes daily usage hours, mental health scores, sleep hours, conflict indicators, addicted score, and a Yes/No outcome for whether social media affects academic performance.

---

## Repository Structure

```
INST414-Jessica-Barke--Project/
├── data/                 # Raw, interim, and processed datasets
├── inst414_project/      # Cookiecutter scaffold modules
├── notebooks/            # All sprint-by-sprint analysis notebooks (as .py exports)
├── reports/              # Generated figures, tables, model outputs
├── scripts/              # Full modeling pipeline and helper scripts
└── README.md             # This file
```

Each folder has a purpose:  
- **data** holds raw and cleaned datasets.  
- **notebooks** contains step-by-step work from Sprint 1 → Sprint 3.  
- **scripts** contains reusable scripts, including the full logistic regression pipeline.  
- **reports** stores all figures, metrics, summaries, and model results.  
- **inst414_project** is the project scaffold used throughout the semester.

---
# Data Description

**Source:**  
Two Kaggle datasets on social media usage and relationships.

**Final modeling dataset:**  
`data/processed/social_media_addiction_vs_relationships.csv`

**Rows:** 705  
**Columns:** 13  

Key variables include:  
- Daily usage hours  
- Addiction score  
- Sleep hours  
- Social media conflict indicators  
- Mental health score  
- Academic impact (binary outcome)

---
# Sprint Summaries

## Sprint 1 Summary

Sprint 1 focused on setting up the project and understanding the data structure.

**Completed:**  
- Full variable inventory (QSSR version + simplified version)  
- Quick-load checks to confirm dataset formatting  
- Dataset summaries and early descriptive insights  
- Initial exploration of how social media themes might connect to academic performance  

---

## Sprint 2 Summary

Sprint 2 focused on cleaning the dataset and preparing it for modeling.

**Completed:**  
- Cleaning pipeline script  
- Cleaning log documenting decisions  
- Updated variable inventory  
- Missingness checks, outlier checks, and distribution analysis  
- Correlation heatmaps and early exploratory relationships  
- Final processed dataset saved under `data/processed/`  

This sprint completed all preparation steps needed for modeling.

---

# Sprint 3: Modeling Strategy

Sprint 3 required building a complete logistic regression workflow, handling quasi-separation, evaluating model performance, and generating all diagnostic outputs.

## Outcome Variable

`Affects_Academic_Performance`  
(Yes → 1, No → 0)

## Predictors Used

These predictors capture meaningful behavioral signals connected to academic strain:

- Addicted_Score  
- Avg_Daily_Usage_Hours  
- Sleep_Hours_Per_Night  
- Mental_Health_Score  
- Conflicts_Over_Social_Media  

## Modeling Pipeline

Implemented components:

- Baseline accuracy  
- Logistic regression  
- Automatic fallback to regularized logistic regression  
- Train/test split with fixed random seed  
- Odds ratios + confidence intervals  
- Predicted probability curves  
- Confusion matrix and classification metrics  
- ROC curve + AUC  
- Feature importance via coefficient magnitude  

Outputs are saved under:

```
reports/models/
reports/figures/
```

## Key Sprint 3 Findings

- Behavioral predictors strongly relate to academic strain.  
- Addicted score and conflict variables show the sharpest class separation.  
- The model achieves strong accuracy and sensitivity.  
- Predicted probability curves reveal nearly perfect separation for many students.  
- Quasi-separation warnings confirm some predictors are extremely strong and push logistic regression toward infinite estimates.  
- Regularization will be necessary in Sprint 4 to stabilize coefficients.

---

# Reproducibility and How to Run

To reproduce the Sprint 3 modeling results:

```bash
python scripts/run_logit_pipeline.py

# What to Open 

### Notebooks  
- `notebooks/04_models_qssr.py`  
- `notebooks/05_heatmap.py`  
- `notebooks/06_predicted_probabilities.py`

### Pipeline Script  
- `scripts/run_logit_pipeline.py`

### Key Outputs  
- `reports/models/model_summary.txt`  
- `reports/models/odds_ratios.csv`  
- `reports/models/classification_metrics.csv`  
- `reports/figures/confusion_matrix.png`  
- `reports/figures/predicted_probabilities.png`  

---

# Sprint 4 Roadmap

To improve the model and address quasi-separation:

- Add L1/L2-regularized logistic regression  
- Explore interaction terms (usage × sleep, conflict × mental health)  
- Reduce multicollinearity  
- Add k-fold cross-validation  
- Refine visualizations for the final report and presentation  

---

# Data Sources

- Kaggle: Social Media Addiction & Relationships Dataset  
  (Shamim, 2023; Mashayekhi, 2024)

---

# Contact

Jessica Barke: jbarke1@terpmail.umd.edu
GitHub:(https://github.com/JessicaBarke/INST414-Jessica-Barke--Project/tree/main)
University of Maryland – INST414 QSSR  

