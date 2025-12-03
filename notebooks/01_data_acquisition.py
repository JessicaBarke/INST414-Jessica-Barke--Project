"""
01_data_acquisition.py

What this script does:
    1. Loads the original Kaggle social media survey data
    2. Keeps only the variables needed for this project
    3. Saves cleaned CSVs into data/raw for later steps

Why this script exists:
    The project should have a clear, reproducible starting point.
    Anyone running the repo can see exactly how the raw data was pulled
    and what early decisions were made before cleaning and modeling.
"""

from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
CARDS, RAW = ROOT/"cards", ROOT/"data"/"raw"
REPORTS, FIGS = ROOT/"reports", ROOT/"reports"/"figures"
for p in [REPORTS, FIGS]: p.mkdir(parents=True, exist_ok=True)

# Table 0: Kaggle cards summary
card_add = pd.read_csv(CARDS / "addiction_card.csv")
card_prod = pd.read_csv(CARDS / "productivity_card.csv")
cards = pd.concat([
    card_add.assign(role="Primary (Addiction vs Relationships)"),
    card_prod.assign(role="Secondary (Social Media vs Productivity)")
], ignore_index=True)
cards[['role','title','id','subtitle','isPrivate','description']].to_csv(REPORTS / "cards_summary.csv", index=False)

# Quick load check (prints)
pri = pd.read_csv(RAW / "social_media_addiction_vs_relationships.csv")
sec = pd.read_csv(RAW / "social_media_vs_productivity.csv")
print("Primary shape:", pri.shape, "| Secondary shape:", sec.shape)
print("Wrote reports/cards_summary.csv")
