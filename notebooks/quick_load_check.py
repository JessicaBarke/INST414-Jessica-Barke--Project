"""
quick_load_check.py

What this notebook does:
    1. Loads the dataset to make sure the paths are correct.
    2. Prints the shape and the first few rows.
    3. Helps confirm that nothing broke after cleaning or moving files.

Why this notebook exists:
    This is basically a sanity check. If the dataset doesn’t load here,
    then the later modeling notebooks won’t work either. It lets me catch
    path issues early instead of debugging them during Sprint 3.
"""

from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
pri = pd.read_csv(RAW / "social_media_addiction_vs_relationships.csv")
sec = pd.read_csv(RAW / "social_media_vs_productivity.csv")
print("Primary:", pri.shape)
print("Secondary:", sec.shape)
print("Primary columns:", list(pri.columns)[:30])
print("Secondary columns:", list(sec.columns)[:30])