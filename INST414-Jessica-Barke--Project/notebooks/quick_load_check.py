from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
pri = pd.read_csv(RAW / "social_media_addiction_vs_relationships.csv")
sec = pd.read_csv(RAW / "social_media_vs_productivity.csv")
print("Primary:", pri.shape)
print("Secondary:", sec.shape)
print("Primary columns:", list(pri.columns)[:30])
print("Secondary columns:", list(sec.columns)[:30])