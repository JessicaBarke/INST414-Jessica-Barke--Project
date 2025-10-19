from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()
CLEAN = ROOT / "data" / "clean"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Load primary (cleaned)
dfp = pd.read_csv(CLEAN / "primary_clean.csv")

# Load secondary (raw) and align some names so we can compute missingness
sec_path = RAW / "social_media_vs_productivity.csv"
dfs = pd.read_csv(sec_path)

sec_rename = {
    "daily_social_media_time": "hours_social_media",
    "sleep_hours": "sleep_hours",
    "perceived_productivity_score": "prod_perceived",
    "actual_productivity_score": "prod_actual",
    "social_platform_preference": "platform_pref",
    "number_of_notifications": "notifications",
    "work_hours_per_day": "work_hours",
}
dfs = dfs.rename(columns={k:v for k,v in sec_rename.items() if k in dfs.columns})

def missing_pct(var):
    """Return missing% using whichever dataset has that standardized column."""
    if var in dfp.columns:
        return round(100 * (1 - dfp[var].notna().mean()), 1)
    if var in dfs.columns:
        return round(100 * (1 - dfs[var].notna().mean()), 1)
    return np.nan

rows = []

# ---- Define variables for QSSR inventory ----
# Variable, Type, Description, Role, Present In
items = [
    ("hours_social_media",  "Continuous", "Self-reported hours per day on social media",                    "Independent (main)",        "Both"),
    ("acad_impact",         "Binary",     "Self-reported academic impact (Yes=1, No=0)",                    "Dependent Variable",        "Primary"),
    ("sleep_hours",         "Continuous", "Average nightly sleep (hours)",                                  "Control",                   "Primary"),
    ("addiction_score",     "Continuous", "Social media addiction/self-control index",                      "Control / Mechanism",       "Primary"),
    ("platform_group",      "Categorical","Collapsed platform type (Short-video / Image-centric / Forum)",  "Independent",               "Primary"),
    ("heavy_user",          "Binary",     "1 if hours >= 75th percentile",                                  "Derived / Moderator",       "Primary"),
    ("sleep_ok",            "Binary",     "1 if sleep_hours >= 7",                                          "Derived / Control",         "Primary"),
    # Secondary-only outcomes & controls (for productivity analysis)
    ("prod_perceived",      "Continuous", "Perceived productivity score",                                   "Dependent Variable",        "Secondary"),
    ("prod_actual",         "Continuous", "Actual productivity score",                                      "Dependent Variable",        "Secondary"),
    ("notifications",       "Continuous", "Number of notifications per day",                                "Control",                   "Secondary"),
    ("work_hours",          "Continuous", "Work hours per day",                                             "Control",                   "Secondary"),
    ("platform_pref",       "Categorical","Most-used platform (secondary dataset)",                         "Control",                   "Secondary"),
]

for var, typ, desc, role, present in items:
    miss = missing_pct(var)
    rows.append([var, typ, desc, role, present, f"{miss}%" if pd.notna(miss) else "—"])

cols = ["Variable","Type","Description","Role","Present In","Missing %"]
df = pd.DataFrame(rows, columns=cols)
csv_path = REPORTS / "Variable_inventory_qssr.csv"
md_path  = REPORTS / "Variable_inventory_qssr.md"
df.to_csv(csv_path, index=False)

# Write a pretty Markdown table for pasting into your PDF/Doc
with open(md_path, "w") as f:
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
    for _, r in df.iterrows():
        f.write("| " + " | ".join(str(x) for x in r.tolist()) + " |\n")

print(f"✅ Wrote {csv_path}")
print(f"✅ Wrote {md_path}")
