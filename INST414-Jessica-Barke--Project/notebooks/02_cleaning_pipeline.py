from pathlib import Path
import pandas as pd, numpy as np

ROOT   = Path.cwd()
RAW    = ROOT/"data"/"raw"
CLEAN  = ROOT/"data"/"clean"
REPORT = ROOT/"reports"
CLEAN.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

log = []
def add(msg): print(msg); log.append(msg)

# Load primary
src = RAW / "social_media_addiction_vs_relationships.csv"
df0 = pd.read_csv(src)
add(f"Loaded raw: {src} | shape={df0.shape}")
add("Raw columns: " + ", ".join(df0.columns))

# Column map (from your dataset)
HOURS_COL = "Avg_Daily_Usage_Hours"
SLEEP_COL = "Sleep_Hours_Per_Night"
PLAT_COL  = "Most_Used_Platform"
DV_COL    = "Affects_Academic_Performance"   # "Yes"/"No"
ADD_COL   = "Addicted_Score"

def to_num(series):
    if pd.api.types.is_numeric_dtype(series): 
        return pd.to_numeric(series, errors="coerce")
    extr = series.astype(str).str.extract(r"(-?\d+\.?\d*)", expand=False)
    return pd.to_numeric(extr, errors="coerce")

out = pd.DataFrame(index=df0.index)
out["hours_social_media"] = to_num(df0[HOURS_COL])
out["sleep_hours"]        = to_num(df0[SLEEP_COL])
out["platform_primary"]   = df0[PLAT_COL]
out["addiction_score"]    = to_num(df0[ADD_COL])

# === Map Yes/No -> 1/0 for the DV
s = df0[DV_COL].astype(str).str.strip().str.lower()
out["acad_impact"] = s.map({"yes": 1.0, "no": 0.0})

# Basic sanity bounds (keep rows; null invalids)
out.loc[(out["hours_social_media"]<0) | (out["hours_social_media"]>24), "hours_social_media"] = np.nan
out.loc[(out["sleep_hours"]<0) | (out["sleep_hours"]>24), "sleep_hours"] = np.nan

# Impute medians when possible
for c in ["hours_social_media","sleep_hours","acad_impact","addiction_score"]:
    if out[c].notna().any():
        out[c] = out[c].fillna(out[c].median())

# Engineered features
q75 = out["hours_social_media"].quantile(0.75) if out["hours_social_media"].notna().any() else np.nan
out["heavy_user"] = (out["hours_social_media"] >= q75).astype("Int64") if pd.notna(q75) else pd.Series(pd.NA, index=out.index, dtype="Int64")
out["sleep_ok"]   = (out["sleep_hours"] >= 7).astype("Int64") if out["sleep_hours"].notna().any() else pd.Series(pd.NA, index=out.index, dtype="Int64")

def map_platform(x):
    if pd.isna(x): return "Other"
    s = str(x).lower()
    if any(k in s for k in ["tiktok","short","reels","youtube"]): return "Short-video"
    if any(k in s for k in ["instagram","insta","snap","pinterest"]): return "Image-centric"
    if any(k in s for k in ["linkedin","reddit","quora","discord"]): return "Professional/Forum"
    return "Other"
out["platform_group"] = out["platform_primary"].map(map_platform)

# Diagnostics
for c in ["hours_social_media","sleep_hours","acad_impact","addiction_score"]:
    nz = int(out[c].notna().sum())
    m  = float(out[c].dropna().mean()) if nz>0 else float("nan")
    add(f"{c} non-null: {nz} | mean={m:.3f}")

# Save + log
dst = CLEAN / "primary_clean.csv"
out.to_csv(dst, index=False)
add(f"Saved {dst} | shape={out.shape}")
with open(REPORT/"cleaning_log.md","w") as f: f.write("\n".join(log))
print("✅ Cleaning done. See reports/cleaning_log.md")
