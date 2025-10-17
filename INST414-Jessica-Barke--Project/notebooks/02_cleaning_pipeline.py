from pathlib import Path
import pandas as pd, numpy as np

ROOT   = Path.cwd()
RAW    = ROOT/"data"/"raw"
CLEAN  = ROOT/"data"/"clean"
REPORT = ROOT/"reports"
CLEAN.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

log = []
def add(msg):
    print(msg); log.append(msg)

# --- load primary raw ---
src = RAW / "social_media_addiction_vs_relationships.csv"
df0 = pd.read_csv(src)
add(f"Loaded raw: {src} | shape={df0.shape}")
add("Raw columns: " + ", ".join(df0.columns))

# === EXACT column names from YOUR dataset ===
ALIASES = {
    "hours_social_media": ["Avg_Daily_Usage_Hours"],
    "sleep_hours":        ["Sleep_Hours_Per_Night"],
    "platform_primary":   ["Most_Used_Platform"],
    "acad_impact":        ["Affects_Academic_Performance"],  # DV for Sprint 2
    "addiction_score":    ["Addicted_Score"],
}

def pick(df, names):
    for n in names:
        if n in df.columns: 
            return n
    return None

def to_num(series):
    if series is None:
        return pd.Series(np.nan, index=range(len(df0)))
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    # extract numeric substring like "3.5 hrs", "~4"
    extr = series.astype(str).str.extract(r"(-?\d+\.?\d*)", expand=False)
    return pd.to_numeric(extr, errors="coerce")

out = pd.DataFrame(index=df0.index)

# Map real columns -> standard names
h_col = pick(df0, ALIASES["hours_social_media"])
s_col = pick(df0, ALIASES["sleep_hours"])
p_col = pick(df0, ALIASES["platform_primary"])
y_col = pick(df0, ALIASES["acad_impact"])
a_col = pick(df0, ALIASES["addiction_score"])

add(f"Map hours_social_media <- {h_col}")
add(f"Map sleep_hours       <- {s_col}")
add(f"Map platform_primary  <- {p_col}")
add(f"Map acad_impact (DV)  <- {y_col}")
add(f"Map addiction_score   <- {a_col}")

out["hours_social_media"] = to_num(df0[h_col]) if h_col else np.nan
out["sleep_hours"]        = to_num(df0[s_col]) if s_col else np.nan
out["platform_primary"]   = df0[p_col] if p_col else pd.Series(np.nan, index=df0.index)
out["acad_impact"]        = to_num(df0[y_col]) if y_col else np.nan
out["addiction_score"]    = to_num(df0[a_col]) if a_col else np.nan

# no GPA in this dataset (placeholder to keep scripts happy)
out["gpa"] = np.nan

# sanity bounds (don't drop rows)
out.loc[(out["hours_social_media"]<0) | (out["hours_social_media"]>24), "hours_social_media"] = np.nan
out.loc[(out["sleep_hours"]<0) | (out["sleep_hours"]>24), "sleep_hours"] = np.nan
out.loc[(out["acad_impact"]<0), "acad_impact"] = np.nan

# DO NOT drop duplicates yet (we'll inspect first)
# prev = len(out); out = out.drop_duplicates(); add(f"drop_duplicates removed: {prev-len(out)}")

# median impute only if we have at least one non-null
for c in ["hours_social_media","sleep_hours","acad_impact","addiction_score"]:
    if out[c].notna().any():
        out[c] = out[c].fillna(out[c].median())

# features
if out["hours_social_media"].notna().any():
    cut = out["hours_social_media"].quantile(0.75)
    out["heavy_user"] = (out["hours_social_media"] >= cut).astype("Int64")
else:
    out["heavy_user"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

if out["sleep_hours"].notna().any():
    out["sleep_ok"] = (out["sleep_hours"] >= 7).astype("Int64")
else:
    out["sleep_ok"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

def map_platform(x):
    if pd.isna(x): return "Other"
    s = str(x).lower()
    if any(k in s for k in ["tiktok","short","reels","youtube"]): return "Short-video"
    if any(k in s for k in ["instagram","insta","snap","pinterest"]): return "Image-centric"
    if any(k in s for k in ["linkedin","reddit","quora","discord"]): return "Professional/Forum"
    return "Other"

out["platform_group"] = out["platform_primary"].map(map_platform)

# diagnostics BEFORE saving
for c in ["hours_social_media","sleep_hours","acad_impact","addiction_score"]:
    nz = int(out[c].notna().sum())
    mean = float(out[c].dropna().mean()) if nz>0 else float("nan")
    add(f"{c} non-null: {nz} | mean={mean:.3f}")

dst = CLEAN / "primary_clean.csv"
out.to_csv(dst, index=False)
add(f"Saved {dst} | shape={out.shape}")

with open(REPORT/"cleaning_log.md","w") as f:
    f.write("\n".join(log))

print("✅ Cleaning done. Check reports/cleaning_log.md")
