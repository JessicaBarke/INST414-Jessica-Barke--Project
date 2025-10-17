from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path.cwd()
RAW, CLEAN, REPORTS = ROOT/"data"/"raw", ROOT/"data"/"clean", ROOT/"reports"
CLEAN.mkdir(parents=True, exist_ok=True)

log = []
def add_log(x): print(x); log.append(x)

pri_raw = pd.read_csv(RAW / "social_media_addiction_vs_relationships.csv")
add_log(f"Start N: {len(pri_raw)}")

ALIASES = {
    'hours_social_media':  ['Time spent on social media (hours)','hours_social_media','daily_usage_hours','time_spent_daily'],
    'gpa':                 ['Grade','gpa','cgpa','academic_score'],
    'study_hours':         ['study_hours','Study time (hours)','daily_study_hours','study_time'],
    'sleep_hours':         ['Sleep (hours)','sleep_hours','sleep_time'],
    'work_hours':          ['work_hours','employment_hours','weekly_work_hours'],
    'platform_primary':    ['Most used platform','platform_primary','primary_platform','most_used_platform'],
}
def pick(df, opts):
    for c in opts:
        if c in df.columns: return c
    return None
def to_num(s):
    return pd.to_numeric(s.replace(r"[^0-9\.\-]", "", regex=True), errors="coerce") if s.dtype==object else pd.to_numeric(s, errors="coerce")

out = pd.DataFrame()
# numeric fields
for k in ['hours_social_media','study_hours','sleep_hours','work_hours']:
    col = pick(pri_raw, ALIASES[k]); out[k] = to_num(pri_raw[col]) if col else np.nan

# GPA normalize to 0–4 /10 /100 fallback
gcol = pick(pri_raw, ALIASES['gpa'])
graw = to_num(pri_raw[gcol]) if gcol else pd.Series([np.nan]*len(pri_raw))
if graw.dropna().quantile(0.95) <= 4.5:    g = graw.clip(0,4)
elif graw.dropna().quantile(0.95) <= 10:  g = (graw/10*4).clip(0,4)
elif graw.dropna().quantile(0.95) <= 100: g = (graw/100*4).clip(0,4)
else:                                      g = (graw - graw.mean())/graw.std(ddof=0)
out['gpa'] = g

# categorical
pcol = pick(pri_raw, ALIASES['platform_primary'])
out['platform_primary'] = pri_raw[pcol] if pcol else pd.Series([np.nan]*len(pri_raw))

# duplicates
prev = len(out); out = out.drop_duplicates(); add_log(f"drop_duplicates removed: {prev-len(out)}")

# invalid values
if 'hours_social_media' in out:
    out.loc[(out['hours_social_media']<0)|(out['hours_social_media']>24),'hours_social_media'] = np.nan

# median imputation
for c in ['hours_social_media','gpa','sleep_hours','study_hours','work_hours']:
    if c in out.columns: out[c] = out[c].fillna(out[c].median())

# features
cut = out['hours_social_media'].quantile(0.75)
out['heavy_user'] = (out['hours_social_media'] >= cut).astype(int)
out['sleep_ok']   = (out['sleep_hours'] >= 7).astype('Int64')

def map_platform(x):
    if pd.isna(x): return 'Other'
    s = str(x).lower()
    if any(k in s for k in ['tiktok','short','reels','youtube shorts']): return 'Short-video'
    if any(k in s for k in ['instagram','snap','pinterest']):            return 'Image-centric'
    if any(k in s for k in ['linkedin','reddit','quora']):               return 'Professional/Forum'
    return 'Other'
out['platform_group'] = out['platform_primary'].map(map_platform)

# save
out.to_csv(CLEAN / "primary_clean.csv", index=False)
add_log(f"Saved data/clean/primary_clean.csv | N={len(out)}")
with open(REPORTS / "cleaning_log.md","w") as f: f.write("\n".join(log))
print("Wrote reports/cleaning_log.md")
