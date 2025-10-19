from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path.cwd()
CLEAN = ROOT/"data"/"clean"
RAW   = ROOT/"data"/"raw"
REPORTS = ROOT/"reports"; REPORTS.mkdir(parents=True, exist_ok=True)

dfp = pd.read_csv(CLEAN/"primary_clean.csv")
dfs = pd.read_csv(RAW/"social_media_vs_productivity.csv")

rename = {
    "daily_social_media_time":"hours_social_media",
    "sleep_hours":"sleep_hours",
    "perceived_productivity_score":"prod_perceived",
    "actual_productivity_score":"prod_actual",
    "social_platform_preference":"platform_pref",
    "number_of_notifications":"notifications",
    "work_hours_per_day":"work_hours",
}
dfs = dfs.rename(columns={k:v for k,v in rename.items() if k in dfs.columns})

def miss_pct(v):
    if v in dfp.columns: return round(100*(1-dfp[v].notna().mean()),1)
    if v in dfs.columns: return round(100*(1-dfs[v].notna().mean()),1)
    return np.nan

rows = [
    ["hours_social_media","Continuous","Daily social media hours","Independent (main)","Primary", f"{miss_pct('hours_social_media')}%"],
    ["acad_impact","Binary","Self-reported academic impact (Yes=1/No=0)","Dependent Variable","Primary", f"{miss_pct('acad_impact')}%"],
    ["sleep_hours","Continuous","Average nightly sleep (hours)","Control","Primary", f"{miss_pct('sleep_hours')}%"],
    ["addiction_score","Continuous","Addiction/self-control index","Control/Mechanism","Primary", f"{miss_pct('addiction_score')}%"],
    ["platform_group","Categorical","Platform type (Short-video/Image-centric/Forum/Other)","Independent","Primary", f"{miss_pct('platform_group')}%"],
    ["heavy_user","Binary","1 if hours ≥ 75th pct","Derived/Moderator","Primary", f"{miss_pct('heavy_user')}%"],
    ["sleep_ok","Binary","1 if sleep ≥ 7h","Derived/Control","Primary", f"{miss_pct('sleep_ok')}%"],
    ["prod_perceived","Continuous","Perceived productivity score","Dependent Variable","Secondary", f"{miss_pct('prod_perceived')}%"],
    ["prod_actual","Continuous","Actual productivity score","Dependent Variable","Secondary", f"{miss_pct('prod_actual')}%"],
    ["platform_pref","Categorical","Most-used platform (secondary)","Control","Secondary", f"{miss_pct('platform_pref')}%"],
    ["notifications","Continuous","Daily notifications","Control","Secondary", f"{miss_pct('notifications')}%"],
    ["work_hours","Continuous","Work hours per day","Control","Secondary", f"{miss_pct('work_hours')}%"],
]
cols = ["Variable","Type","Description","Role","Present In","Missing %"]
inv = pd.DataFrame(rows, columns=cols)

(inv.sort_values(["Present In","Role","Variable"])
   .to_csv(REPORTS/"Variable_inventory_qssr.csv", index=False))

with open(REPORTS/"Variable_inventory_qssr.md","w") as f:
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
    for _,r in inv.iterrows():
        f.write("| " + " | ".join(str(x) for x in r.tolist()) + " |\n")

print("✅ Wrote reports/Variable_inventory_qssr.csv and .md")
