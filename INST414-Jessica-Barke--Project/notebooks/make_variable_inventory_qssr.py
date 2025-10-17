from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

pri = pd.read_csv(RAW / "social_media_addiction_vs_relationships.csv")
sec = pd.read_csv(RAW / "social_media_vs_productivity.csv")

ALIASES = {
    "hours_social_media":  ["Time spent on social media (hours)", "hours_social_media"],
    "gpa":                 ["Grade", "gpa", "cgpa"],
    "sleep_hours":         ["Sleep (hours)", "sleep_hours"],
    "study_hours":         ["study_hours", "Study time (hours)"],
    "work_hours":          ["work_hours", "employment_hours"],
    "platform_primary":    ["Most used platform", "platform_primary"],
    "productivity_index":  ["productivity_score", "Productivity score"],
    "addiction_score":     ["Social Media Addiction", "addiction_score"],
}
def present_in(name):
    def has(df, cols): return any(c in df.columns for c in cols)
    p, s = has(pri, ALIASES[name]), has(sec, ALIASES[name])
    return "Both" if (p and s) else ("Primary" if p else ("Secondary" if s else "—"))

rows = [
    ("gpa","Continuous","GPA or academic score (0–4 normalized)","Dependent Variable","Primary"),
    ("hours_social_media","Continuous","Self-reported hours per day on social media","Independent Variable (Main)",present_in("hours_social_media")),
    ("platform_primary","Categorical","Most-used social media platform","Independent Variable","Primary"),
    ("sleep_hours","Continuous","Average nightly sleep duration (hours)","Control Variable","Primary"),
    ("study_hours","Continuous","Average daily study time (hours)","Control Variable","Primary"),
    ("work_hours","Continuous","Weekly paid work hours","Control Variable","Primary"),
    ("productivity_index","Continuous","Self-rated productivity or efficiency score","Secondary Outcome (Validation)",present_in("productivity_index")),
    ("addiction_score","Continuous","Social media addiction/self-control index","Supplementary Variable","Primary"),
]
df = pd.DataFrame(rows, columns=["Variable","Type","Description","Role","Present In"])
df.to_csv(REPORTS / "variable_inventory_qssr.csv", index=False)
with open(REPORTS / "variable_inventory_qssr.md","w",encoding="utf-8") as f:
    f.write("| Variable | Type | Description | Role | Present In |\n|---|---|---|---|---|\n")
    for _, r in df.iterrows():
        f.write(f"| {r['Variable']} | {r['Type']} | {r['Description']} | {r['Role']} | {r['Present In']} |\n")
print("Wrote reports/variable_inventory_qssr.[csv|md]")
