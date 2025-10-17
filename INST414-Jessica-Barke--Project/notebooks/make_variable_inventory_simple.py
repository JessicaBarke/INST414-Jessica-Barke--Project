from pathlib import Path
import pandas as pd

ROOT = Path.cwd(); RAW = ROOT / "data" / "raw"; REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

pri = pd.read_csv(RAW / "social_media_addiction_vs_relationships.csv")
sec = pd.read_csv(RAW / "social_media_vs_productivity.csv")

ALIASES = {
    "hours_social_media":  ["Time spent on social media (hours)", "hours_social_media"],
    "gpa":                 ["Grade","gpa","cgpa"],
    "productivity_index":  ["productivity_score","Productivity score","productivity_index"],
    "sleep_hours":         ["Sleep (hours)","sleep_hours"],
    "study_hours":         ["study_hours","Study time (hours)"],
    "work_hours":          ["work_hours","employment_hours"],
    "platform_primary":    ["Most used platform","platform_primary"],
}
SCHEMA = [
    ("hours_social_media","Continuous","Self reported hours per day on social media","Key"),
    ("gpa","Continuous","GPA or academic score proxy","Key"),
    ("productivity_index","Continuous","Productivity / self efficacy score","Key"),
    ("sleep_hours","Continuous","Average nightly sleep","Control"),
    ("study_hours","Continuous","Average study time","Control"),
    ("platform_primary","Categorical","Most used platform","Feature"),
    ("work_hours","Continuous","Weekly paid work hours","Control"),
]
def present_in(name):
    pin = any(c in pri.columns for c in ALIASES.get(name, []))
    sin = any(c in sec.columns for c in ALIASES.get(name, []))
    return "Both" if (pin and sin) else ("Primary" if pin else ("Secondary" if sin else "—"))
rows = [{"Variable":v,"Type":t,"Description":d,"Relevance":r,"Present In":present_in(v)} for v,t,d,r in SCHEMA]
df = pd.DataFrame(rows)[["Variable","Type","Description","Relevance","Present In"]]
df.to_csv(REPORTS / "variable_inventory_simple.csv", index=False)
with open(REPORTS / "variable_inventory_simple.md","w",encoding="utf-8") as f:
    f.write("| Variable | Type | Description | Relevance | Present In |\n|---|---|---|---|---|\n")
    for _, r in df.iterrows():
        f.write(f"| {r['Variable']} | {r['Type']} | {r['Description']} | {r['Relevance']} | {r['Present In']} |\n")
print("Wrote reports/variable_inventory_simple.[csv|md]")
