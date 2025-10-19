from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
CLEAN = ROOT/"data"/"clean"
REPORTS = ROOT/"reports"
FIGS = REPORTS/"figures"
for d in (CLEAN, REPORTS, FIGS): d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN/"primary_clean.csv")
print("Loaded:", df.shape, "cols:", df.columns.tolist())

keep = [c for c in ["hours_social_media","acad_impact","sleep_hours","addiction_score"] if c in df.columns]
df[keep].describe().T.round(3).to_csv(REPORTS/"summary_stats.csv")

corr = df[keep].corr().round(3)
corr.to_csv(REPORTS/"correlations.csv")

# 1) Hours hist
plt.hist(df["hours_social_media"].dropna(), bins=30, edgecolor="black")
plt.title("Distribution of Daily Social Media Hours")
plt.xlabel("Hours per day"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(FIGS/"fig1_hours_hist.png"); plt.close()

# 2) Academic impact hist
plt.hist(df["acad_impact"].dropna(), bins=20, edgecolor="black")
plt.title("Distribution of Academic Impact (Yes=1, No=0)")
plt.xlabel("Academic impact"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig(FIGS/"fig2_acad_impact_hist.png"); plt.close()

# 3) Scatter + trend
sns.scatterplot(data=df, x="hours_social_media", y="acad_impact", alpha=0.4)
sns.regplot(data=df, x="hours_social_media", y="acad_impact", scatter=False, color="darkred", line_kws={"lw":2})
plt.title("Academic Impact vs Social Media Hours"); plt.tight_layout()
plt.savefig(FIGS/"fig3_scatter_impact_vs_hours.png"); plt.close()

# 4) Boxplot by platform group
if "platform_group" in df.columns:
    sns.boxplot(data=df, x="platform_group", y="acad_impact")
    plt.title("Academic Impact by Platform Group"); plt.xlabel("Platform group"); plt.ylabel("Academic impact")
    plt.tight_layout(); plt.savefig(FIGS/"fig4_impact_by_platform.png"); plt.close()

# 5) Correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix"); plt.tight_layout()
plt.savefig(FIGS/"fig5_corr.png"); plt.close()

# 6) Group means + CI
if all(c in df.columns for c in ["heavy_user","sleep_ok","acad_impact"]):
    grp = df.groupby(["heavy_user","sleep_ok"])["acad_impact"].agg(["mean","count","std"]).reset_index()
    se = grp["std"]/np.sqrt(grp["count"].clip(lower=1))
    grp["ci_lo"] = grp["mean"] - 1.96*se
    grp["ci_hi"] = grp["mean"] + 1.96*se
    grp.to_csv(REPORTS/"group_means_impact_by_heavy_sleep.csv", index=False)

    plt.errorbar(range(len(grp)), grp["mean"], yerr=1.96*se, fmt="o")
    plt.xticks(range(len(grp)), [f"heavy={int(h)}|sleep={int(s)}" for h,s in zip(grp["heavy_user"], grp["sleep_ok"])], rotation=30)
    plt.title("Mean Academic Impact (95% CI) by Heavy Use × Sleep OK")
    plt.xlabel("Group"); plt.ylabel("Mean impact"); plt.tight_layout()
    plt.savefig(FIGS/"fig6_group_means_ci.png"); plt.close()

print("✅ EDA complete")
