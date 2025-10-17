from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
CLEAN, REPORTS, FIGS = ROOT/"data"/"clean", ROOT/"reports", ROOT/"reports"/"figures"
FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN / "primary_clean.csv")

# Table: summary stats
keep = [c for c in ['hours_social_media','gpa','sleep_hours','study_hours','work_hours'] if c in df.columns]
df[keep].describe().T.round(3).to_csv(REPORTS / "summary_stats.csv")

# Fig 1: hours hist
ax = df['hours_social_media'].plot(kind='hist', bins=30, title='Distribution of Daily Social Media Hours')
ax.set_xlabel('Hours per day'); ax.set_ylabel('Count'); plt.tight_layout()
plt.savefig(FIGS / "fig1_hours_hist.png"); plt.close()

# Fig 2: GPA hist
ax = df['gpa'].plot(kind='hist', bins=20, title='Distribution of GPA (0–4)')
ax.set_xlabel('GPA'); ax.set_ylabel('Count'); plt.tight_layout()
plt.savefig(FIGS / "fig2_gpa_hist.png"); plt.close()

# Fig 3: GPA vs hours + smoother
x, y = df['hours_social_media'], df['gpa']
plt.scatter(x, y, alpha=0.35)
plt.title('GPA vs Daily Social Media Hours'); plt.xlabel('Hours per day'); plt.ylabel('GPA')
order = np.argsort(x); xo, yo = x.iloc[order], y.iloc[order]
y_smooth = yo.rolling(window=max(5,len(x)//20), min_periods=1).mean()
plt.plot(xo, y_smooth, linewidth=2); plt.tight_layout()
plt.savefig(FIGS / "fig3_scatter_smooth.png"); plt.close()

# Fig 4: GPA by platform
if 'platform_group' in df.columns:
    df.boxplot(column='gpa', by='platform_group', grid=False)
    plt.title('GPA by Platform Group'); plt.suptitle('')
    plt.xlabel('Platform Group'); plt.ylabel('GPA')
    plt.tight_layout(); plt.savefig(FIGS / "fig4_gpa_by_platform.png"); plt.close()

# Table + Fig 5: correlations
cont = [c for c in ['hours_social_media','gpa','sleep_hours','study_hours','work_hours'] if c in df.columns]
if len(cont) >= 2:
    corr = df[cont].corr().round(3)
    corr.to_csv(REPORTS / "correlations.csv")
    fig, ax = plt.subplots(); im = ax.imshow(corr, interpolation='nearest')
    ax.set_title('Correlation Matrix (Continuous Variables)')
    ax.set_xticks(range(len(cont))); ax.set_xticklabels(cont, rotation=45, ha='right')
    ax.set_yticks(range(len(cont))); ax.set_yticklabels(cont)
    fig.colorbar(im); plt.tight_layout(); plt.savefig(FIGS / "fig5_corr.png"); plt.close()

# Table + Fig 6: group means (heavy x sleep)
if all(c in df.columns for c in ['heavy_user','sleep_ok','gpa']):
    grp = df.groupby(['heavy_user','sleep_ok'])['gpa'].agg(['mean','count','std']).reset_index()
    se = grp['std']/np.sqrt(grp['count'])
    grp.assign(se=se, ci_lo=grp['mean']-1.96*se, ci_hi=grp['mean']+1.96*se)\
       .to_csv(REPORTS / "group_means_gpa_by_heavy_sleep.csv", index=False)
    plt.errorbar(range(len(grp)), grp['mean'], yerr=1.96*se, fmt='o')
    cats = [f"heavy={int(h)} | sleep={int(s)}" for h,s in zip(grp['heavy_user'], grp['sleep_ok'])]
    plt.xticks(range(len(grp)), cats, rotation=30, ha='right')
    plt.title('Mean GPA (95% CI) by Heavy Use and Sleep Adequacy')
    plt.xlabel('Group'); plt.ylabel('GPA'); plt.tight_layout()
    plt.savefig(FIGS / "fig6_group_means_ci.png"); plt.close()

print("Saved EDA tables & figures to reports/")
