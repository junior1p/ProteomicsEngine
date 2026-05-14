
import os, sys, json, warnings, time, math
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from collections import defaultdict

OUT = "proteomics_output"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
np.random.seed(42)

N_PROTEINS = 1200
N_SAMPLES = 6

PROTEIN_NAMES = [
    "EGFR","TP53","KRAS","MYC","PTEN","AKT1","MTOR","VEGFA","CDK2","CDK4",
    "RB1","MDM2","BCL2","STAT3","NF1","VHL","CDKN2A","BRCA1","BRCA2","PIK3CA",
    "MAPK1","MAPK3","RAF1","BRAF","MEK1","MEK2","ERK1","ERK2","JNK1","JNK2",
    "CASP3","CASP8","CASP9","BCL2L1","BAX","BAD","BID","PUMA","NOXA","MCL1",
    "HSPA1A","HSP90AA1","HSPA5","CALR","PDIA3","GRP78","ATF6","IRE1","PERK","GRP94",
    "ACTIN","TUBULIN","GAPDH","LDHA","PKM","ENO1","PGK1","TPI1","ALDOA","GPI",
    "FASN","ACC1","ACLY","HMGCR","SQLE","FDFT1","LSS","CYP51","SC4MOL","DHCR7",
    "COX1","COX2","COX4","COX5","ATP5A","ATP5B","NDUF1","SDHA","SDHB","FH",
    "IDH1","IDH2","IDH3","OGDH","SUCLA2","SUCLG1","MDH1","MDH2","CS","ACO2",
    "PCNA","MCM2","MCM4","MCM6","CDC6","CDT1","ORC1","RPA1","RFC1","LIG1",
    "SF3B1","U2AF1","SRSF1","SRSF2","HNRNPA1","HNRNPC","PTBP1","ELAVL1","FUS","TDP43",
    "EIF4E","EIF4G","EIF4A","EIF2A","EIF3A","PABP","RPL5","RPL11","RPS6","RPS14",
]
while len(PROTEIN_NAMES) < N_PROTEINS:
    PROTEIN_NAMES.append(f"PROT{len(PROTEIN_NAMES):04d}")
PROTEIN_NAMES = PROTEIN_NAMES[:N_PROTEINS]

print("[ProteomicsEngine] Generating synthetic DDA proteomics data...")

# True log2 abundances
true_ctrl = np.random.normal(10, 2, N_PROTEINS)
# Case: ~25% proteins differentially abundant
diff_idx = np.random.choice(N_PROTEINS, N_PROTEINS // 4, replace=False)
effects = np.random.normal(0, 2.0, len(diff_idx))
true_case = true_ctrl.copy()
for i, eff in zip(diff_idx, effects):
    true_case[i] += eff

# Quantification matrix (log2 intensity)
quant_matrix = np.zeros((N_PROTEINS, N_SAMPLES))
for s in range(N_SAMPLES):
    is_case = s >= 3
    true_abund = true_case if is_case else true_ctrl
    # Add technical noise
    noise = np.random.normal(0, 0.3, N_PROTEINS)
    # Missing values: ~20% proteins not detected per sample
    detected = np.random.random(N_PROTEINS) > 0.2
    quant_matrix[:, s] = np.where(detected, true_abund + noise, 0)

print(f"  {N_PROTEINS} proteins × {N_SAMPLES} samples")
print(f"  Mean proteins detected per sample: {(quant_matrix > 0).sum(axis=0).mean():.0f}")

# Simulate PSMs
N_PSMS = 8432
psm_proteins = np.random.choice(N_PROTEINS, N_PSMS)
hyperscores = np.random.exponential(4, N_PSMS) + 2
psm_df = pd.DataFrame({"protein_idx": psm_proteins, "hyperscore": hyperscores})
psm_filtered = psm_df[psm_df["hyperscore"] > 3.5]
identified = psm_filtered["protein_idx"].unique()
print(f"  PSMs: {len(psm_df)} total, {len(psm_filtered)} passing (score>3.5)")
print(f"  Proteins identified: {len(identified)}")

# Differential abundance
print("[ProteomicsEngine] Differential abundance analysis...")
case_cols = list(range(3, 6))
ctrl_cols = list(range(0, 3))

da_results = []
for i in range(N_PROTEINS):
    case_vals = quant_matrix[i, case_cols]
    ctrl_vals = quant_matrix[i, ctrl_cols]
    if (case_vals > 0).sum() < 2 or (ctrl_vals > 0).sum() < 2:
        continue
    log2fc = case_vals[case_vals > 0].mean() - ctrl_vals[ctrl_vals > 0].mean()
    _, pval = stats.ttest_ind(case_vals[case_vals > 0], ctrl_vals[ctrl_vals > 0])
    da_results.append({"protein": PROTEIN_NAMES[i], "log2FC": round(log2fc, 4), "p_value": pval})

da_df = pd.DataFrame(da_results).sort_values("p_value")
n = len(da_df)
da_df["rank"] = range(1, n+1)
da_df["padj"] = np.minimum(da_df["p_value"] * n / da_df["rank"], 1.0)
sig_df = da_df[(da_df["padj"] < 0.05) & (da_df["log2FC"].abs() > 1.0)]
print(f"  Significant proteins (FDR<0.05, |log2FC|>1): {len(sig_df)}")
if len(sig_df):
    top = sig_df.iloc[0]
    print(f"  Top: {top['protein']} (log2FC={top['log2FC']:.2f}, FDR={top['padj']:.4f})")
da_df.to_csv(f"{OUT}/differential_proteins.csv", index=False)

# GO enrichment
GO_TERMS = {
    "signal transduction": ["EGFR","KRAS","BRAF","RAF1","MEK1","MEK2","ERK1","ERK2","AKT1","MTOR","PTEN","PIK3CA","STAT3","MAPK1","MAPK3"],
    "apoptosis": ["CASP3","CASP8","CASP9","BCL2","BCL2L1","BAX","BAD","BID","PUMA","NOXA","MCL1","TP53"],
    "DNA replication": ["PCNA","MCM2","MCM4","MCM6","CDC6","CDT1","ORC1","RPA1","RFC1","LIG1"],
    "glycolysis": ["LDHA","PKM","ENO1","PGK1","TPI1","ALDOA","GPI","GAPDH"],
    "protein folding": ["HSPA1A","HSP90AA1","HSPA5","CALR","PDIA3","GRP78"],
    "RNA splicing": ["SF3B1","U2AF1","SRSF1","SRSF2","HNRNPA1","HNRNPC","PTBP1","ELAVL1"],
    "translation": ["EIF4E","EIF4G","EIF4A","EIF2A","EIF3A","PABP","RPL5","RPL11","RPS6","RPS14"],
    "TCA cycle": ["IDH1","IDH2","IDH3","OGDH","SUCLA2","MDH1","MDH2","CS","ACO2"],
    "cell cycle": ["CDK2","CDK4","RB1","CDKN2A","PCNA","MCM2","CDC6"],
    "fatty acid synthesis": ["FASN","ACC1","ACLY","HMGCR"],
}
sig_proteins = set(sig_df["protein"].tolist())
all_proteins_set = set(PROTEIN_NAMES)
go_results = []
for term, members in GO_TERMS.items():
    members_in_data = [m for m in members if m in all_proteins_set]
    sig_in_term = [m for m in members_in_data if m in sig_proteins]
    k = len(sig_in_term); M = len(all_proteins_set); K = len(sig_proteins); n = len(members_in_data)
    if n < 2: continue
    pval = stats.hypergeom.sf(k-1, M, K, n)
    go_results.append({"term": term, "n_members": n, "n_sig": k, "p_value": round(pval, 6)})
go_df = pd.DataFrame(go_results).sort_values("p_value")
print(f"  Enriched GO terms (p<0.05): {(go_df['p_value']<0.05).sum()}")
if len(go_df): print(f"  Top: {go_df.iloc[0]['term']} (p={go_df.iloc[0]['p_value']:.4f})")
go_df.to_csv(f"{OUT}/go_enrichment.csv", index=False)

# Dashboard
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("ProteomicsEngine: DDA Proteomics Analysis\n"
             f"({N_SAMPLES} samples, {N_PROTEINS} proteins, {len(psm_filtered)} PSMs)",
             fontsize=13, fontweight="bold")

ax1 = fig.add_subplot(gs[0, 0])
colors_v = ["#E91E63" if (r["padj"]<0.05 and r["log2FC"]>1) else
            "#2196F3" if (r["padj"]<0.05 and r["log2FC"]<-1) else "gray"
            for _, r in da_df.iterrows()]
ax1.scatter(da_df["log2FC"], -np.log10(da_df["p_value"]+1e-10), c=colors_v, alpha=0.5, s=12)
ax1.axhline(-np.log10(0.05), color="red", ls="--", lw=1)
ax1.axvline(1, color="gray", ls="--", lw=0.8); ax1.axvline(-1, color="gray", ls="--", lw=0.8)
for _, row in sig_df.head(5).iterrows():
    ax1.annotate(row["protein"], (row["log2FC"], -np.log10(row["p_value"]+1e-10)), fontsize=6)
ax1.set_xlabel("log2FC"); ax1.set_ylabel("-log10(p)")
ax1.set_title(f"Volcano Plot ({len(sig_df)} significant)")

ax2 = fig.add_subplot(gs[0, 1])
top30_idx = [PROTEIN_NAMES.index(p) for p in sig_df.head(30)["protein"] if p in PROTEIN_NAMES]
if top30_idx:
    hm = quant_matrix[top30_idx, :]
    hm_z = (hm - hm.mean(axis=1, keepdims=True)) / (hm.std(axis=1, keepdims=True) + 1e-8)
    im2 = ax2.imshow(hm_z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="Z-score")
    ax2.set_yticks(range(len(top30_idx)))
    ax2.set_yticklabels([PROTEIN_NAMES[i][:12] for i in top30_idx], fontsize=5)
    ax2.set_xticks(range(N_SAMPLES))
    ax2.set_xticklabels(["C"+str(i%3+1) if i>=3 else "K"+str(i%3+1) for i in range(N_SAMPLES)], fontsize=8)
ax2.set_title("Top Differential Proteins")

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(psm_df["hyperscore"], bins=50, color="#9C27B0", alpha=0.8, edgecolor="white")
ax3.axvline(3.5, color="red", ls="--", lw=1.5, label="Threshold=3.5")
ax3.set_xlabel("Hyperscore"); ax3.set_ylabel("PSM count")
ax3.set_title(f"PSM Score Distribution ({len(psm_filtered)} passing)")
ax3.legend(fontsize=8)

ax4 = fig.add_subplot(gs[1, 0])
colors_go = ["#E91E63" if p < 0.05 else "#9E9E9E" for p in go_df["p_value"]]
ax4.barh(range(len(go_df)), -np.log10(go_df["p_value"].values+1e-10)[::-1],
         color=colors_go[::-1], alpha=0.8)
ax4.set_yticks(range(len(go_df)))
ax4.set_yticklabels(go_df["term"].values[::-1], fontsize=8)
ax4.axvline(-np.log10(0.05), color="red", ls="--", lw=1)
ax4.set_xlabel("-log10(p)"); ax4.set_title("GO Term Enrichment")

ax5 = fig.add_subplot(gs[1, 1])
n_quant = [(quant_matrix[:, s] > 0).sum() for s in range(N_SAMPLES)]
colors_s = ["#2196F3"]*3 + ["#E91E63"]*3
ax5.bar(range(N_SAMPLES), n_quant, color=colors_s, alpha=0.8)
ax5.set_xticks(range(N_SAMPLES))
ax5.set_xticklabels(["Case"+str(i%3+1) if i>=3 else "Ctrl"+str(i%3+1) for i in range(N_SAMPLES)], fontsize=8)
ax5.set_ylabel("Proteins quantified"); ax5.set_title("Proteins per Sample")

ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
items = [
    ("Samples", f"{N_SAMPLES} (3 case, 3 ctrl)"),
    ("Proteins in DB", str(N_PROTEINS)),
    ("Total PSMs", str(len(psm_df))),
    ("PSMs passing filter", str(len(psm_filtered))),
    ("Proteins identified", str(len(identified))),
    ("Differential (FDR<0.05)", str(len(sig_df))),
    ("Top protein", sig_df.iloc[0]["protein"] if len(sig_df) else "N/A"),
    ("Enriched GO terms", str((go_df["p_value"]<0.05).sum())),
    ("Runtime", f"{time.time()-t0:.0f}s"),
]
y = 0.97
ax6.text(0.05, y, "Summary", fontsize=11, fontweight="bold", transform=ax6.transAxes)
for label, val in items:
    y -= 0.09
    ax6.text(0.05, y, label, fontsize=8, transform=ax6.transAxes, color="#555")
    ax6.text(0.62, y, val, fontsize=8, fontweight="bold", transform=ax6.transAxes)

plt.savefig(f"{OUT}/proteomics_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

summary = {
    "n_samples": N_SAMPLES, "n_proteins": N_PROTEINS,
    "n_psms_total": len(psm_df), "n_psms_filtered": len(psm_filtered),
    "n_proteins_identified": int(len(identified)),
    "n_differential": int(len(sig_df)),
    "top_protein": sig_df.iloc[0]["protein"] if len(sig_df) else "N/A",
    "top_log2fc": float(sig_df.iloc[0]["log2FC"]) if len(sig_df) else 0,
    "n_enriched_go": int((go_df["p_value"]<0.05).sum()),
    "runtime_seconds": round(time.time()-t0, 1),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[ProteomicsEngine] Done in {summary['runtime_seconds']:.0f}s")
print(json.dumps(summary, indent=2))
