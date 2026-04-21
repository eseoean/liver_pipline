# Liver Performance Gap Analysis

## Summary

The LIHC/HCC liver pipeline was rebuilt with the same BRCA-style input family used for the lung comparison: numeric slim features, strong context one-hot features, and SMILES SVD features. The final liver model remains below the BRCA/lung reference performance under drug-level GroupCV, but the follow-up BRCA downsample experiment shows that reduced cell-line and pair diversity materially contributes to the gap.

## Liver Final Setting

| Item | Value |
|---|---:|
| Cell lines | 15 |
| SMILES-valid drugs | 243 |
| Model rows | 3,396 |
| Model features | 4,998 |
| Strong context one-hot features | 32 |
| SMILES SVD features | 64 |
| GroupCV weighted ensemble Spearman | 0.5158 |
| GroupCV weighted ensemble RMSE | 2.1912 |

The liver random-CV models still reached approximately 0.81-0.84 Spearman for the main LightGBM, XGBoost, and ResidualMLP tracks. This means the model can learn within-distribution response patterns, but generalization to held-out drugs is more limited.

## BRCA Downsample Control

To test whether the liver performance drop was mainly caused by the small number of LIHC/HCC cell lines, the BRCA exact slim + strong context + SMILES input was re-evaluated with the same model family and then downsampled to 15 high-coverage BRCA cell lines.

| Experiment | Cell lines | Pairs | Drugs | Features | GroupCV ensemble Spearman | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| BRCA full, liver-like 4-model set | 29 | 6,366 | 243 | 5,625 | 0.5836 | 2.1144 |
| BRCA 15-cell-line downsample | 15 | 3,475 | 233 | 5,625 | 0.5428 | 2.2406 |
| LIHC/HCC liver final | 15 | 3,396 | 243 | 4,998 | 0.5158 | 2.1912 |

Selected BRCA downsample cell lines:

```text
BT-549, CAMA-1, EFM-19, HCC38, HCC70,
MDA-MB-157, MDA-MB-231, MDA-MB-436, MDA-MB-453, MFM-223,
CAL-120, CAL-51, HCC1428, HCC1937, HCC1954
```

## Interpretation

The BRCA downsample control reduced BRCA GroupCV Spearman from 0.5836 to 0.5428, bringing it closer to the LIHC/HCC result of 0.5158. This supports the conclusion that limited cell-line coverage and response-pair diversity have a meaningful impact on liver model performance.

The effect is not absolute. BRCA with 15 cell lines still remained slightly above liver with 15 cell lines, so the residual gap is likely explained by liver-specific factors: HCC cell-line representativeness, label noise, weaker target-response signal, and mismatch between TCGA-LIHC disease context and GDSC cell-line response.

Recommended wording for reports:

> The lower LIHC/HCC GroupCV performance is not best interpreted as a pipeline failure. A BRCA downsample control showed that reducing BRCA to a comparable 15-cell-line scale decreased GroupCV Spearman from 0.5836 to 0.5428, approaching the LIHC/HCC result of 0.5158. This indicates that limited supervised response diversity substantially contributes to the liver performance gap, although disease-specific biological heterogeneity and TCGA-to-cell-line context mismatch likely also contribute.

## Follow-Up Options Without Adding New Labels

1. Run LIHC/HCC cell-line QC and ablation to identify unstable or weakly representative cell lines.
2. Add ranking or top-sensitive classification objectives alongside `ln_IC50` regression.
3. Strengthen cell-line-specific DepMap context with expression, mutation, CNV, and pathway activity scores.
4. Reweight drug-target features by LIHC target expression, dependency, and pathway relevance.
5. Use fold/bootstrap stability as part of the final candidate ranking.

## Source Artifacts

- Liver final metrics: `outputs/model_runs/metrics_summary.json`
- Liver QC: `outputs/reports/quality_check_results.json`
- BRCA downsample summary: `/Users/skku_aws2_18/lung/artifacts/brca_downsample_15cell_groupcv_20260421/summary.json`
- BRCA full liver-like model summary: `/Users/skku_aws2_18/lung/artifacts/brca_full_liverlike_groupcv_20260421/summary.json`
