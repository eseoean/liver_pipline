# Liver SMILES ablation tests

- Run ID: `20260422_liver_smiles_ablation_v1`
- Fold policy: 4-fold random/drug/scaffold

## Conclusion

Unlike HNSC, the no-drop/zero-SMILES policy did not improve the liver benchmark.
The existing valid-SMILES baseline was better on all three 4-fold checks:
random 0.8502 vs 0.8056, drug-group 0.5225 vs 0.4163, and scaffold-group 0.5145 vs 0.4082.
For liver, keep the current valid-SMILES-only training input as the stronger baseline.

## legacy_drop_smiles_baseline_4fold
- Input shape: [3396, 5011]
- Feature count: 4998
- Drugs: 243
- Valid SMILES drugs: 243
- Invalid SMILES pairs kept: 0

### legacy_drop_smiles_baseline_4fold / drug_group_4fold
- weighted_top3_ensemble: Spearman 0.5225, RMSE 2.1810
- xgboost: Spearman 0.4993, RMSE 2.2237
- randomforest: Spearman 0.4964, RMSE 2.2238
- extratrees: Spearman 0.4866, RMSE 2.2202
- residual_mlp: Spearman 0.4848, RMSE 2.1951

### legacy_drop_smiles_baseline_4fold / random_4fold
- lightgbm: Spearman 0.8502, RMSE 1.1495
- weighted_top3_ensemble: Spearman 0.8499, RMSE 1.1569
- xgboost: Spearman 0.8460, RMSE 1.1679
- flat_mlp: Spearman 0.8129, RMSE 1.2869
- randomforest: Spearman 0.8060, RMSE 1.2961

### legacy_drop_smiles_baseline_4fold / scaffold_group_4fold
- weighted_top3_ensemble: Spearman 0.5145, RMSE 2.2514
- extratrees: Spearman 0.4817, RMSE 2.2854
- randomforest: Spearman 0.4694, RMSE 2.3061
- flat_mlp: Spearman 0.4623, RMSE 2.3084
- residual_mlp: Spearman 0.4419, RMSE 2.3283

## legacy_all_drugs_zero_smiles
- Input shape: [4164, 5001]
- Feature count: 4988
- Drugs: 295
- Valid SMILES drugs: 243
- Invalid SMILES pairs kept: 768

### legacy_all_drugs_zero_smiles / drug_group_4fold
- weighted_top3_ensemble: Spearman 0.4163, RMSE 2.2607
- extratrees: Spearman 0.4071, RMSE 2.2813
- randomforest: Spearman 0.3957, RMSE 2.2875
- xgboost: Spearman 0.3778, RMSE 2.3088
- lightgbm: Spearman 0.3765, RMSE 2.3611

### legacy_all_drugs_zero_smiles / random_4fold
- lightgbm: Spearman 0.8056, RMSE 1.2877
- xgboost: Spearman 0.7914, RMSE 1.3282
- weighted_top3_ensemble: Spearman 0.7874, RMSE 1.3359
- randomforest: Spearman 0.7097, RMSE 1.5227
- extratrees: Spearman 0.7028, RMSE 1.5727

### legacy_all_drugs_zero_smiles / scaffold_group_4fold
- weighted_top3_ensemble: Spearman 0.4082, RMSE 2.2999
- extratrees: Spearman 0.3754, RMSE 2.3617
- flat_mlp: Spearman 0.3705, RMSE 2.3349
- randomforest: Spearman 0.3654, RMSE 2.3559
- residual_mlp: Spearman 0.3375, RMSE 2.4264
