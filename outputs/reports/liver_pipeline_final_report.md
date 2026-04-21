# LIHC/HCC Drug Repurposing Pipeline Report

## Model Input
- GDSC rows: 4164
- Cell lines: 15
- Drugs: 295
- Slim input shape: [3396, 5011]
- Slim feature count: 4998
- Strong-context one-hot features: 32
- SMILES SVD features: 64

## Model Metrics
- groupcv__lightgbm: Spearman 0.4784, RMSE 2.3271
- groupcv__xgboost: Spearman 0.4894, RMSE 2.2471
- groupcv__residual_mlp: Spearman 0.4666, RMSE 2.2524
- groupcv__tabnet: Spearman 0.0798, RMSE 2.6064
- randomcv__lightgbm: Spearman 0.8446, RMSE 1.1800
- randomcv__xgboost: Spearman 0.8434, RMSE 1.1842
- randomcv__residual_mlp: Spearman 0.8124, RMSE 1.2854
- randomcv__tabnet: Spearman 0.1095, RMSE 2.5829
- groupcv weighted top3 ensemble: Spearman 0.5158, RMSE 2.1912

## Top Drugs
- #1 Paclitaxel (score=0.9442, class=liver_cancer_research_or_related)
- #2 Epirubicin (score=0.9299, class=repurposing_candidate_unclear_liver_relation)
- #3 Docetaxel (score=0.9248, class=liver_cancer_research_or_related)
- #4 MG-132 (score=0.9233, class=repurposing_candidate_unclear_liver_relation)
- #5 Topotecan (score=0.9140, class=repurposing_candidate_unclear_liver_relation)
- #6 Bortezomib (score=0.9125, class=repurposing_candidate_unclear_liver_relation)
- #7 Temsirolimus (score=0.9035, class=liver_cancer_research_or_related)
- #8 Buparlisib (score=0.9023, class=repurposing_candidate_unclear_liver_relation)
- #9 Elesclomol (score=0.8967, class=repurposing_candidate_unclear_liver_relation)
- #10 Mitoxantrone (score=0.8910, class=repurposing_candidate_unclear_liver_relation)
- #11 Camptothecin (score=0.8908, class=repurposing_candidate_unclear_liver_relation)
- #12 Dactinomycin (score=0.8897, class=repurposing_candidate_unclear_liver_relation)
- #13 Staurosporine (score=0.8851, class=repurposing_candidate_unclear_liver_relation)
- #14 Vinblastine (score=0.8524, class=repurposing_candidate_unclear_liver_relation)
- #15 Bleomycin (score=0.8423, class=repurposing_candidate_unclear_liver_relation)

## ADMET Pass Top Candidates
- MG-132 (adjusted=0.9233)
- Buparlisib (adjusted=0.9023)
- Elesclomol (adjusted=0.8967)
- Camptothecin (adjusted=0.8908)
- Staurosporine (adjusted=0.8851)
- MK-2206 (adjusted=0.8064)
- Lestaurtinib (adjusted=0.7952)
- Dihydrorotenone (adjusted=0.7777)
- Vinblastine (adjusted=0.7724)
- Pevonedistat (adjusted=0.7681)
- SN-38 (adjusted=0.7667)
- BIBR-1532 (adjusted=0.7630)
- PRT062607 (adjusted=0.7322)
- Ulixertinib (adjusted=0.7095)
- GDC0810 (adjusted=0.6934)
