# LIHC/HCC Drug Repurposing Pipeline Report

## Model Input
- GDSC rows: 4164
- Cell lines: 15
- Drugs: 295
- Slim input shape: [3396, 4914]
- Slim feature count: 4902

## Model Metrics
- groupcv__lightgbm: Spearman 0.4941, RMSE 2.3357
- groupcv__xgboost: Spearman 0.5091, RMSE 2.2492
- groupcv__residual_mlp: Spearman 0.4575, RMSE 2.2957
- groupcv__tabnet: Spearman 0.0893, RMSE 2.6119
- randomcv__lightgbm: Spearman 0.8448, RMSE 1.1780
- randomcv__xgboost: Spearman 0.8362, RMSE 1.2118
- randomcv__residual_mlp: Spearman 0.8003, RMSE 1.3342
- randomcv__tabnet: Spearman 0.0819, RMSE 2.5848
- groupcv weighted top3 ensemble: Spearman 0.5270, RMSE 2.2076

## Top Drugs
- #1 Paclitaxel (score=0.9476, class=liver_cancer_research_or_related)
- #2 Epirubicin (score=0.9239, class=repurposing_candidate_unclear_liver_relation)
- #3 Docetaxel (score=0.9237, class=liver_cancer_research_or_related)
- #4 Bortezomib (score=0.9060, class=repurposing_candidate_unclear_liver_relation)
- #5 Topotecan (score=0.9022, class=repurposing_candidate_unclear_liver_relation)
- #6 Camptothecin (score=0.8974, class=repurposing_candidate_unclear_liver_relation)
- #7 Temsirolimus (score=0.8962, class=liver_cancer_research_or_related)
- #8 Dactinomycin (score=0.8957, class=repurposing_candidate_unclear_liver_relation)
- #9 Elesclomol (score=0.8956, class=repurposing_candidate_unclear_liver_relation)
- #10 Teniposide (score=0.8890, class=repurposing_candidate_unclear_liver_relation)
- #11 Mitoxantrone (score=0.8886, class=repurposing_candidate_unclear_liver_relation)
- #12 Pevonedistat (score=0.8674, class=repurposing_candidate_unclear_liver_relation)
- #13 Bleomycin (score=0.8413, class=repurposing_candidate_unclear_liver_relation)
- #14 Lestaurtinib (score=0.8081, class=repurposing_candidate_unclear_liver_relation)
- #15 MK-2206 (score=0.8065, class=repurposing_candidate_unclear_liver_relation)

## ADMET Pass Top Candidates
- Camptothecin (adjusted=0.8974)
- Elesclomol (adjusted=0.8956)
- Pevonedistat (adjusted=0.8674)
- Lestaurtinib (adjusted=0.8081)
- MK-2206 (adjusted=0.8065)
- Staurosporine (adjusted=0.7949)
- Buparlisib (adjusted=0.7948)
- MG-132 (adjusted=0.7873)
- BIBR-1532 (adjusted=0.7764)
- SN-38 (adjusted=0.7751)
- Entinostat (adjusted=0.7452)
- GDC0810 (adjusted=0.7226)
- Alisertib (adjusted=0.6823)
- Vinblastine (adjusted=0.6808)
- IOX2 (adjusted=0.6710)
